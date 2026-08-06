import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from typing import Optional, Dict, Any, List, Tuple
from contextlib import nullcontext

from soulxsinger.models.modules.vocoder import Vocoder
from soulxsinger.models.modules.decoder import CFMDecoder
from soulxsinger.models.modules.mel_transform import MelSpectrogramEncoder
from soulxsinger.models.modules.whisper_encoder import WhisperEncoder


def _autocast_if(enabled: bool):
    """Return autocast(context) if enabled else no-op context. Use: with _autocast_if(use_amp): ..."""
    return torch.amp.autocast(device_type="cuda", enabled=True) if enabled else nullcontext()

class SoulXSingerSVC(nn.Module):
    """
    SoulXSinger SVC model.
    """
    def __init__(self, config: Dict):
        super(SoulXSingerSVC, self).__init__()
        self.audio_cfg = config.audio
        enc_cfg = config.model.encoder
        cfm_cfg = config.model.flow_matching
        
        self.whisper_encoder = WhisperEncoder()
        self.f0_encoder = nn.Embedding(enc_cfg["f0_bin"], enc_cfg["f0_dim"])
        self.cfm_decoder = CFMDecoder(cfm_cfg)

        self.mel = MelSpectrogramEncoder(self.audio_cfg)
        self.vocoder = Vocoder()

    @staticmethod
    def _nearest_octave_shift(seg_f0_median: float, ref_f0_median: float) -> int:
        """
        Compute the nearest octave shift (0, ±12, ±24, ±36) to align seg_f0_median with ref_f0_median.
        Returns the shift in semitones.
        """
        if seg_f0_median <= 0 or ref_f0_median <= 0:
            return 0
        raw_shift = 12 * np.log2(ref_f0_median / seg_f0_median)
        # round to nearest multiple of 12
        octave_shift = int(round(raw_shift / 12.0)) * 12
        # clamp to ±36 semitones (±3 octaves)
        return max(-36, min(36, octave_shift))

    @staticmethod
    def f0_to_coarse(f0, f0_bin=361, f0_min=32.7031956625, f0_shift=0):
        """
        Convert continuous F0 values to discrete F0 bins (SIL and C1 - B6, 361 bins).
        args:
            f0: continuous F0 values
            f0_bin: number of F0 bins
            f0_min: minimum F0 value
            f0_shift: shift value for F0 bins
        returns:
            f0_coarse: discrete F0 bins
        """
        is_torch = isinstance(f0, torch.Tensor)
        uv_mask = f0 <= 0    

        if is_torch:  
            f0_safe = torch.maximum(f0, torch.tensor(f0_min))
            f0_cents = 1200 * torch.log2(f0_safe / f0_min)
        else:
            f0_safe = np.maximum(f0, f0_min)
            f0_cents = 1200 * np.log2(f0_safe / f0_min)

        f0_coarse = (f0_cents / 20) + 1
        
        if is_torch:
            f0_coarse = torch.round(f0_coarse).long()
            f0_coarse = torch.clamp(f0_coarse, min=1, max=f0_bin - 1)
        else:
            f0_coarse = np.rint(f0_coarse).astype(int)
            f0_coarse = np.clip(f0_coarse, 1, f0_bin - 1)

        f0_coarse[uv_mask] = 0

        if f0_shift != 0:
            if is_torch:
                voiced = f0_coarse > 0
                if voiced.any():
                    shifted = f0_coarse[voiced] + f0_shift
                    f0_coarse[voiced] = torch.clamp(shifted, 1, f0_bin - 1)
            else:
                voiced = f0_coarse > 0
                if np.any(voiced):
                    shifted = f0_coarse[voiced] + f0_shift
                    f0_coarse[voiced] = np.clip(shifted, 1, f0_bin - 1)
        
        return f0_coarse

    @staticmethod
    def build_vocal_segments(
        f0,
        f0_rate: int = 50,
        uv_frames_th: int = 5,
        min_duration_sec: float = 5.0,
        max_duration_sec: float = 30.0,
        num_overlaps: int = 1,
        ignore_silent_segments: bool = True,
    ) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
        """Build vocal segments based on F0 contour. First split by long silent runs, then merge into segments based on min and max duration constraints.
        args:
            f0: F0 contour of the audio, 1D array or tensor with shape (T,)
            f0_rate: F0 sampling rate in Hz (e.g., 50 for 20ms hop size)
            uv_frames_th: number of consecutive zero F0 frames to consider as a split point
            min_duration_sec: minimum duration of each segment in seconds
            max_duration_sec: maximum duration of each segment in seconds
            num_overlaps: number of overlapping segments to create for each non-overlapping segment (for smooth inference)
            ignore_silent_segments: whether to ignore segments that are mostly silent (e.g., > 95% zero F0)
        returns:
            overlap_segments: list of (overlap_start_sec, overlap_end_sec) for each segment, which may overlap with adjacent segments for smooth inference
            segments: list of (seg_start_sec, seg_end_sec) for each segment, which are non-overlapping and used for final merging
        """
        if isinstance(f0, torch.Tensor):
            f0_np = f0.detach().float().cpu().numpy()
        else:
            f0_np = np.asarray(f0, dtype=np.float32)
        f0_np = np.squeeze(f0_np)

        total_frames = int(f0_np.shape[0])
        if total_frames == 0:
            return [], []

        min_frames = max(1, int(round(min_duration_sec * f0_rate)))
        max_frames = max(1, int(round(max_duration_sec * f0_rate)))

        split_points = [0]      # silence split points in frame indices, starting with 0 and ending with total_frames

        def append_split_point(point: int):
            # Ensure split points are within valid range and respect max_frames constraint
            point = int(max(0, min(point, total_frames)))
            while point - split_points[-1] > max_frames:
                split_points.append(split_points[-1] + max_frames)
            if point > split_points[-1]:
                split_points.append(point)

        idx = 0
        while idx < total_frames:
            if f0_np[idx] == 0:
                run_start = idx
                while idx < total_frames and f0_np[idx] == 0:
                    idx += 1
                run_end = idx
                if (run_end - run_start) >= uv_frames_th:
                    # 静音 run 起点和终点各打一个切分点，把静音段独立出来，
                    # 否则切中点会让尾奏/间奏的无人声被夹进相邻段一起推理
                    append_split_point(run_start)
                    append_split_point(run_end)
            else:
                idx += 1
        append_split_point(total_frames)
        # print(f"Initial split points (in seconds): {[round(p / f0_rate, 2) for p in split_points]}")

        segments: List[Tuple[int, int]] = []
        overlap_segments: List[Tuple[int, int]] = []

        def active(gi: int) -> bool:
            """split_points 的第 gi 格（格 = 相邻两个切分点之间）是否有人声。"""
            start_f = split_points[gi]
            end_f = split_points[gi + 1]
            total = end_f - start_f
            if total <= 0:
                return False
            v = int(np.sum(f0_np[start_f:end_f] > 0))
            return v / total > 0.05 and v >= 10

        def append_segment(start_idx: int, end_idx: int, num_overlaps: int = num_overlaps):
            segments.append((split_points[start_idx] / f0_rate, split_points[end_idx] / f0_rate))
            # overlap 起点：往前找活性格，供 infer_segment 带前置上下文（模型靠它建立音色）。
            # 本地补丁：silence-aware 分段把静音 run 独立成格之后，start_idx-1 必然是静音格，
            # 原来的 `while ... and active(k)` 第一轮就失败 —— 实测 song_2 6/6、song_9 12/12
            # 段的上下文都是 0.00s，num_overlaps 完全是废参。现在允许跨过短静音格
            # （< min_frames，与前向合并同一判据）去够前面的活性格；长静音（间奏/尾奏）
            # 仍是硬边界。注意总长仍受 max_frames 约束（whisper 30s 硬顶），所以
            # max_seg_sec 越接近 30，可用的上下文空间越小。
            overlap_start_idx = start_idx
            k = start_idx - 1
            cnt = 0
            while k >= 0 and cnt < num_overlaps:
                if not active(k):
                    if (split_points[k + 1] - split_points[k]) < min_frames:
                        k -= 1      # 短静音格：跳过继续往前找
                        continue
                    break           # 长静音格：硬边界
                if split_points[end_idx] - split_points[k] > max_frames:
                    break
                overlap_start_idx = k
                cnt += 1
                k -= 1
            overlap_segments.append((split_points[overlap_start_idx] / f0_rate, split_points[end_idx] / f0_rate))

        # 活性优先合并：静音格（run_start/run_end 已独立切分）不进段，直接跳过；
        # 相邻活性格可跨短静音格（< min_frames）合并，但不得吞入长静音格。
        # 短活性段不再被 min_frames 强制拼长——宁可独立推理，也不带尾部静音。
        i = 0
        n_grids = len(split_points) - 1
        while i < n_grids:
            if not active(i):
                i += 1
                continue
            j = i
            while j + 1 < n_grids:
                if split_points[j + 2] - split_points[i] > max_frames:
                    break
                if active(j + 1) or (split_points[j + 2] - split_points[j + 1]) < min_frames:
                    j += 1
                else:
                    break
            append_segment(i, j + 1)
            i = j + 1

        # 兜底：静音格可能携带噪声帧导致 active 误判，这里按整段活性再滤一遍
        if ignore_silent_segments:
            filtered_idx = []
            for i, seg in enumerate(segments):
                start_frame = int(seg[0] * f0_rate)
                end_frame = int(seg[1] * f0_rate)
                total_frames = end_frame - start_frame
                voice_frames = np.sum(f0_np[start_frame:end_frame] > 0)
                if voice_frames / total_frames > 0.05 and voice_frames >= 10:
                    filtered_idx.append(i)
            overlap_segments = [overlap_segments[i] for i in filtered_idx]
            segments = [segments[i] for i in filtered_idx]

        return overlap_segments, segments
    
    def infer(
        self, 
        pt_wav: str|torch.Tensor,
        gt_wav: str|torch.Tensor,
        pt_f0: str|torch.Tensor,
        gt_f0: str|torch.Tensor,
        auto_shift=False,
        pitch_shift=0,
        n_steps=32,
        cfg=3,
        rescale_cfg=0.75,
        use_fp16=False,
        max_seg_sec=30.0,
        seed=None,
        num_overlaps=1,
        knn_alpha=0.0,
        knn_k=4,
        knn_pool_wav: str|torch.Tensor|None=None,
    ):
        """
        SVC inference pipeline. First build vocal segments based on F0 contour, then run inference for each segment and merge results.
        args:
            pt_wav: prompt waveform path or tensor
            gt_wav: target waveform path or tensor
            pt_f0: prompt F0 path or tensor
            gt_f0: target F0 path or tensor
            auto_shift: whether to automatically calculate pitch shift based on median F0 of prompt and target
            pitch_shift: manual pitch shift in semitones (overrides auto_shift if > 0).
                         Special: pitch_shift="auto" enables per-segment nearest-octave alignment.
            n_steps: number of diffusion steps for inference
            cfg: classifier-free guidance scale for inference
            rescale_cfg: how much of the CFG extrapolation to std-normalise (1.0 = fully
                  normalised/conservative, 0.0 = raw extrapolation). Upstream hard-codes 0.75.
                  Dead when cfg == 0, which skips the CFG branch entirely.
            use_fp16: if True, run in FP16 except mel extraction to save memory and speed.
            seed: base seed for the CFM initial noise. None = upstream behaviour (global RNG,
                  non-reproducible). When set, segment i uses seed + i so segments stay
                  independent while the whole run is reproducible.
            num_overlaps: how many preceding active grids to prepend as context for each
                  segment (see build_vocal_segments). Total window is still capped by
                  max_seg_sec, so raising this only helps if segments are short.
            knn_pool_wav: optional separate waveform for kNN retrieval pool (bypasses the 30s
                  whisper truncation). If None, falls back to pt_wav (current 30s-capped
                  behaviour). Typically the full clean reference (e.g. 163s refsep2/lead.wav),
                  vs pt_wav being the 28.9s prompt_ext.wav. Measured on a 163s reference:
                  frames with nn1<0.5 drop from 9.8% to 1.9% vs the 30s pool.
        """

        # handle "auto" mode: per-segment octave alignment
        auto_octave = False
        if isinstance(pitch_shift, str) and pitch_shift.lower() == "auto":
            auto_octave = True
            pitch_shift = 0  # will be overridden per segment
        
        # calculate auto pitch shift (legacy whole-song mode)
        if auto_shift and pitch_shift == 0 and not auto_octave:
            if gt_f0 is not None and pt_f0 is not None:
                gt_f0_median = torch.median(gt_f0[gt_f0 > 0])
                pt_f0_median = torch.median(pt_f0[pt_f0 > 0])
                pitch_shift = torch.round(torch.log2(pt_f0_median / gt_f0_median) * 1200 / 100).int().item()
            else:
                print("Warning: pitch_shift is True but note_pitch or f0 is None. Set f0_shift to 0.")
                pitch_shift = 0
        else:
            pitch_shift = pitch_shift

        use_fp16 = use_fp16 and pt_wav.is_cuda
        # mel is kept in fp32 (see build_model: model.mel.float() after model.half())
        pt_mel = self.mel(pt_wav.float() if pt_wav.dtype != torch.float32 else pt_wav)
        if use_fp16:
            pt_mel = pt_mel.half()
            pt_wav = pt_wav.half()
            gt_wav = gt_wav.half()
            pt_f0 = pt_f0.half()
            gt_f0 = gt_f0.half()

        # pre-compute prompt median for auto-octave mode
        pt_f0_median = None
        if auto_octave:
            pt_f0_median = torch.median(pt_f0[pt_f0 > 0]).item()

        # Build the kNN retrieval pool once, outside the segment loop: it is identical for every
        # segment and chunk-encoding a 163s reference costs ~6 whisper passes.
        knn_pool = None
        if knn_alpha > 0.0 and knn_pool_wav is not None:
            pool_wav = knn_pool_wav.half() if use_fp16 else knn_pool_wav
            with _autocast_if(use_fp16):
                knn_pool = self.encode_long(pool_wav)
            print(f"kNN pool: {knn_pool.shape[1]} frames "
                  f"({knn_pool.shape[1] / 50:.1f}s) from knn_pool_wav")

        # if target audio is less than 30 seconds, infer the whole audio
        if gt_wav.shape[-1] < max_seg_sec * self.audio_cfg.sample_rate:
            seg_shift = pitch_shift
            if auto_octave:
                seg_f0_median = torch.median(gt_f0[gt_f0 > 0]).item()
                seg_shift = self._nearest_octave_shift(seg_f0_median, pt_f0_median)
            with _autocast_if(use_fp16):
                generated_audio = self.infer_segment(
                    pt_mel=pt_mel,
                    pt_wav=pt_wav,
                    gt_wav=gt_wav,
                    pt_f0=pt_f0,
                    gt_f0=gt_f0,
                    pitch_shift=seg_shift,
                    n_steps=n_steps,
                    cfg=cfg,
                    rescale_cfg=rescale_cfg,
                    seed=seed,
                    knn_alpha=knn_alpha,
                    knn_k=knn_k,
                    knn_pool=knn_pool,
                )
            return generated_audio, seg_shift

        # if target audio is longer than 30 seconds, build vocal segments and infer each segment
        generated_audio = []

        f0_rate = self.audio_cfg.sample_rate // self.audio_cfg.hop_size
        
        overlap_segments, segments = self.build_vocal_segments(
            gt_f0,
            f0_rate=f0_rate,
            uv_frames_th=10,
            min_duration_sec=min(15.0, max_seg_sec / 2),
            max_duration_sec=max_seg_sec,
            num_overlaps=num_overlaps,
        )
        if len(segments) == 0:
            segments = [(0.0, gt_wav.shape[-1] / self.audio_cfg.sample_rate)]
            overlap_segments = [(0.0, gt_wav.shape[-1] / self.audio_cfg.sample_rate)]

        generated_audio = torch.zeros_like(gt_wav)
        applied_shifts = []
        for idx in tqdm(range(len(segments)), total=len(segments), desc="Inferring segments (SVC)", dynamic_ncols=True):
            overlap_start_sec, overlap_end_sec = overlap_segments[idx]
            seg_start_sec, seg_end_sec = segments[idx]

            wav_start = int(round(overlap_start_sec * self.audio_cfg.sample_rate))
            wav_end = int(round(overlap_end_sec * self.audio_cfg.sample_rate))
            f0_start = int(round(overlap_start_sec * f0_rate))
            f0_end = int(round(overlap_end_sec * f0_rate))

            wav_start = max(0, min(wav_start, gt_wav.shape[-1]))
            wav_end = max(wav_start, min(wav_end, gt_wav.shape[-1]))
            f0_start = max(0, min(f0_start, gt_f0.shape[-1]))
            f0_end = max(f0_start, min(f0_end, gt_f0.shape[-1]))

            segment_gt_wav = gt_wav[:, wav_start:wav_end]
            segment_gt_f0 = gt_f0[:, f0_start:f0_end]

            # per-segment nearest-octave alignment: decide the shift from the F0 of the
            # part that actually lands in the output (segments[idx]), not the overlap
            # context, otherwise a low-pitched lead-in would bias the whole segment.
            seg_shift = pitch_shift
            if auto_octave:
                keep_f0 = gt_f0[:, int(round(seg_start_sec * f0_rate)): int(round(seg_end_sec * f0_rate))]
                voiced = keep_f0[keep_f0 > 0]
                if voiced.numel() > 0:
                    seg_shift = self._nearest_octave_shift(torch.median(voiced).item(), pt_f0_median)
                else:
                    seg_shift = 0
            applied_shifts.append((seg_start_sec, seg_end_sec, seg_shift))

            with _autocast_if(use_fp16):
                segment_generated_audio = self.infer_segment(
                    pt_mel=pt_mel,
                    pt_wav=pt_wav,
                    gt_wav=segment_gt_wav,
                    pt_f0=pt_f0,
                    gt_f0=segment_gt_f0,
                    pitch_shift=seg_shift,
                    n_steps=n_steps,
                    cfg=cfg,
                    rescale_cfg=rescale_cfg,
                    seed=None if seed is None else seed + idx,
                    knn_alpha=knn_alpha,
                    knn_k=knn_k,
                    knn_pool=knn_pool,
                )

            segment_start = int(round(seg_start_sec * self.audio_cfg.sample_rate))
            segment_end = int(round(seg_end_sec * self.audio_cfg.sample_rate))
            segment_generated_audio = segment_generated_audio[segment_start - wav_start: segment_end - wav_start]

            generated_audio[:, segment_start:segment_end] = segment_generated_audio

        if auto_octave:
            print(f"Auto-octave alignment (prompt median {pt_f0_median:.1f}Hz):")
            for s, e, sh in applied_shifts:
                print(f"  {s:7.1f}-{e:7.1f}s  shift {sh:+3d} st")
            shifts_only = [sh for _, _, sh in applied_shifts]
            pitch_shift = max(set(shifts_only), key=shifts_only.count) if shifts_only else 0

        return generated_audio, pitch_shift

    @staticmethod
    def _knn_replace_content(gt_content_feat, pt_content_feat, alpha, k, valid_pt_frames=None):
        """Move the source's whisper content features toward the prompt speaker via kNN retrieval.

        Rationale (measured in this workspace): whisper features carry a source-singer timbre
        residual, which is why *lowering* `cfg` raises speaker similarity — amplifying the
        condition amplifies the residual. Deleting the residual does not work: mean-shifting the
        features toward the prompt made things worse at every alpha, because the decoder was
        trained with the residual present (and with `cfg_drop_prob: 0.2`, only the true condition
        and exactly-zero are in-distribution; a constant offset is neither).

        This instead *replaces* the residual with the target's. Each source frame is blended with
        the mean of its k nearest neighbours among the prompt's own whisper frames, so every
        substituted vector is a convex combination of genuine whisper-base final-layer outputs and
        stays on the trained manifold. Same idea as kNN-VC / RVC's faiss index, applied to the
        condition tensor SoulX already computes.

        Args:
            gt_content_feat: (1, T_gt, D) source features, the thing being edited.
            pt_content_feat: (1, T_pt, D) prompt features, used as the matching pool.
            alpha: blend weight. 0.0 returns gt untouched; 1.0 is full replacement. Partial
                   interpolation degrades gracefully where the prompt lacks a phone, which matters
                   because the pool is ~25s of speech being used to reconstruct singing.
            k: neighbours to average (kNN-VC uses 4).
            valid_pt_frames: optional int, restrict the pool to the first N prompt frames so
                   right-padding introduced by the caller cannot be matched against.
        Returns:
            (1, T_gt, D) tensor, same dtype/device as gt_content_feat.
        """
        if alpha <= 0.0:
            return gt_content_feat

        pool = pt_content_feat[0]
        if valid_pt_frames is not None:
            pool = pool[:max(1, min(int(valid_pt_frames), pool.shape[0]))]
        if pool.shape[0] == 0:
            return gt_content_feat

        src = gt_content_feat[0]
        # cosine matching in float32: the features can arrive in fp16 under --use_fp16, and
        # normalize+matmul in half loses enough precision to shuffle near-ties in the topk.
        g = F.normalize(src.float(), dim=-1)
        p = F.normalize(pool.float(), dim=-1)
        k_eff = max(1, min(int(k), pool.shape[0]))
        idx = (g @ p.T).topk(k=k_eff, dim=-1).indices           # (T_gt, k)
        knn = pool[idx].to(torch.float32).mean(dim=1)           # (T_gt, D)
        out = (1.0 - alpha) * src.float() + alpha * knn
        return out.to(gt_content_feat.dtype).unsqueeze(0)

    def encode_long(self, wav, chunk_sec=28.0):
        """Encode arbitrarily long audio by chunking, bypassing whisper's 30s truncation.

        `WhisperEncoder.encode` hard-truncates to WHISPER_MEL_FRAMES=3000 (30s). That limit is
        mandatory for the in-context prompt (`pt_mel` must stay aligned with `pt_content_feat`),
        but a kNN *retrieval pool* never enters the model, so it can be built from the full
        reference. Measured on a 163s reference: frames with no usable neighbour (cos < 0.5)
        drop from 9.8% to 1.9% versus the 30s-truncated pool.

        Returns (1, N, D); each chunk contributes only its real frames, never right-padding.
        """
        sr = self.audio_cfg.sample_rate
        step = int(chunk_sec * sr)
        n = wav.shape[-1]
        if n <= step:
            return self.whisper_encoder.encode(wav, sr=sr)
        parts = []
        for s in range(0, n, step):
            seg = wav[..., s:s + step]
            if seg.shape[-1] < sr * 0.5:   # a <0.5s tail gives unreliable whisper output
                break
            f = self.whisper_encoder.encode(seg, sr=sr)
            n_real = min(int(np.ceil(seg.shape[-1] / sr * 50)), f.shape[1])
            parts.append(f[:, :n_real, :])
        return torch.cat(parts, 1)

    def infer_segment(self, pt_mel, pt_wav, gt_wav, pt_f0, gt_f0, pitch_shift=0, n_steps=32, cfg=3,
                      rescale_cfg=0.75, seed=None, knn_alpha=0.0, knn_k=4, knn_pool=None):
        len_prompt_mel = pt_mel.shape[1]
        pt_f0 = F.pad(pt_f0, (0, 0, 0, max(0, len_prompt_mel - pt_f0.shape[1])))[:, :len_prompt_mel]

        f0_course_pt = self.f0_to_coarse(pt_f0)
        f0_course_gt = self.f0_to_coarse(gt_f0, f0_shift=pitch_shift * 5)
        f0_course = torch.cat([f0_course_pt, f0_course_gt], 1)

        pt_content_feat = self.whisper_encoder.encode(pt_wav, sr=self.audio_cfg.sample_rate)
        gt_content_feat = self.whisper_encoder.encode(gt_wav, sr=self.audio_cfg.sample_rate)
        t_pt, t_gt = f0_course_pt.shape[1], f0_course_gt.shape[1]
        # remember how many prompt frames whisper actually produced, so the kNN pool below
        # excludes the zero-padding that the next line may append.
        n_pt_real = min(pt_content_feat.shape[1], t_pt)
        pt_content_feat = F.pad(pt_content_feat, (0, 0, 0, max(0, t_pt - pt_content_feat.shape[1])))[:, :t_pt, :]
        gt_content_feat = F.pad(gt_content_feat, (0, 0, 0, max(0, t_gt - gt_content_feat.shape[1])))[:, :t_gt, :]

        if knn_alpha > 0.0:
            # The retrieval pool is decoupled from the in-context prompt: `knn_pool` (whole
            # reference, chunk-encoded) when supplied, else the 30s-capped prompt features.
            if knn_pool is not None:
                pool, n_pool = knn_pool, None
            else:
                pool, n_pool = pt_content_feat, n_pt_real
            gt_content_feat = self._knn_replace_content(
                gt_content_feat, pool, alpha=knn_alpha, k=knn_k,
                valid_pt_frames=n_pool,
            )

        content_feat = torch.cat([pt_content_feat, gt_content_feat], 1)

        f0_feat = self.f0_encoder(f0_course)
        features = content_feat + f0_feat
        
        gt_decoder_inp = features[:, len_prompt_mel:, :]
        pt_decoder_inp = features[:, :len_prompt_mel, :]

        generated_mel = self.cfm_decoder.reverse_diffusion(
            pt_mel,
            pt_decoder_inp,
            gt_decoder_inp,
            n_timesteps=n_steps,
            cfg=cfg,
            rescale_cfg=rescale_cfg,
            seed=seed,
        )
        
        generated_audio = self.vocoder(generated_mel.transpose(1, 2)[0:1, ...])
        generated_audio = generated_audio.squeeze().float()

        # cut or pad to match gt_wav length
        if generated_audio.shape[-1] > gt_wav.shape[-1]:
            generated_audio = generated_audio[:gt_wav.shape[-1]]
        elif generated_audio.shape[-1] < gt_wav.shape[-1]:
            generated_audio = F.pad(generated_audio, (0, gt_wav.shape[-1] - generated_audio.shape[-1]))

        return generated_audio