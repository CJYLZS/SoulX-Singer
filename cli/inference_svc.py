import os
import torch
import json
import argparse
from tqdm import tqdm
import numpy as np
import soundfile as sf
from collections import OrderedDict
from omegaconf import DictConfig

from soulxsinger.utils.file_utils import load_config
from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
from soulxsinger.utils.audio_utils import load_wav


def _pitch_shift_arg(v: str):
    """--pitch_shift 接受整数半音数或字面量 "auto"（按段对齐到最近八度）。"""
    if isinstance(v, str) and v.lower() == "auto":
        return "auto"
    try:
        return int(v)
    except ValueError:
        raise argparse.ArgumentTypeError(f'--pitch_shift 需要整数或 "auto"，收到 {v!r}')


def build_model(
    model_path: str,
    config: DictConfig,
    device: str = "cuda",
    use_fp16: bool = False,
):
    """
    Build the model from the pre-trained model path and model configuration.

    Args:
        model_path (str): Path to the checkpoint file.
        config (DictConfig): Model configuration.
        device (str, optional): Device to use. Defaults to "cuda".
        use_fp16 (bool, optional): If True and device is CUDA, convert model to FP16 after load. Defaults to False.

    Returns:
        SoulXSingerSVC: The initialized model.
    """

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Please download the pretrained model and place it at the path, or set --model_path."
        )
    model = SoulXSingerSVC(config).to(device)
    print("Model initialized.")
    print("Model parameters:", sum(p.numel() for p in model.parameters()) / 1e6, "M")
    
    checkpoint = torch.load(model_path, weights_only=False, map_location="cpu")
    if "state_dict" not in checkpoint:
        raise KeyError(
            f"Checkpoint at {model_path} has no 'state_dict' key. "
            "Expected a checkpoint saved with model.state_dict()."
        )
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    
    if use_fp16 and ((isinstance(device, str) and device.startswith("cuda")) or (hasattr(device, "type") and getattr(device, "type", None) == "cuda")):
        model.half()
        model.mel.float()
        print("Model converted to FP16 (mel kept in FP32).")
    print("Model checkpoint loaded.")
    model.eval()
    model.to(device)

    return model


def process(args, config, model: torch.nn.Module):
    """Run the full inference pipeline given a data_processor and model.
    """

    os.makedirs(args.save_dir, exist_ok=True)
    pt_wav = load_wav(args.prompt_wav_path, config.audio.sample_rate).to(args.device)
    gt_wav = load_wav(args.target_wav_path, config.audio.sample_rate).to(args.device)
    pt_f0 = torch.from_numpy(np.load(args.prompt_f0_path)).unsqueeze(0).to(args.device)
    gt_f0 = torch.from_numpy(np.load(args.target_f0_path)).unsqueeze(0).to(args.device)
    # kNN retrieval pool: only loaded when kNN is on and a separate pool was given.
    knn_pool_wav = None
    if getattr(args, "knn_alpha", 0.0) > 0.0 and getattr(args, "knn_pool_wav", None):
        knn_pool_wav = load_wav(args.knn_pool_wav, config.audio.sample_rate).to(args.device)

    n_step = args.n_steps if hasattr(args, "n_steps") else config.infer.n_steps
    cfg = args.cfg if hasattr(args, "cfg") else config.infer.cfg

    with torch.no_grad():
        generated_audio, generated_shift = model.infer(
            pt_wav=pt_wav,
            gt_wav=gt_wav,
            pt_f0=pt_f0,
            gt_f0=gt_f0,
            auto_shift=args.auto_shift, 
            pitch_shift=args.pitch_shift, 
            n_steps=n_step, 
            cfg=cfg,
            use_fp16=args.use_fp16,
            max_seg_sec=getattr(args, "max_seg_sec", 30.0),
            seed=getattr(args, "seed", None),
            num_overlaps=getattr(args, "num_overlaps", 1),
            rescale_cfg=getattr(args, "rescale_cfg", 0.75),
            knn_alpha=getattr(args, "knn_alpha", 0.0),
            knn_k=getattr(args, "knn_k", 4),
            knn_pool_wav=knn_pool_wav,
        )
    generated_audio = generated_audio.squeeze().float().cpu().numpy()
    if args.pitch_shift != generated_shift:
        args.pitch_shift = generated_shift
        # print(f"Applied pitch shift of {generated_shift} semitones to match GT F0 contour.")

    sf.write(os.path.join(args.save_dir, "generated.wav"), generated_audio, config.audio.sample_rate)
    print(f"Generated audio saved to {os.path.join(args.save_dir, 'generated.wav')}")


def main(args, config):
    model = build_model(
        model_path=args.model_path,
        config=config,
        device=args.device,
        use_fp16=getattr(args, "use_fp16", False),
    )
    process(args, config, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--model_path", type=str, default='pretrained_models/soulx-singer/model.pt')
    parser.add_argument("--config", type=str, default='soulxsinger/config/soulxsinger.yaml')
    parser.add_argument("--prompt_wav_path", type=str, default='example/audio/zh_prompt.wav')
    parser.add_argument("--target_wav_path", type=str, default='example/audio/zh_target.wav')
    parser.add_argument("--prompt_f0_path", type=str, default='example/audio/zh_prompt_f0.npy')
    parser.add_argument("--target_f0_path", type=str, default='example/audio/zh_target_f0.npy')
    parser.add_argument("--save_dir", type=str, default='outputs')
    parser.add_argument("--auto_shift", action="store_true")
    # 本地补丁：除整数外接受 "auto" —— 按段对齐到最近八度（见 soulxsinger_svc.py
    # _nearest_octave_shift）。男女对唱/跨音区的歌里，全曲单一移调必然一头顾不上。
    parser.add_argument("--pitch_shift", type=_pitch_shift_arg, default=0,
                        help='移调半音数，或 "auto" 按段对齐到最近八度')
    parser.add_argument("--n_steps", type=int, default=32)
    parser.add_argument("--cfg", type=float, default=3.0)
    parser.add_argument("--rescale_cfg", type=float, default=0.75,
                        help="CFG 外推结果的 std 归一化混合比（1=全归一化，0=用原始外推；上游 0.75）")
    # 上游硬编码 30s/段（soulxsinger_svc.py:257），+num_overlaps 余量实测可达 33s，
    # 6GB 卡上最长段会爆显存换页（song_2 卡在 8/10 段 12 分钟无进展）
    parser.add_argument("--max_seg_sec", type=float, default=30.0,
                        help="SVC 单段最长秒数（6GB 显存建议 20）")
    parser.add_argument("--seed", type=int, default=None,
                        help="CFM 初始噪声 seed（None = 每次随机，设定后可复现；"
                             "方差实验/多采样选优时使用）")
    parser.add_argument("--num_overlaps", type=int, default=1,
                        help="每段往前取几个活性格作为上下文（默认 1；总窗仍受 max_seg_sec 限制）")
    parser.add_argument("--knn_alpha", type=float, default=0.0,
                        help="kNN 内容特征替换强度（0=关闭，1=完全替换；0.2-0.4 起步）")
    parser.add_argument("--knn_k", type=int, default=4,
                        help="kNN 近邻数（默认 4，同 kNN-VC）")
    parser.add_argument("--knn_pool_wav", type=str, default=None,
                        help="kNN 检索池音频（默认用 prompt_wav，受 whisper 30s 截断）。"
                             "指向完整参考人声（如 refsep2/lead.wav, 163s）可分块编码绕开 "
                             "30s 限制：匹配失败帧 9.8%%->1.9%%。检索池不进模型，故不受 30s 约束")
    parser.add_argument(
        "--fp16",
        action="store_true",
        default=False,
        help="Use FP16 inference (faster on GPU)",
    )
    args = parser.parse_args()
    args.use_fp16 = args.fp16

    config = load_config(args.config)
    main(args, config)
