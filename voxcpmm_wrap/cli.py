"""
VoxCPM2 CLI 工具
用法：
    voxcpmm-wrap design  --text "Hello" --output out.wav
    voxcpmm-wrap clone   --text "Hello" --reference-audio voice.wav --output out.wav
    voxcpmm-wrap batch   --input input.txt --output-dir outputs/
    voxcpmm-wrap --help
"""

import argparse
import sys
import os
from pathlib import Path

try:
    from voxcpmm_wrap import VoxCPMWrap
except ImportError:
    from voxcpmm_wrap.core import VoxCPMWrap


def cmd_design(args):
    """声音设计命令"""
    model = VoxCPMWrap.from_pretrained(args.model, device=args.device)
    model.generate_to_file(
        output_path=args.output,
        text=args.text,
        cfg_value=args.cfg,
        inference_timesteps=args.steps,
    )
    print(f"✅ 声音设计完成: {args.output}")


def cmd_clone(args):
    """语音克隆命令"""
    model = VoxCPMWrap.from_pretrained(args.model, device=args.device)

    generate_kwargs = dict(
        text=args.text,
        cfg_value=args.cfg,
        inference_timesteps=args.steps,
    )

    if args.reference_audio:
        generate_kwargs["reference_wav_path"] = args.reference_audio

    if args.prompt_audio:
        generate_kwargs["prompt_wav_path"] = args.prompt_audio

    if args.prompt_text:
        generate_kwargs["prompt_text"] = args.prompt_text

    wav = model.generate(**generate_kwargs)

    import soundfile as sf
    sf.write(args.output, wav, model.sample_rate)
    print(f"✅ 语音克隆完成: {args.output}")


def cmd_batch(args):
    """批量处理命令"""
    model = VoxCPMWrap.from_pretrained(args.model, device=args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        output_path = output_dir / f"output_{i:04d}.wav"
        try:
            model.generate_to_file(output_path=str(output_path), text=line)
            print(f"  [{i}/{len(lines)}] ✅ {output_path.name}")
        except Exception as e:
            print(f"  [{i}/{len(lines)}] ❌ {output_path.name}: {e}")

    print(f"\n🎉 批量处理完成，共 {len(lines)} 条")


def main():
    parser = argparse.ArgumentParser(
        prog="voxcpmm-wrap",
        description="VoxCPM2 Python Wrapper CLI - 无分词器多语言 TTS 工具",
    )
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # --- design ---
    p_design = subparsers.add_parser("design", help="声音设计（无需参考音频）")
    p_design.add_argument("--text", "-t", required=True, help="输入文本")
    p_design.add_argument("--output", "-o", required=True, help="输出 WAV 文件路径")
    p_design.add_argument("--model", "-m", default="openbmb/VoxCPM2", help="模型名称或路径")
    p_design.add_argument("--device", "-d", default="auto", help="运行设备 (auto/cpu/cuda)")
    p_design.add_argument("--cfg", type=float, default=2.0, help="CFG 强度")
    p_design.add_argument("--steps", type=int, default=10, help="推理步数")

    # --- clone ---
    p_clone = subparsers.add_parser("clone", help="语音克隆（参考音频）")
    p_clone.add_argument("--text", "-t", required=True, help="输入文本")
    p_clone.add_argument("--output", "-o", required=True, help="输出 WAV 文件路径")
    p_clone.add_argument("--reference-audio", "-r", help="参考音频路径")
    p_clone.add_argument("--prompt-audio", "-p", help="提示音频路径（极致克隆）")
    p_clone.add_argument("--prompt-text", help="提示音频转写文本（极致克隆）")
    p_clone.add_argument("--model", "-m", default="openbmb/VoxCPM2", help="模型名称或路径")
    p_clone.add_argument("--device", "-d", default="auto", help="运行设备")
    p_clone.add_argument("--cfg", type=float, default=2.0, help="CFG 强度")
    p_clone.add_argument("--steps", type=int, default=10, help="推理步数")

    # --- batch ---
    p_batch = subparsers.add_parser("batch", help="批量处理")
    p_batch.add_argument("--input", "-i", required=True, help="输入文本文件（每行一段文本）")
    p_batch.add_argument("--output-dir", "-o", required=True, help="输出目录")
    p_batch.add_argument("--model", "-m", default="openbmb/VoxCPM2", help="模型名称或路径")
    p_batch.add_argument("--device", "-d", default="auto", help="运行设备")

    args = parser.parse_args()

    if args.command == "design":
        cmd_design(args)
    elif args.command == "clone":
        cmd_clone(args)
    elif args.command == "batch":
        cmd_batch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()