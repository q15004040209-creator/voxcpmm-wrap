"""
VoxCPM2 Core Wrapper
核心封装类 - 提供简洁的 API 接口
"""

import os
import gc
from typing import Optional, Iterator, Union, List
import numpy as np

try:
    from voxcpm import VoxCPM
    VOXCPM_AVAILABLE = True
except ImportError:
    VOXCPM_AVAILABLE = False
    VoxCPM = None


class VoxCPMWrap:
    """
    VoxCPM2 Python 封装类

    支持：
    - 文本转语音（30种语言）
    - 声音设计（自然语言描述）
    - 可控语音克隆
    - 极致克隆（音频+转写）
    - 流式推理
    """

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_denoiser: bool = False,
        compile: bool = False,
        dtype: Optional[str] = None,
    ):
        """
        初始化 VoxCPMWrap

        Args:
            model_path: 模型路径或 HuggingFace 模型 ID
            device: 运行设备 ("auto", "cpu", "cuda", "mps")
            load_denoiser: 是否加载去噪器（默认 False）
            compile: 是否启用 PyTorch compile（加速但首次慢）
            dtype: 数据类型 ("float16", "bfloat16", "float32")
        """
        if not VOXCPM_AVAILABLE:
            raise ImportError(
                "voxcpmm-wrap 需要 voxcpm 库。请先安装：pip install voxcpm\n"
                "voxcpmm-wrap requires the voxcpm library. Install first: pip install voxcpm"
            )

        self.device = self._resolve_device(device)

        if dtype is None:
            dtype = "float16" if self.device.startswith("cuda") else "float32"

        self.model = VoxCPM.from_pretrained(
            model_path,
            load_denoiser=load_denoiser,
            compile=compile,
            device=self.device,
            dtype=dtype,
        )

        self.tts_model = self.model.tts_model
        self.sample_rate = self.tts_model.sample_rate

    @staticmethod
    def _resolve_device(device: str) -> str:
        """解析设备字符串"""
        if device == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    return "cuda"
                elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                    return "mps"
                else:
                    return "cpu"
            except Exception:
                return "cpu"
        return device

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str = "openbmb/VoxCPM2",
        device: str = "auto",
        load_denoiser: bool = False,
        compile: bool = False,
        dtype: Optional[str] = None,
    ):
        """
        从预训练模型创建实例（推荐方式）

        Args:
            model_name_or_path: HuggingFace 模型 ID 或本地路径
            device: 运行设备
            load_denoiser: 是否加载去噪器
            compile: 是否启用 compile
            dtype: 数据类型
        """
        return cls(
            model_path=model_name_or_path,
            device=device,
            load_denoiser=load_denoiser,
            compile=compile,
            dtype=dtype,
        )

    def generate(
        self,
        text: str,
        reference_wav_path: Optional[str] = None,
        prompt_wav_path: Optional[str] = None,
        prompt_text: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """
        生成语音

        Args:
            text: 输入文本。可在开头用括号添加声音描述，
                  如："(A young woman, warm and gentle)Hello world"
            reference_wav_path: 参考音频路径（用于语音克隆）
            prompt_wav_path: 提示音频路径（用于极致克隆）
            prompt_text: 提示音频的转写文本（用于极致克隆）
            cfg_value: CFG 强度（1.0-4.0，值越大越贴近描述）
            inference_timesteps: 推理步数（越多越精细，默认10）
            seed: 随机种子（可复现）

        Returns:
            numpy.ndarray: 音频数据，shape=(samples,), dtype=float32
        """
        generate_kwargs = {
            "text": text,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
        }

        if reference_wav_path is not None:
            generate_kwargs["reference_wav_path"] = reference_wav_path

        if prompt_wav_path is not None:
            generate_kwargs["prompt_wav_path"] = prompt_wav_path

        if prompt_text is not None:
            generate_kwargs["prompt_text"] = prompt_text

        if seed is not None:
            generate_kwargs["seed"] = seed

        wav = self.model.generate(**generate_kwargs)

        # 确保返回 float32 numpy 数组
        if hasattr(wav, "numpy"):
            wav = wav.numpy()
        elif not isinstance(wav, np.ndarray):
            wav = np.array(wav, dtype=np.float32)

        # 归一化到 [-1, 1]
        if wav.max() > 1.0 or wav.min() < -1.0:
            wav = wav / max(abs(wav.max()), abs(wav.min()))

        return wav.astype(np.float32)

    def generate_streaming(
        self,
        text: str,
        reference_wav_path: Optional[str] = None,
        cfg_value: float = 2.0,
        inference_timesteps: int = 10,
        seed: Optional[int] = None,
    ) -> Iterator[np.ndarray]:
        """
        流式生成语音（逐块返回）

        Args:
            text: 输入文本
            reference_wav_path: 参考音频路径
            cfg_value: CFG 强度
            inference_timesteps: 推理步数
            seed: 随机种子

        Yields:
            numpy.ndarray: 音频数据块
        """
        generate_kwargs = {
            "text": text,
            "cfg_value": cfg_value,
            "inference_timesteps": inference_timesteps,
        }

        if reference_wav_path is not None:
            generate_kwargs["reference_wav_path"] = reference_wav_path

        if seed is not None:
            generate_kwargs["seed"] = seed

        for chunk in self.model.generate_streaming(**generate_kwargs):
            if hasattr(chunk, "numpy"):
                chunk = chunk.numpy()
            elif not isinstance(chunk, np.ndarray):
                chunk = np.array(chunk, dtype=np.float32)
            yield chunk.astype(np.float32)

    def generate_to_file(
        self,
        output_path: str,
        text: str,
        **kwargs
    ) -> None:
        """
        直接生成并保存为 WAV 文件

        Args:
            output_path: 输出文件路径
            text: 输入文本
            **kwargs: generate() 的其他参数
        """
        import soundfile as sf

        wav = self.generate(text, **kwargs)
        sf.write(output_path, wav, self.sample_rate)

    def unload(self) -> None:
        """卸载模型，释放显存"""
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        if self.device.startswith("cuda"):
            import torch
            torch.cuda.empty_cache()

    def __repr__(self) -> str:
        return f"VoxCPMWrap(device={self.device}, sample_rate={self.sample_rate})"