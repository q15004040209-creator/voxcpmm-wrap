"""
基础示例 - 文本转语音
Basic Example - Text-to-Speech
"""

from voxcpmm_wrap import VoxCPMWrap
import soundfile as sf

# 加载模型（自动选择设备）
model = VoxCPMWrap.from_pretrained(
    "openbmb/VoxCPM2",
    device="auto",       # 自动选择: cuda > mps > cpu
    load_denoiser=False,
)

# 基础 TTS
wav = model.generate(
    text="VoxCPM2 是当前推荐的多语言语音合成模型，支持30种语言。",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("basic_tts.wav", wav, model.sample_rate)
print(f"✅ 已保存: basic_tts.wav (采样率: {model.sample_rate}Hz)")

# 英文示例
wav_en = model.generate(
    text="Hello! VoxCPM2 brings studio-quality multilingual speech synthesis to everyone.",
    cfg_value=2.0,
    inference_timesteps=10,
)
sf.write("basic_tts_en.wav", wav_en, model.sample_rate)
print("✅ 已保存: basic_tts_en.wav")

model.unload()  # 释放显存