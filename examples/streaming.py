"""
流式推理示例 - 实时流式语音生成
Streaming Example - Real-time streaming speech generation
"""

from voxcpmm_wrap import VoxCPMWrap
import soundfile as sf
import numpy as np

model = VoxCPMWrap.from_pretrained("openbmb/VoxCPM2", device="auto")

print("🔄 开始流式生成...")
chunks = []

for i, chunk in enumerate(model.generate_streaming(
    text="Streaming text to speech is easy with VoxCPM! This feature is perfect for real-time applications.",
    cfg_value=2.0,
    inference_timesteps=10,
)):
    chunks.append(chunk)
    print(f"  收到音频块 {i+1}: shape={chunk.shape}, dtype={chunk.dtype}")

# 合并所有音频块
wav = np.concatenate(chunks)
sf.write("streaming_output.wav", wav, model.sample_rate)
print(f"\n✅ 流式生成完成: streaming_output.wav")
print(f"   总采样数: {len(wav)}, 时长: {len(wav)/model.sample_rate:.2f}s")

model.unload()