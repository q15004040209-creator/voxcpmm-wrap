"""
声音设计示例 - 从自然语言描述创建全新声音
Voice Design Example - Create brand-new voices from text descriptions
"""

from voxcpmm_wrap import VoxCPMWrap
import soundfile as sf

model = VoxCPMWrap.from_pretrained("openbmb/VoxCPM2", device="auto")

# 用括号在文本开头描述声音特征
# (性别, 年龄, 语调, 情感, 语速等)

voices = [
    ("A young woman, warm and gentle", "Hello, welcome to our demo!"),
    ("An older man, deep and authoritative", "This is a deep, commanding voice."),
    ("A cheerful kid, excited and playful", "Wow, this is so much fun!"),
    ("A professional news anchor", "Here is the latest news update."),
    ("A soft-spoken elderly grandmother", "Come sit by the fire, dear."),
]

for i, (voice_desc, text) in enumerate(voices, 1):
    full_text = f"({voice_desc}){text}"
    wav = model.generate(
        text=full_text,
        cfg_value=2.5,           # 较高的 CFG 值让声音描述更明显
        inference_timesteps=10,
    )
    sf.write(f"voice_design_{i:02d}.wav", wav, model.sample_rate)
    print(f"✅ [{i}/{len(voices)}] {voice_desc}: voice_design_{i:02d}.wav")

print("\n🎨 5种声音设计完成！")
model.unload()