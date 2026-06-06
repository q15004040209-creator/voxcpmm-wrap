"""
VoxCPM2 Web Demo
启动方式：python -m voxcpmm_wrap.app --port 8880
"""

import os
import argparse
import base64
import tempfile
import uuid
from pathlib import Path

try:
    from flask import Flask, request, jsonify, send_file, Response
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

from .core import VoxCPMWrap


def create_app(model_path="openbmb/VoxCPM2", device="auto"):
    """创建 Flask 应用"""
    if not FLASK_AVAILABLE:
        raise ImportError("Web demo 需要 Flask: pip install flask")

    app = Flask(__name__, static_folder="static")
    app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024  # 50MB

    # 懒加载模型
    model = None

    def get_model():
        nonlocal model
        if model is None:
            model = VoxCPMWrap.from_pretrained(model_path, device=device)
        return model

    @app.route("/")
    def index():
        return """
<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="UTF-8">
<title>VoxCPM2 Web Demo</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f0f1a; color: #e0e0e0; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 40px 20px; }
  h1 { font-size: 2rem; margin-bottom: 8px; background: linear-gradient(135deg, #667eea, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .subtitle { color: #888; margin-bottom: 32px; font-size: 0.95rem; }
  .card { background: #1a1a2e; border-radius: 16px; padding: 32px; width: 100%; max-width: 640px; box-shadow: 0 8px 32px rgba(0,0,0,0.4); }
  label { display: block; font-weight: 600; margin-bottom: 8px; color: #a0a0c0; font-size: 0.9rem; }
  textarea { width: 100%; height: 100px; background: #16213e; border: 1px solid #2a2a4a; border-radius: 10px; color: #e0e0e0; padding: 14px; font-size: 1rem; resize: vertical; outline: none; transition: border 0.2s; }
  textarea:focus { border-color: #667eea; }
  textarea::placeholder { color: #666; }
  .hint { font-size: 0.78rem; color: #666; margin-top: 6px; }
  .row { display: flex; gap: 16px; margin-top: 20px; }
  .field { flex: 1; }
  input[type="file"] { display: none; }
  .file-btn { background: #16213e; border: 1px dashed #2a2a4a; border-radius: 10px; padding: 12px; text-align: center; cursor: pointer; color: #888; font-size: 0.85rem; transition: border 0.2s, color 0.2s; }
  .file-btn:hover { border-color: #667eea; color: #667eea; }
  .file-name { color: #667eea; font-size: 0.82rem; margin-top: 4px; }
  .slider-row { margin-top: 20px; }
  .slider-label { display: flex; justify-content: space-between; font-size: 0.85rem; color: #a0a0c0; margin-bottom: 6px; }
  input[type="range"] { width: 100%; accent-color: #667eea; }
  button { width: 100%; margin-top: 24px; padding: 14px; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border: none; border-radius: 10px; font-size: 1rem; font-weight: 600; cursor: pointer; transition: opacity 0.2s; }
  button:hover { opacity: 0.88; }
  button:disabled { opacity: 0.5; cursor: not-allowed; }
  .status { margin-top: 16px; text-align: center; color: #667eea; font-size: 0.9rem; min-height: 24px; }
  audio { width: 100%; margin-top: 16px; display: none; border-radius: 8px; }
  .mode-tabs { display: flex; gap: 8px; margin-bottom: 20px; }
  .mode-tab { flex: 1; padding: 8px; text-align: center; background: #16213e; border: 1px solid #2a2a4a; border-radius: 8px; cursor: pointer; font-size: 0.85rem; color: #888; transition: all 0.2s; }
  .mode-tab.active { background: #667eea; border-color: #667eea; color: white; }
  .clone-fields { display: none; }
</style>
</head>
<body>
  <h1>🎙️ VoxCPM2 Web Demo</h1>
  <p class="subtitle">无分词器多语言 TTS · 声音设计 · 语音克隆</p>

  <div class="card">
    <div class="mode-tabs">
      <div class="mode-tab active" data-mode="tts">文本转语音</div>
      <div class="mode-tab" data-mode="design">声音设计</div>
      <div class="mode-tab" data-mode="clone">语音克隆</div>
    </div>

    <label>输入文本 / Text Input</label>
    <textarea id="textInput" placeholder="输入要合成的文本... / Enter text to synthesize..."></textarea>
    <p class="hint">💡 声音设计模式：在文本开头用括号描述声音，如：(A young woman, warm and gentle)Hello world</p>

    <div class="clone-fields">
      <label style="margin-top:16px;">参考音频 / Reference Audio</label>
      <label class="file-btn" for="refAudio">
        📎 点击上传参考音频（用于克隆音色）
        <div class="file-name" id="refFileName">未选择文件</div>
      </label>
      <input type="file" id="refAudio" accept="audio/*">
    </div>

    <div class="slider-row">
      <div class="slider-label"><span>CFG 强度</span><span id="cfgVal">2.0</span></div>
      <input type="range" id="cfgSlider" min="0.5" max="4.0" step="0.1" value="2.0">
    </div>
    <div class="slider-row">
      <div class="slider-label"><span>推理步数</span><span id="stepsVal">10</span></div>
      <input type="range" id="stepsSlider" min="4" max="20" step="1" value="10">
    </div>

    <button id="generateBtn" onclick="generate()">🚀 生成语音</button>
    <div class="status" id="status"></div>
    <audio id="audioPlayer" controls></audio>
  </div>

<script>
const modeTabs = document.querySelectorAll('.mode-tab');
const cloneFields = document.querySelector('.clone-fields');
let currentMode = 'tts';

modeTabs.forEach(tab => {
  tab.addEventListener('click', () => {
    modeTabs.forEach(t => t.classList.remove('active'));
    tab.classList.add('active');
    currentMode = tab.dataset.mode;
    cloneFields.style.display = currentMode === 'clone' ? 'block' : 'none';
  });
});

document.getElementById('cfgSlider').addEventListener('input', e => {
  document.getElementById('cfgVal').textContent = e.target.value;
});
document.getElementById('stepsSlider').addEventListener('input', e => {
  document.getElementById('stepsVal').textContent = e.target.value;
});

const refInput = document.getElementById('refAudio');
const refFileName = document.getElementById('refFileName');
refInput.addEventListener('change', () => {
  refFileName.textContent = refInput.files[0] ? refInput.files[0].name : '未选择文件';
});

async function generate() {
  const btn = document.getElementById('generateBtn');
  const status = document.getElementById('status');
  const player = document.getElementById('audioPlayer');

  const text = document.getElementById('textInput').value.trim();
  if (!text) { alert('请输入文本 / Please enter text'); return; }

  btn.disabled = true;
  status.textContent = '⏳ 正在生成...';
  player.style.display = 'none';

  const formData = new FormData();
  formData.append('text', text);
  formData.append('cfg_value', document.getElementById('cfgSlider').value);
  formData.append('inference_timesteps', document.getElementById('stepsSlider').value);
  formData.append('mode', currentMode);

  if (currentMode === 'clone' && refInput.files[0]) {
    formData.append('reference_audio', refInput.files[0]);
  }

  try {
    const resp = await fetch('/api/generate', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) { status.textContent = '❌ ' + data.error; return; }
    player.src = 'data:audio/wav;base64,' + data.audio;
    player.style.display = 'block';
    status.textContent = '✅ 生成完成';
  } catch (e) {
    status.textContent = '❌ 生成失败: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}
</script>
</body>
</html>
        """

    @app.route("/api/generate", methods=["POST"])
    def api_generate():
        try:
            text = request.form.get("text", "")
            cfg_value = float(request.form.get("cfg_value", 2.0))
            inference_timesteps = int(request.form.get("inference_timesteps", 10))
            mode = request.form.get("mode", "tts")

            generate_kwargs = {
                "text": text,
                "cfg_value": cfg_value,
                "inference_timesteps": inference_timesteps,
            }

            if mode == "clone" and "reference_audio" in request.files:
                f = request.files["reference_audio"]
                suffix = Path(f.filename).suffix or ".wav"
                tmp_path = tempfile.gettempdir() + f"/voxcpm_ref_{uuid.uuid4().hex}{suffix}"
                f.save(tmp_path)
                generate_kwargs["reference_wav_path"] = tmp_path

            model = get_model()
            wav = model.generate(**generate_kwargs)

            # 转 base64
            import soundfile as sf
            import io

            buf = io.BytesIO()
            sf.write(buf, wav, model.sample_rate, format="WAV")
            b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

            return jsonify({"audio": b64})

        except Exception as e:
            import traceback
            traceback.print_exc()
            return jsonify({"error": str(e)}), 500

    return app


def main():
    parser = argparse.ArgumentParser(description="VoxCPM2 Web Demo")
    parser.add_argument("--port", type=int, default=8880, help="端口 / Port")
    parser.add_argument("--host", default="0.0.0.0", help="地址 / Host")
    parser.add_argument("--model", default="openbmb/VoxCPM2", help="模型名称 / Model name")
    parser.add_argument("--device", default="auto", help="设备 / Device")
    args = parser.parse_args()

    app = create_app(model_path=args.model, device=args.device)
    print(f"🚀 VoxCPM2 Web Demo 启动中... 访问 http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    main()