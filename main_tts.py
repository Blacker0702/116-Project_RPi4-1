# source Pvenv/bin/activate

from flask import Flask, request, jsonify
import subprocess
import time

# ===== 路徑已統一改成 ~/RPi4-1/ 底下 =====
PIPER_BIN = "/home/leslie/RPi4-1/piper/piper"
PIPER_MODEL = "/home/leslie/RPi4-1/piper/piper_model/zh_CN-huayan-medium.onnx"
TTS_OUTPUT_WAV = "/tmp/response.wav"

PORT = 2953

app = Flask(__name__)


@app.route("/tts", methods=["POST"])
def tts():
    data = request.get_json(force=True, silent=True)
    if not data or "text" not in data:
        return jsonify({"status": "error", "message": "缺少text欄位"}), 400

    text = data["text"].strip()
    if not text:
        return jsonify({"status": "error", "message": "text為空"}), 400

    print(f">>> 收到TTS請求: {text}")

    try:
        start = time.time()
        # 呼叫Piper生成語音
        subprocess.run(
            [PIPER_BIN, "--model", PIPER_MODEL, "--output_file", TTS_OUTPUT_WAV],
            input=text, text=True, check=True, capture_output=True
        )
        # 播放
        subprocess.run(["aplay", TTS_OUTPUT_WAV], check=True, capture_output=True)
        elapsed = time.time() - start
        print(f">>> 播放完成 (耗時 {elapsed:.1f}秒)")

        return jsonify({"status": "ok", "elapsed": elapsed}), 200

    except subprocess.CalledProcessError as e:
        error_msg = e.stderr if hasattr(e, "stderr") else str(e)
        print(f">>> TTS或播放失敗: {error_msg}")
        return jsonify({"status": "error", "message": str(error_msg)}), 500


if __name__ == "__main__":
    print(f">>> TTS server啟動,監聽 0.0.0.0:{PORT} /tts")
    app.run(host="0.0.0.0", port=PORT)