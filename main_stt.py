# source Pvenv/bin/activate

import pyaudio
import numpy as np
from scipy import signal
from openwakeword.model import Model
import webrtcvad
import subprocess
import wave
import time
import requests

# ===== 路徑已統一改成 ~/RPi4-1/ 底下 =====
WHISPER_BIN = "/home/leslie/RPi4-1/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/leslie/RPi4-1/whisper.cpp/models/ggml-base.bin"
THRESHOLD = 0.67


SERVER_URL = "http://192.168.51.158:5000/process"   # 換成你Ubuntu PC的實際區網IP,確認server_dev.py有這個路由
SERVER_TIMEOUT = 30

VAD_AGGRESSIVENESS = 2       # 0-3,越大越嚴格(越不容易誤判環境音為語音)
VAD_FRAME_MS = 30            # webrtcvad只接受10/20/30ms的音框
SILENCE_LIMIT_MS = 1000      # 連續靜音超過這個時間就判斷講完話
MAX_RECORD_SECONDS = 10      # 最長錄音上限,避免使用者不說話卡死
MIN_SPEECH_MS = 300          # 至少要偵測到這麼多語音才算有效開始說話

# ===== 載入喚醒詞模型 =====
model = Model(
    wakeword_model_paths=["/home/leslie/RPi4-1/oww_models/alexa_v0.1.onnx"],
    melspec_onnx_model_path="/home/leslie/RPi4-1/oww_models/melspectrogram.onnx",
    embedding_onnx_model_path="/home/leslie/RPi4-1/oww_models/embedding_model.onnx"
)

######################################################################################################
# 尋找耳機()
######################################################################################################
# ===== 麥克風裝置關鍵字,依實際裝置名稱調整 =====
MIC_NAME_KEYWORD = "JBL"

def find_input_device_index(audio, name_keyword):
    """依裝置名稱動態尋找輸入裝置index,避免每次插拔USB裝置index改變的問題"""
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if name_keyword in info['name'] and info['maxInputChannels'] > 0:
            return i
    raise RuntimeError(f"找不到名稱包含 '{name_keyword}' 的輸入裝置,請檢查麥克風是否已插上")

# ===== 尋找耳機 =====
audio = pyaudio.PyAudio()
mic_index = find_input_device_index(audio, MIC_NAME_KEYWORD)
print(f">>> 使用麥克風裝置 index: {mic_index}")
######################################################################################################

# ===== 載入VAD =====
vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)

RATE_IN = 48000
RATE_OUT = 16000
CHUNK = 3840  # 對應16000Hz時的1280 samples(喚醒詞偵測用)

# VAD音框大小:16kHz下30ms = 480 samples,換算成48kHz輸入端要讀多少samples
VAD_FRAME_SAMPLES_OUT = int(RATE_OUT * VAD_FRAME_MS / 1000)   # 480 (16kHz, 給VAD判斷用)
VAD_FRAME_SAMPLES_IN = int(RATE_IN * VAD_FRAME_MS / 1000)     # 1440 (48kHz, 實際讀取用)

# 靜音/最短語音判斷,換算成幾個VAD音框
SILENCE_FRAMES_LIMIT = int(SILENCE_LIMIT_MS / VAD_FRAME_MS)
MIN_SPEECH_FRAMES = int(MIN_SPEECH_MS / VAD_FRAME_MS)
MAX_FRAMES = int(MAX_RECORD_SECONDS * 1000 / VAD_FRAME_MS)

stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE_IN,
                     input=True, frames_per_buffer=CHUNK,
                     input_device_index=mic_index)

def flush_audio_buffer():
    """清空stream裡在錄音+轉錄過程中積壓的舊音訊,避免殘留片段誤觸發喚醒詞"""
    available = stream.get_read_available()
    while available > 0:
        stream.read(min(available, CHUNK), exception_on_overflow=False)
        available = stream.get_read_available()

def record_and_transcribe():
    print("\n>>> 偵測到喚醒詞!開始錄音(講完話會自動停止)...")

    frames_16k = []       # 存已經resample成16kHz的音框,直接拿來寫wav
    silence_count = 0
    speech_count = 0
    triggered = False     # 是否已經偵測到足夠的語音,開始正式倒數靜音
    total_frames = 0

    while total_frames < MAX_FRAMES:
        raw_data = stream.read(VAD_FRAME_SAMPLES_IN, exception_on_overflow=False)
        audio_np = np.frombuffer(raw_data, dtype=np.int16)

        # 降採樣到16kHz給VAD跟whisper用
        resampled = signal.resample(audio_np, VAD_FRAME_SAMPLES_OUT).astype(np.int16)
        frames_16k.append(resampled.tobytes())

        is_speech = vad.is_speech(resampled.tobytes(), RATE_OUT)

        if is_speech:
            speech_count += 1
            silence_count = 0
        else:
            silence_count += 1

        # 先確認有講到足夠的話,才開始用靜音判斷是否結束
        if not triggered and speech_count >= MIN_SPEECH_FRAMES:
            triggered = True

        if triggered and silence_count >= SILENCE_FRAMES_LIMIT:
            break

        total_frames += 1

    print(">>> 錄音結束,存檔中...")

    raw_audio = b"".join(frames_16k)
    resampled = np.frombuffer(raw_audio, dtype=np.int16)

    wav_path = "/tmp/command.wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE_OUT)
        wf.writeframes(resampled.tobytes())

    print(">>> 呼叫whisper.cpp轉文字...")
    start = time.time()
    result = subprocess.run(
        [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_path, "-l", "zh", "-nt"],
        capture_output=True, text=True
    )
    elapsed = time.time() - start

    text = result.stdout.strip()
    print(f">>> 辨識結果 (耗時 {elapsed:.1f}秒):")
    print(f"    {text}")

    if not text:
        print(">>> 沒有辨識到文字內容,跳過送出")
        print(">>> 重新開始監聽喚醒詞...\n")
        return

    # ===== POST請求送出文字給 server_dev.py,等待處理完的package回應 =====
    print(">>> 傳送文字給 server_dev.py ...")
    try:
        resp = requests.post(
            SERVER_URL,
            json={"usr_sentence": text},
            timeout=SERVER_TIMEOUT
        )
        print(f">>> server已回應 (HTTP {resp.status_code})")

    except requests.exceptions.RequestException as e:
        print(f">>> 呼叫 server_dev.py 失敗: {e}")

    time.sleep(2)
    print(">>> 重新開始監聽喚醒詞...\n")


print("開始監聽喚醒詞(說 'alexa')...")
try:
    while True:
        raw_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        resampled = signal.resample(raw_data, 1280).astype(np.int16)

        prediction = model.predict(resampled)
        for wakeword, score in prediction.items():
            if score > THRESHOLD:
                record_and_transcribe()
                # 清一下buffer避免立刻重複觸發
                model.reset()
                flush_audio_buffer()
except KeyboardInterrupt:
    print("\n結束監聽")
    stream.stop_stream()
    stream.close()
    audio.terminate()