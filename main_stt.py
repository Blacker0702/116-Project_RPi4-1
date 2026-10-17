

# 給我看的：找耳機 -> 定義變數 -> 載入模型跟參數 -> MAIN PROGRAM (辨識喚醒詞 -> VAD+STT -> 撈資料 -> 上傳後台 -> 釋放資源)

import pyaudio
import numpy as np
from scipy import signal
from openwakeword.model import Model
import webrtcvad
import subprocess
import wave
import time
import requests
import serial     # 💡 UART 通訊
import pickle     # 💡 解析序列化串列
import struct     # 💡 解析長度標頭
import json

# SERVER_URL = "http://100.102.30.33:5678/webhook-test/STT2Agent"
SERVER_URL = "http://100.102.41.67:5678/webhook-test/STT2Agent"

SERVER_TIMEOUT = 30

######################################################################################################
# 尋找耳機()
######################################################################################################
MIC_NAME_KEYWORD = "JBL"

def find_input_device_index(audio, name_keyword):
    """依裝置名稱動態尋找輸入裝置 index"""
    for i in range(audio.get_device_count()):
        info = audio.get_device_info_by_index(i)
        if name_keyword in info['name'] and info['maxInputChannels'] > 0:
            return i
    raise RuntimeError(f"找不到名稱包含 '{name_keyword}' 的輸入裝置, 請檢查麥克風是否已插上")

audio = pyaudio.PyAudio()
mic_index = find_input_device_index(audio, MIC_NAME_KEYWORD)
print(f">>> 使用麥克風裝置 index: {mic_index}")
######################################################################################################


# ===== 喚醒詞模型初始化 =====
model = Model(
    wakeword_model_paths=["/home/leslie/RPi4-1/oww_models/alexa_v0.1.onnx"],
    melspec_onnx_model_path="/home/leslie/RPi4-1/oww_models/melspectrogram.onnx",
    embedding_onnx_model_path="/home/leslie/RPi4-1/oww_models/embedding_model.onnx"
)
THRESHOLD = 0.67

# ===== webrtcvad初始化 =====

RATE_IN = 48000
RATE_OUT = 16000
CHUNK = 3840

VAD_AGGRESSIVENESS = 2       # 0-3, 越大越嚴格
VAD_FRAME_MS = 30            # webrtcvad 只接受 10/20/30ms 的音框
SILENCE_LIMIT_MS = 1000      # 連續靜音超過這個時間就判斷講完話
MAX_RECORD_SECONDS = 10      # 最長錄音上限
MIN_SPEECH_MS = 300          # 至少要偵測到這麼多語音才算有效開始說話

VAD_FRAME_SAMPLES_OUT = int(RATE_OUT * VAD_FRAME_MS / 1000)
VAD_FRAME_SAMPLES_IN = int(RATE_IN * VAD_FRAME_MS / 1000)

SILENCE_FRAMES_LIMIT = int(SILENCE_LIMIT_MS / VAD_FRAME_MS)
MIN_SPEECH_FRAMES = int(MIN_SPEECH_MS / VAD_FRAME_MS)
MAX_FRAMES = int(MAX_RECORD_SECONDS * 1000 / VAD_FRAME_MS)

stream = audio.open(format=pyaudio.paInt16, channels=1, rate=RATE_IN,
                     input=True, frames_per_buffer=CHUNK,
                     input_device_index=mic_index)

vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)


# ===== Whisper初始化 =====
WHISPER_BIN = "/home/leslie/RPi4-1/whisper.cpp/build/bin/whisper-cli"
WHISPER_MODEL = "/home/leslie/RPi4-1/whisper.cpp/models/ggml-base.bin"


# ===== UART 初始化 =====
# 建議兩台 RPi 的 baudrate 都設為 115200 確保高速傳輸穩定
ser = serial.Serial('/dev/serial0', baudrate=115200, timeout=1)


# ====================================================================================================
# 💡 VAD+STT
# ====================================================================================================
def record_and_transcribe():
    print("\n>>> 偵測到喚醒詞! 開始錄音(講完話會自動停止)...")

    frames_16k = []       
    silence_count = 0
    speech_count = 0
    triggered = False     
    total_frames = 0

    while total_frames < MAX_FRAMES:
        raw_data = stream.read(VAD_FRAME_SAMPLES_IN, exception_on_overflow=False)
        audio_np = np.frombuffer(raw_data, dtype=np.int16)

        resampled = signal.resample(audio_np, VAD_FRAME_SAMPLES_OUT).astype(np.int16)
        frames_16k.append(resampled.tobytes())

        is_speech = vad.is_speech(resampled.tobytes(), RATE_OUT)

        if is_speech:
            speech_count += 1
            silence_count = 0
        else:
            silence_count += 1

        if not triggered and speech_count >= MIN_SPEECH_FRAMES:
            triggered = True

        if triggered and silence_count >= SILENCE_FRAMES_LIMIT:
            break

        total_frames += 1

    print(">>> 錄音結束, 存檔中...")

    raw_audio = b"".join(frames_16k)
    resampled = np.frombuffer(raw_audio, dtype=np.int16)

    wav_path = "/tmp/command.wav"
    with wave.open(wav_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(RATE_OUT)
        wf.writeframes(resampled.tobytes())

    print(">>> 呼叫 whisper.cpp 轉文字...")
    start = time.time()
    result = subprocess.run(
        [WHISPER_BIN, "-m", WHISPER_MODEL, "-f", wav_path, "-l", "zh", "-nt"],
        capture_output=True, text=True
    )
    elapsed = time.time() - start

    text = result.stdout.strip()
    print(f">>> 辨識結果 (耗時 {elapsed:.1f} 秒):")
    
    return text  # 💡 回傳識別出來的文字內容


# ====================================================================================================
# 💡 撈取影樣辨識與重量感測資料（暫時）
# ====================================================================================================
def get_latest_uart_data():
    print(">>> [UART] 正在撈取發送端最新的串列資料...")
    f_name, f_value = "unknown", 0

    ser.reset_input_buffer()   # 丟掉 STT 期間積壓的舊資料
    ser.readline()             # 第一行可能是從中間切入的半包,丟掉
    line = ser.readline()      # 下一行一定是完整的一包

    if not line.endswith(b'\n'):
        print(">>> ⚠️ [UART] 未在時限內收到完整封包,請確認 RPi4-2 是否正常運作。")
        f_name = "RPi4-2_Error"
        f_value = -9999.9
        return f_name, f_value

    try:
        data = json.loads(line.decode('utf-8'))
        f_name, f_value = data[0], data[1]
        print(f">>> [UART] 成功即時撈取 -> 食物名稱: {f_name}, 重量: {f_value}")
    except (ValueError, IndexError) as e:   # JSON 錯誤與解碼錯誤都屬於 ValueError
        print(f">>> ❌ [UART] 解析失敗: {e}, 原始內容={line!r}")

    return f_name, f_value

# ====================================================================================================
# 💡 資料上傳後台
# ====================================================================================================
def upload_to_backend(text, f_name, f_value):
    print(">>> 傳送文字與 UART 資料給 N8N webhook ...")
    payload = {
        "usr_sentence": text,
        "f_name": f_name,
        "f_value": f_value
    }
    
    try:
        resp = requests.post(SERVER_URL, json=payload, timeout=SERVER_TIMEOUT)
        print(f">>> server 已回應 (HTTP {resp.status_code})")
    except requests.exceptions.RequestException as e:
        print(f">>> 呼叫 server_dev.py 失敗: {e}")


# ====================================================================================================
# 💡 資料釋放
# ====================================================================================================
def flush_audio_buffer():
    """清空 stream 裡積壓的舊音訊，避免殘留片段誤觸發喚醒詞"""
    available = stream.get_read_available()
    while available > 0:
        stream.read(min(available, CHUNK), exception_on_overflow=False)
        available = stream.get_read_available()


# ====================================================================================================
# MAIN_PROGRAM
# ====================================================================================================

print("開始監聽喚醒詞('alexa')...")

try:
    while True:
        # KWS喚醒詞辨識
        raw_data = np.frombuffer(stream.read(CHUNK, exception_on_overflow=False), dtype=np.int16)
        resampled = signal.resample(raw_data, 1280).astype(np.int16)

        prediction = model.predict(resampled)
        for wakeword, score in prediction.items():
            if score > THRESHOLD:


                # 1. 執行錄音並取得 STT 文字結果
                recognized_text = record_and_transcribe()
                
                # recognized_text = "你看到了什麼？"
                # recognized_text = "我今天吃了塊蛋糕"

                print(f">>> {recognized_text}")

                f_name, f_value = get_latest_uart_data()
                upload_to_backend(recognized_text, f_name, f_value)


                # 3. 重置狀態，清空快取，重新開始監聽下一次喚醒
                model.reset()
                flush_audio_buffer()
                print(">>> 重新開始監聽喚醒詞...\n")
                
except KeyboardInterrupt:
    print("\n結束監聽")
    stream.stop_stream()
    stream.close()
    audio.terminate()