# RPi4-1 語音助理(喚醒詞 + STT + TTS)

Raspberry Pi 4 上執行的語音互動前端:偵測喚醒詞 → 錄音 → 語音轉文字 → 傳送到後端分析 → 語音合成回覆播放。

## 系統架構

```
麥克風輸入
   ↓
openWakeWord(喚醒詞偵測)
   ↓
webrtcvad(語音活動偵測,判斷講話起訖)
   ↓
whisper.cpp(語音轉文字)
   ↓
PC後台
   ↓
Piper TTS(文字轉語音)→ 本機播放
```

## 系統需求

- Raspberry Pi 4(建議 4GB RAM 以上)
- Raspberry Pi OS(或其他 Debian-based Linux)
- Python 3.10+
- USB 麥克風(程式透過裝置名稱關鍵字自動搜尋,預設關鍵字為 `JBL`,需依實際裝置調整)
- 喇叭或音源輸出裝置

## 安裝步驟

### 1. 建立專案目錄與虛擬環境

```bash
mkdir -p ~/RPi4-1
cd ~/RPi4-1
python3 -m venv Pvenv
source Pvenv/bin/activate
```

### 2. 安裝系統套件

```bash
sudo apt update
sudo apt install -y portaudio19-dev alsa-utils git build-essential cmake
```

### 3. 安裝 Python 套件 (pip install -r requirements.txt)

> **已知相容性問題**:較新版本的 `setuptools`(81 以上)已移除 `pkg_resources` 模組,而 `webrtcvad` 仍依賴它,會導致 `ModuleNotFoundError: No module named 'pkg_resources'`。請先安裝限定版本的 `setuptools`,或改用下方替代套件 `webrtcvad-wheels`。

```bash
pip install "setuptools<81"
pip install pyaudio numpy scipy openwakeword webrtcvad requests flask
```

或使用對新版 Python 相容性較好的替代套件:
```bash
pip install pyaudio numpy scipy openwakeword webrtcvad-wheels requests flask
```

### 4. 取得 openWakeWord 模型

openWakeWord 套件本身內建預訓練模型,安裝套件後即可從套件目錄複製,不需另外下載:

```bash
mkdir -p ~/RPi4-1/oww_models
cp Pvenv/lib/python3.*/site-packages/openwakeword/resources/models/alexa_v0.1.onnx ~/RPi4-1/oww_models/
cp Pvenv/lib/python3.*/site-packages/openwakeword/resources/models/melspectrogram.onnx ~/RPi4-1/oww_models/
cp Pvenv/lib/python3.*/site-packages/openwakeword/resources/models/embedding_model.onnx ~/RPi4-1/oww_models/
```

### 5. 編譯 whisper.cpp

```bash
cd ~/RPi4-1
git clone https://github.com/ggerganov/whisper.cpp
cd whisper.cpp
cmake -B build
cmake --build build --config Release
bash ./models/download-ggml-model.sh base
```

編譯完成後確認:
```bash
ls ~/RPi4-1/whisper.cpp/build/bin/whisper-cli
ls ~/RPi4-1/whisper.cpp/models/ggml-base.bin
```
### 6. 安裝 Piper TTS

```bash
cd ~/RPi4-1
wget https://github.com/rhasspy/piper/releases/latest/download/piper_linux_aarch64.tar.gz
tar -xzf piper_linux_aarch64.tar.gz

mkdir -p ~/RPi4-1/piper/piper_model

cd ~/RPi4-1/piper/piper_model
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium/zh_CN-huayan-medium.onnx.json
```

## 設定

執行前請確認並依實際環境調整以下檔案內的設定:

**`main_stt.py`**

| 變數 | 說明 |
|---|---|
| `WHISPER_BIN` / `WHISPER_MODEL` | whisper.cpp 執行檔與模型路徑 |
| `MIC_NAME_KEYWORD` | 麥克風裝置名稱關鍵字,需依 `lsusb` 或實際裝置名稱調整 |
| `THRESHOLD` | 喚醒詞觸發門檻(0~1),越高越不易誤觸發,但也可能較不敏感 |
| `SERVER_URL` | 後端分析伺服器的 API 位址 |

**`main_tts.py`**

| 變數 | 說明 |
|---|---|
| `PIPER_BIN` / `PIPER_MODEL` | Piper 執行檔與語音模型路徑 |
| `PORT` | TTS server 監聽埠號,預設 `2953` |

## 執行方式

需開啟兩個獨立程序:

```bash
# 終端機 1:啟動 TTS server
python3 main_tts.py

# 終端機 2:啟動喚醒詞偵測 + STT
python3 main_stt.py
```

對麥克風說出喚醒詞(預設模型為 `alexa`)後開始錄音,偵測到靜音會自動結束錄音並轉文字送出。

### 測試 TTS 是否正常

```bash
curl -X POST http://localhost:2953/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "測試語音輸出"}'
```

## 已知限制

- 目前 STT 結果傳送到後端後,尚未串接「後端回覆 → 自動呼叫 TTS 播放」的完整閉環,此部分需另外整合
- 未內建降噪(denoiser)處理,環境噪音較大時可能影響 VAD 判斷與辨識準確率
- 麥克風裝置辨識依賴裝置名稱關鍵字比對,更換麥克風型號需同步修改設定

## License

（依你的專案需求填寫,例如 MIT License）
