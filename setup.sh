#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "=============================================="
echo "          AI Studio - 1-Click Setup           "
echo "=============================================="

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/output"
mkdir -p "$BASE_DIR/assets"
mkdir -p "$BASE_DIR/bin"
mkdir -p ~/storage/shared/pipeline

# 1. Non-Interactive Storage Check
if [ ! -d "$HOME/storage" ]; then
    echo "[*] Setting up Android storage access..."
    termux-setup-storage
    sleep 2
else
    echo "[*] Android storage verified (~/storage already linked)."
fi

# 2. Dependencies (no pip upgrade)
echo "[*] Ensuring required packages are installed..."
pkg update -y
pkg install -y python git ffmpeg fontconfig curl tar bzip2 clang cmake

echo "[*] Installing Python server dependencies..."
pip install flask requests

# 3. Sherpa-ONNX TTS Engine
if [ ! -f "$BASE_DIR/bin/sherpa-onnx-offline-tts" ]; then
    EXISTING_SHERPA=$(find "$HOME" -maxdepth 4 -name "sherpa-onnx-offline-tts" -type f 2>/dev/null | head -n 1 || true)
    if [ -n "$EXISTING_SHERPA" ] && [ -f "$EXISTING_SHERPA" ]; then
        echo "[*] Found existing Sherpa-ONNX binary at: $EXISTING_SHERPA"
        cp "$EXISTING_SHERPA" "$BASE_DIR/bin/"
    else
        echo "[*] Downloading Sherpa-ONNX TTS engine (~150MB)..."
        cd "$BASE_DIR"
        curl -f -L -o sherpa.tar.bz2 "https://sourceforge.net/projects/sherpa-onnx.mirror/files/v1.13.8/sherpa-onnx-v1.13.8-android-aarch64-termux-static.tar.bz2/download"
        tar -xvf sherpa.tar.bz2
        cp sherpa-onnx-*-android-aarch64-termux-static/bin/sherpa-onnx-offline-tts "$BASE_DIR/bin/"
        rm -rf sherpa.tar.bz2 sherpa-onnx-*-android-aarch64-termux-static
    fi
    chmod +x "$BASE_DIR/bin/sherpa-onnx-offline-tts"
fi

# 4. Kokoro-82M Voice Bundle
KOKORO_DIR="$BASE_DIR/models/kokoro-en-v0_19"
if [ ! -f "$KOKORO_DIR/model.onnx" ]; then
    EXISTING_KOKORO=$(find "$HOME" -maxdepth 4 -type d -name "kokoro-en-v0_19" 2>/dev/null | head -n 1 || true)
    if [ -n "$EXISTING_KOKORO" ] && [ -f "$EXISTING_KOKORO/model.onnx" ]; then
        echo "[*] Found existing Kokoro models at: $EXISTING_KOKORO"
        cp -r "$EXISTING_KOKORO" "$BASE_DIR/models/"
    else
        echo "[*] Downloading Kokoro-82M Voice Bundle (~350MB)..."
        cd "$BASE_DIR/models"
        curl -f -L -o kokoro.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
        tar -xvf kokoro.tar.bz2
        rm -f kokoro.tar.bz2
    fi
fi

# 5. Qwen 2.5 3B GGUF Model
LLM_PATH="$BASE_DIR/models/qwen2.5-3b-instruct-q4_k_m.gguf"
if [ ! -f "$LLM_PATH" ]; then
    EXISTING_QWEN=$(find "$HOME" -maxdepth 4 -name "qwen2.5-3b-instruct-q4_k_m.gguf" -type f 2>/dev/null | head -n 1 || true)
    if [ -n "$EXISTING_QWEN" ] && [ -f "$EXISTING_QWEN" ]; then
        echo "[*] Found existing Qwen GGUF model at: $EXISTING_QWEN"
        cp "$EXISTING_QWEN" "$LLM_PATH"
    else
        echo "[*] Downloading Qwen 2.5 3B Model (~1.9GB)..."
        curl -f -L -o "$LLM_PATH" "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
    fi
fi

# 6. llama-cli Binary
if [ ! -f "$BASE_DIR/bin/llama-cli" ]; then
    EXISTING_LLAMA=$(find "$HOME" -maxdepth 4 -name "llama-cli" -type f 2>/dev/null | head -n 1 || true)
    if [ -n "$EXISTING_LLAMA" ] && [ -f "$EXISTING_LLAMA" ]; then
        echo "[*] Found existing llama-cli at: $EXISTING_LLAMA"
        cp "$EXISTING_LLAMA" "$BASE_DIR/bin/"
    else
        echo "[*] Building llama-cli from source..."
        cd "$BASE_DIR"
        git clone --depth 1 https://github.com/ggerganov/llama.cpp
        cd llama.cpp
        cmake -B build -DGGML_OPENMP=OFF
        cmake --build build --config Release -j$(nproc) --target llama-cli
        cp build/bin/llama-cli "$BASE_DIR/bin/"
        rm -rf llama.cpp
    fi
    chmod +x "$BASE_DIR/bin/llama-cli"
fi

echo ""
echo "=============================================="
echo "[✔] AI Studio Setup Finished Successfully!"
echo "    Start the studio anytime by running: ./start.sh"
echo "=============================================="
