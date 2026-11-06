#!/data/data/com.termux/files/usr/bin/bash
set -e

echo "=============================================="
echo "          AI Studio - 1-Click Setup           "
echo "=============================================="

# 1. Request Android Storage Permission
echo "[*] Setting up storage access..."
termux-setup-storage
sleep 2

# 2. Update and Install Core Dependencies
echo "[*] Installing required packages (Python, FFmpeg, Fontconfig)..."
pkg update -y
pkg install -y python git ffmpeg fontconfig curl tar

pip install --upgrade pip
pip install flask requests

# 3. Create Necessary Directories
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$BASE_DIR/models"
mkdir -p "$BASE_DIR/output"
mkdir -p "$BASE_DIR/assets"
mkdir -p ~/storage/shared/pipeline

# 4. Install Sherpa-ONNX Static Binary (Prebuilt for Android aarch64)
if [ ! -f "$BASE_DIR/bin/sherpa-onnx-offline-tts" ]; then
    echo "[*] Downloading Sherpa-ONNX TTS engine..."
    mkdir -p "$BASE_DIR/bin"
    cd "$BASE_DIR"
    curl -L -o sherpa.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.10.35/sherpa-onnx-v1.10.35-android-aarch64-termux-static.tar.bz2"
    tar -xvf sherpa.tar.bz2
    cp sherpa-onnx-*-android-aarch64-termux-static/bin/sherpa-onnx-offline-tts "$BASE_DIR/bin/"
    rm -rf sherpa.tar.bz2 sherpa-onnx-*-android-aarch64-termux-static
    chmod +x "$BASE_DIR/bin/sherpa-onnx-offline-tts"
fi

# 5. Download Kokoro-82M Voice Bundle
KOKORO_DIR="$BASE_DIR/models/kokoro-en-v0_19"
if [ ! -f "$KOKORO_DIR/model.onnx" ]; then
    echo "[*] Downloading Kokoro-82M Voice Models (~350MB)..."
    cd "$BASE_DIR/models"
    curl -L -o kokoro.tar.bz2 "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-en-v0_19.tar.bz2"
    tar -xvf kokoro.tar.bz2
    rm -f kokoro.tar.bz2
fi

# 6. Download Qwen 2.5 3B Instruct GGUF
LLM_PATH="$BASE_DIR/models/qwen2.5-3b-instruct-q4_k_m.gguf"
if [ ! -f "$LLM_PATH" ]; then
    echo "[*] Downloading Qwen 2.5 3B Brainstorming Model (~1.9GB)..."
    curl -L -o "$LLM_PATH" "https://huggingface.co/Qwen/Qwen2.5-3B-Instruct-GGUF/resolve/main/qwen2.5-3b-instruct-q4_k_m.gguf"
fi

# 7. Check or Build llama.cpp
if ! command -v llama-cli &> /dev/null && [ ! -f "$BASE_DIR/bin/llama-cli" ]; then
    echo "[*] Installing llama.cpp CLI..."
    pkg install -y clang cmake
    cd "$BASE_DIR"
    if [ ! -d "llama.cpp" ]; then
        git clone --depth 1 https://github.com/ggerganov/llama.cpp
    fi
    cd llama.cpp
    cmake -B build -DGGML_OPENMP=OFF
    cmake --build build --config Release -j$(nproc) --target llama-cli
    mkdir -p "$BASE_DIR/bin"
    cp build/bin/llama-cli "$BASE_DIR/bin/"
fi

echo ""
echo "=============================================="
echo "[✔] AI Studio Setup Finished Successfully!"
echo "    Start the server anytime by running: ./start.sh"
echo "=============================================="
