🎬 AI Studio (Local Android Video Generator)
Turn your Android phone into an autonomous, offline AI video production studio. AI Studio scripts, narrates, generates visuals, crossfades scenes, burns subtitles, and exports full HD vertical (9:16 Shorts/Reels) or landscape (16:9) videos—running 100% locally on your device hardware.
⚡ Requirements
 * Android Device with a modern Snapdragon processor (e.g., Snapdragon 8 Gen 2 / Gen 3 / 8 Elite).
 * Local Dream Android app installed (used for hardware-accelerated NPU/GPU image generation).
 * Termux installed (F-Droid / GitHub release, not Google Play Store).
🚀 Quick Start (3 Steps)
Step 1: Start Local Dream
 * Open the Local Dream app.
 * Select and load your preferred image model (e.g., AnythingV5, Realistic Vision, or SD 1.5).
 * Tap Device link -> Host mode (this device runs models).
Step 2: Install AI Studio
Open Termux and run this single setup command:
git clone https://github.com/YOUR_USERNAME/ai-studio.git
cd ai-studio
chmod +x setup.sh start.sh
./setup.sh

(The setup script installs all core packages, downloads the Kokoro-82M voice models, and prepares the Qwen LLM automatically).
Step 3: Launch AI Studio
Start the studio server anytime by running:
./start.sh

Open Chrome or Samsung Internet on your phone and navigate to:
👉 http://localhost:5000
🎛️ How to Use the Studio
 * Aspect Ratio: Choose between 9:16 Vertical (Shorts/Reels) or 16:9 Landscape (YouTube).
 * Topic: Type any topic (e.g., Deep Sea Mysteries, Ancient Rome, Cyberpunk Samurai), or leave it blank to let the local LLM brainstorm an original concept.
 * Custom Script (Optional): Paste your own script or JSON scene array, or leave it blank to let the AI draft a high-retention narration hook.
 * Captions: Toggle auto-synchronized burnt-in captions.
 * Click Render Video.
All completed videos are saved directly to your phone's gallery in:
📁 /sdcard/pipeline/ (~/storage/shared/pipeline/)
🛠️ Under the Hood
 * LLM Scripting: Qwen 2.5 3B Instruct running locally via llama.cpp.
 * Voice Synthesis: Kokoro-82M ONNX running via sherpa-onnx.
 * Visual Rendering: Hardware-accelerated NPU/GPU generation via Local Dream Host API (127.0.0.1:8081).
 * Video Mastering: FFmpeg hardware-timed pan/zoom (Ken Burns), chained xfade & acrossfade transitions, and dynamic .ass subtitle burning.

