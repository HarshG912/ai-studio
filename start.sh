#!/data/data/com.termux/files/usr/bin/bash
termux-wake-lock
cd "$(dirname "$0")"

echo "Checking Local Dream Host Mode..."
if ! curl -s http://127.0.0.1:8808/status | grep -q "running"; then
    echo "========================================================="
    echo " [!] ATTENTION: Local Dream is not running in Host Mode!"
    echo " 1. Open the Local Dream app on your phone."
    echo " 2. Load any model (e.g. AnythingV5, RealisticVision)."
    echo " 3. Tap 'Host mode', then 'Black screen (stay awake)'."
    echo "========================================================="
    echo ""
fi

python server.py
