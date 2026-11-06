import os
import sys
import glob
import json
import re
import shutil
import base64
import threading
import requests
import subprocess
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string, send_from_directory

app = Flask(__name__)

# --- Paths & Directories ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
OUT_DIR = os.path.join(BASE_DIR, "output")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
HISTORY_FILE = os.path.join(BASE_DIR, "studio_history.json")
BG_MUSIC = os.path.join(ASSETS_DIR, "bg_music.mp3")
EXPORT_DIR = os.path.expanduser("~/storage/shared/pipeline")

LD_STATUS_URL = "http://127.0.0.1:8808/status"
LD_GENERATE_URL = "http://127.0.0.1:8081/generate"

def resolve_bin(name):
    local_p = os.path.join(BASE_DIR, "bin", name)
    if os.path.exists(local_p):
        return local_p
    which = shutil.which(name)
    if which:
        return which
    return os.path.expanduser(f"~/llama.cpp/build/bin/{name}")

LLAMA_BIN = resolve_bin("llama-cli")
LLM_MODEL = os.path.join(MODELS_DIR, "qwen2.5-3b-instruct-q4_k_m.gguf")
KOKORO_DIR = os.path.join(MODELS_DIR, "kokoro-en-v0_19")

def get_sherpa_bin():
    local_sherpa = os.path.join(BASE_DIR, "bin", "sherpa-onnx-offline-tts")
    if os.path.exists(local_sherpa):
        return local_sherpa
    matches = glob.glob(os.path.expanduser("~/sherpa-onnx-*/bin/sherpa-onnx-offline-tts"))
    if matches and os.path.exists(matches[0]):
        return matches[0]
    return os.path.expanduser("~/sherpa-onnx-v1.13.8-android-aarch64-termux-static/bin/sherpa-onnx-offline-tts")

SHERPA_TTS = get_sherpa_bin()

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(EXPORT_DIR, exist_ok=True)

# --- State ---
job_state = {
    "running": False,
    "progress": 0,
    "status": "Idle",
    "logs": [],
    "last_video": None
}

def log(msg):
    timestamp = datetime.now().strftime("%H:%M:%S")
    entry = f"[{timestamp}] {msg}"
    print(entry)
    job_state["logs"].append(entry)
    if len(job_state["logs"]) > 200:
        job_state["logs"].pop(0)

# --- LLM Execution ---
def run_local_llm(prompt: str) -> str:
    cmd = [
        LLAMA_BIN,
        "-m", LLM_MODEL,
        "-p", prompt,
        "-n", "1200",
        "--temp", "0.4",
        "--top-p", "0.9",
        "-c", "2048"
    ]
    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True
    )
    return res.stdout or ""

def run_cloud_llm(prompt: str, api_key: str, base_url: str, model_name: str) -> str:
    base = base_url.rstrip("/")
    endpoint = f"{base}/chat/completions" if base.endswith("/v1") else f"{base}/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model_name or "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.5
    }
    resp = requests.post(endpoint, json=payload, headers=headers, timeout=60)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"]

def extract_scenes_json(raw_text: str):
    candidates = []
    for match in re.finditer(r'\[\s*\{.*?\}\s*\]', raw_text, re.DOTALL):
        candidates.append(match.group(0))
    for match in re.finditer(r'\[\s*\{.*\}\s*\]', raw_text, re.DOTALL):
        candidates.append(match.group(0))

    for cand in reversed(candidates):
        clean = cand.strip('`').strip()
        if clean.startswith("json"):
            clean = clean[4:].strip()
        try:
            parsed = json.loads(clean)
            if isinstance(parsed, list) and len(parsed) > 0:
                if all("text" in item and "prompt" in item for item in parsed):
                    return parsed
        except Exception:
            continue
    return None

def sanitize_scene_text(text: str) -> str:
    clean = re.sub(
        r"^(?:in\s+)?(?:the\s+)?(?:first|second|third|fourth|fifth|final|next|\d+)(?:st|nd|rd|th)?\s+scene\s*[:\s-]*",
        "", text, flags=re.IGNORECASE
    ).strip()
    return clean[0].upper() + clean[1:] if clean else ""

# --- Image Generation ---
def generate_image_cloud(prompt: str, out_path: str, aspect_ratio: str, api_key: str, base_url: str, model_name: str):
    base = base_url.rstrip("/")
    endpoint = f"{base}/images/generations" if base.endswith("/v1") else f"{base}/v1/images/generations"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    size = "1024x1024"
    if aspect_ratio == "9:16":
        size = "1024x1792" if "dall-e-3" in model_name.lower() else "1024x1024"
    elif aspect_ratio == "16:9":
        size = "1792x1024" if "dall-e-3" in model_name.lower() else "1024x1024"

    payload = {
        "model": model_name or "dall-e-3",
        "prompt": f"{prompt}, cinematic lighting, photorealistic, 8k",
        "n": 1,
        "size": size,
        "response_format": "b64_json"
    }

    resp = requests.post(endpoint, json=payload, headers=headers, timeout=90)
    resp.raise_for_status()
    data = resp.json()
    img_data = data["data"][0]

    if "b64_json" in img_data:
        with open(out_path, "wb") as f:
            f.write(base64.b64decode(img_data["b64_json"]))
    elif "url" in img_data:
        r = requests.get(img_data["url"], timeout=60)
        with open(out_path, "wb") as f:
            f.write(r.content)
    else:
        raise ValueError("Unsupported image response format from Cloud API")

def generate_image_local_dream(prompt: str, out_path: str, aspect_ratio: str):
    status = requests.get(LD_STATUS_URL, timeout=5).json()
    width = status.get("width", 512)
    height = status.get("height", 512)

    framing = "centered vertical framing" if aspect_ratio == "9:16" else "wide cinematic composition"
    payload = {
        "prompt": f"{prompt}, {framing}, cinematic dramatic lighting, photorealistic, highly detailed, 8k",
        "negative_prompt": "blurry, low quality, distorted, watermark, deformed, text, anime, cartoon, widescreen borders",
        "size": width,
        "steps": 10,
        "cfg": 7.0
    }

    res = requests.post(LD_GENERATE_URL, json=payload, stream=True, headers={"Accept": "text/event-stream"}, timeout=120)
    img_bytes = None
    meta = {}
    for line in res.iter_lines():
        if not line: continue
        dec = line.decode("utf-8")
        if not dec.startswith("data: "): continue
        data_str = dec[6:].strip()
        if data_str == "[DONE]": break
        msg = json.loads(data_str)
        if msg.get("type") == "complete":
            img_bytes = base64.b64decode(msg["image"])
            meta = msg
            break

    if not img_bytes:
        raise RuntimeError("No image returned from Local Dream.")

    if img_bytes.startswith(b"\x89PNG") or img_bytes.startswith(b"\xff\xd8"):
        with open(out_path, "wb") as f:
            f.write(img_bytes)
    else:
        w = meta.get("width", width)
        h = meta.get("height", height)
        raw_path = out_path + ".raw"
        with open(raw_path, "wb") as f:
            f.write(img_bytes)
        subprocess.run(["ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-i", raw_path, out_path],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        if os.path.exists(raw_path):
            os.remove(raw_path)

# --- Subtitles ---
def format_ass_time(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    cs = int(round((seconds - int(seconds)) * 100))
    if cs >= 100:
        s += 1
        cs = 0
    return f"{h}:{m:02d}:{s:02d}.{cs:02d}"

def build_ass_subtitles(scenes, scene_durations, ass_filepath, aspect_ratio: str):
    res_x = 1080 if aspect_ratio == "9:16" else 1920
    res_y = 1920 if aspect_ratio == "9:16" else 1080
    font_size = 66 if aspect_ratio == "9:16" else 52
    margin_v = 480 if aspect_ratio == "9:16" else 110

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {res_x}
PlayResY: {res_y}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: AIStudio,sans-serif,{font_size},&H0000FFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,5,1,2,70,70,{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    events = []
    current_time = 0.0

    for idx, scene in enumerate(scenes):
        dur = scene_durations[idx]
        words = scene["text"].split()
        if not words:
            current_time += dur
            continue

        chunk_size = 4
        chunks = [" ".join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
        chunk_dur = dur / len(chunks)
        chunk_start = current_time

        for chunk in chunks:
            chunk_end = chunk_start + chunk_dur
            t_start = format_ass_time(chunk_start)
            t_end = format_ass_time(chunk_end)
            clean_text = chunk.replace("\\", "").replace("{", "").replace("}", "")
            events.append(f"Dialogue: 0,{t_start},{t_end},AIStudio,,0,0,0,,{clean_text}")
            chunk_start = chunk_end

        current_time += dur

    with open(ass_filepath, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events) + "\n")

# --- Core Pipeline Execution ---
def execute_pipeline(params):
    job_state["running"] = True
    job_state["progress"] = 5
    job_state["logs"] = []

    aspect_ratio = params.get("aspect_ratio", "9:16")
    topic = params.get("topic", "").strip()
    custom_script_raw = params.get("custom_script", "").strip()
    include_subtitles = params.get("subtitles", True)

    llm_mode = params.get("llm_mode", "local")
    llm_key = params.get("llm_key", "").strip()
    llm_url = params.get("llm_url", "https://api.openai.com/v1").strip()
    llm_model = params.get("llm_model", "gpt-4o-mini").strip()

    img_mode = params.get("img_mode", "local")
    img_key = params.get("img_key", "").strip()
    img_url = params.get("img_url", "https://api.openai.com/v1").strip()
    img_model = params.get("img_model", "dall-e-3").strip()

    try:
        if not topic:
            log(f"Brainstorming video topic ({'Cloud LLM: ' + llm_model if llm_mode == 'cloud' and llm_key else 'Local Qwen'})...")
            topic_prompt = "Pick ONE intriguing documentary subject across science, history, space, mysteries, or ancient civilizations. Reply with ONLY the topic title. Do not add quotes or markdown."
            if llm_mode == "cloud" and llm_key:
                raw_top = run_cloud_llm(topic_prompt, llm_key, llm_url, llm_model)
            else:
                raw_top = run_local_llm(f"<|im_start|>user\n{topic_prompt}<|im_end|>\n<|im_start|>assistant\n")
            topic = [l.strip(' "*-#') for l in raw_top.splitlines() if len(l.strip()) > 3][-1]
        log(f"Active Topic: {topic}")

        scenes = []
        if custom_script_raw:
            log("Parsing custom script...")
            try:
                parsed = json.loads(custom_script_raw)
                if isinstance(parsed, list):
                    scenes = parsed
            except Exception:
                pass

            if not scenes:
                paragraphs = [p.strip() for p in custom_script_raw.split("\n\n") if p.strip()]
                for idx, p in enumerate(paragraphs):
                    scenes.append({
                        "id": idx + 1,
                        "text": p,
                        "prompt": f"cinematic scene depicting {topic}, highly detailed, dramatic lighting, 8k"
                    })
        else:
            log(f"Drafting script ({'Cloud LLM: ' + llm_model if llm_mode == 'cloud' and llm_key else 'Local Qwen'})...")
            script_prompt = f"""Write a 35-to-45 second documentary script about '{topic}'.
Break it into 4 chronological scenes.
For each scene provide:
- "text": Punchy, gripping narration (1 to 2 sentences). Hook viewer immediately. Never say meta-text like 'Scene 1' or 'The first scene'.
- "prompt": A descriptive visual prompt for an image generator (8k, photorealistic).

Format strictly as a raw JSON array:
[
  {{"id": 1, "text": "...", "prompt": "..."}}
]"""
            if llm_mode == "cloud" and llm_key:
                raw_script = run_cloud_llm(script_prompt, llm_key, llm_url, llm_model)
            else:
                raw_script = run_local_llm(f"<|im_start|>user\n{script_prompt}<|im_end|>\n<|im_start|>assistant\n")

            scenes = extract_scenes_json(raw_script)

        if not scenes:
            raise ValueError("Failed to extract valid scenes from script output.")

        for s in scenes:
            s["text"] = sanitize_scene_text(s["text"])

        job_state["progress"] = 25
        log(f"Script ready ({len(scenes)} scenes).")

        clip_paths = []
        clip_durations = []
        voice_durations = []

        for idx, scene in enumerate(scenes):
            progress_base = 25 + int((idx / len(scenes)) * 45)
            job_state["progress"] = progress_base

            audio_path = os.path.join(OUT_DIR, f"web_voice_{idx}.wav")
            img_path = os.path.join(OUT_DIR, f"web_scene_{idx}.png")
            clip_path = os.path.join(OUT_DIR, f"web_clip_{idx}.mp4")

            log(f"[Scene {idx+1}/{len(scenes)}] Synthesizing speech (Kokoro bm_george)...")
            tts_cmd = [
                SHERPA_TTS,
                f"--kokoro-model={KOKORO_DIR}/model.onnx",
                f"--kokoro-voices={KOKORO_DIR}/voices.bin",
                f"--kokoro-tokens={KOKORO_DIR}/tokens.txt",
                f"--kokoro-data-dir={KOKORO_DIR}/espeak-ng-data",
                f"--output-filename={audio_path}",
                "--sid=9",
                scene["text"]
            ]
            subprocess.run(tts_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            v_dur = float(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path
            ]).decode().strip())
            voice_durations.append(v_dur)

            fps = 30
            frames = int((v_dur + 0.5) * fps)

            if img_mode == "cloud" and img_key:
                log(f"[Scene {idx+1}/{len(scenes)}] Generating visual via Cloud ({img_model})...")
                generate_image_cloud(scene["prompt"], img_path, aspect_ratio, img_key, img_url, img_model)
            else:
                log(f"[Scene {idx+1}/{len(scenes)}] Generating visual via Local Dream NPU/GPU...")
                generate_image_local_dream(scene["prompt"], img_path, aspect_ratio)

            log(f"[Scene {idx+1}/{len(scenes)}] Compiling motion clip ({aspect_ratio})...")
            if aspect_ratio == "9:16":
                filt = (
                    f"[0:v]crop=w='min(iw,ih*9/16)':h='min(ih,iw*16/9)',scale=2160:3840,"
                    f"zoompan=z='min(zoom+0.0012,1.20)':d={frames}:"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1080x1920:fps={fps}[v];"
                    f"[1:a]apad=pad_dur=0.5[a]"
                )
            else:
                filt = (
                    f"[0:v]scale=3840:-2,zoompan=z='min(zoom+0.0012,1.20)':d={frames}:"
                    f"x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)':s=1920x1080:fps={fps}[v];"
                    f"[1:a]apad=pad_dur=0.5[a]"
                )

            ffmpeg_clip = [
                "ffmpeg", "-y", "-loop", "1", "-i", img_path, "-i", audio_path,
                "-filter_complex", filt, "-map", "[v]", "-map", "[a]",
                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-shortest",
                clip_path
            ]
            subprocess.run(ffmpeg_clip, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

            clip_dur = float(subprocess.check_output([
                "ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", clip_path
            ]).decode().strip())
            clip_paths.append(clip_path)
            clip_durations.append(clip_dur)

        job_state["progress"] = 75
        log("Crossfading scene segments...")
        temp_stitched = os.path.join(OUT_DIR, "web_stitched.mp4")

        if len(clip_paths) == 1:
            shutil.copy(clip_paths[0], temp_stitched)
        else:
            inputs = []
            for cp in clip_paths:
                inputs.extend(["-i", cp])

            v_filters, a_filters = [], []
            v_chain, a_chain = "[0:v]", "[0:a]"
            accum_offset = clip_durations[0] - 0.5

            for i in range(1, len(clip_paths)):
                next_v, next_a = f"[{i}:v]", f"[{i}:a]"
                out_v = f"[v{i}]" if i < len(clip_paths) - 1 else "[v_out]"
                out_a = f"[a{i}]" if i < len(clip_paths) - 1 else "[a_out]"

                v_filters.append(f"{v_chain}{next_v}xfade=transition=fade:duration=0.5:offset={accum_offset:.3f}{out_v}")
                a_filters.append(f"{a_chain}{next_a}acrossfade=d=0.5:c1=tri:c2=tri{out_a}")

                v_chain, a_chain = out_v, out_a
                if i < len(clip_paths) - 1:
                    accum_offset += clip_durations[i] - 0.5

            filter_str = ";".join(v_filters + a_filters)
            concat_cmd = [
                "ffmpeg", "-y", *inputs, "-filter_complex", filter_str,
                "-map", "[v_out]", "-map", "[a_out]", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", temp_stitched
            ]
            subprocess.run(concat_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        job_state["progress"] = 88
        safe_topic = re.sub(r'[^a-zA-Z0-9_-]', '_', topic)[:25]
        ratio_tag = "Shorts" if aspect_ratio == "9:16" else "Landscape"
        final_filename = f"{safe_topic}_{ratio_tag}.mp4"
        final_output = os.path.join(EXPORT_DIR, final_filename)

        log("Generating captions & mastering audio...")
        ass_path = os.path.join(OUT_DIR, "web_captions.ass")
        build_ass_subtitles(scenes, voice_durations, ass_path, aspect_ratio)

        has_bg = os.path.exists(BG_MUSIC)
        if include_subtitles and has_bg:
            burn_cmd = [
                "ffmpeg", "-y", "-i", temp_stitched, "-stream_loop", "-1", "-i", BG_MUSIC,
                "-filter_complex", f"[0:v]subtitles='{ass_path}'[sub_v];[1:a]volume=0.08[bg];[0:a][bg]amix=inputs=2:duration=first[aout]",
                "-map", "[sub_v]", "-map", "[aout]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k", final_output
            ]
        elif include_subtitles and not has_bg:
            burn_cmd = [
                "ffmpeg", "-y", "-i", temp_stitched, "-vf", f"subtitles='{ass_path}'",
                "-c:a", "copy", "-c:v", "libx264", "-pix_fmt", "yuv420p", final_output
            ]
        elif not include_subtitles and has_bg:
            burn_cmd = [
                "ffmpeg", "-y", "-i", temp_stitched, "-stream_loop", "-1", "-i", BG_MUSIC,
                "-filter_complex", "[1:a]volume=0.08[bg];[0:a][bg]amix=inputs=2:duration=first[aout]",
                "-map", "0:v", "-map", "[aout]", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", final_output
            ]
        else:
            shutil.copy(temp_stitched, final_output)
            burn_cmd = None

        if burn_cmd:
            subprocess.run(burn_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)

        job_state["progress"] = 100
        job_state["status"] = "Completed"
        job_state["last_video"] = final_filename
        log(f"Video saved to: {final_output}")

    except Exception as e:
        log(f"ERROR: {str(e)}")
        job_state["status"] = f"Failed: {str(e)}"
    finally:
        job_state["running"] = False

# --- Web UI ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AI Studio</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>body { font-family: 'Plus Jakarta Sans', sans-serif; }</style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen pb-12">
    <header class="border-b border-slate-800 bg-slate-900/60 backdrop-blur-md sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div class="flex items-center space-x-3">
            <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-fuchsia-500 flex items-center justify-center font-bold text-white text-base shadow-lg shadow-indigo-500/20">
                AI
            </div>
            <div>
                <h1 class="text-lg font-extrabold tracking-tight">AI Studio</h1>
                <p class="text-xs text-indigo-400 font-medium">Local & Cloud Accelerated Pipeline</p>
            </div>
        </div>
        <div id="ld-badge" class="px-3 py-1.5 rounded-full text-xs font-semibold bg-slate-800 text-slate-400 border border-slate-700 flex items-center gap-2">
            <span class="w-2 h-2 rounded-full bg-slate-500"></span> Checking Local Dream...
        </div>
    </header>

    <main class="max-w-6xl mx-auto px-4 mt-8 grid grid-cols-1 lg:grid-cols-12 gap-8">
        <div class="lg:col-span-7 space-y-6">
            <div class="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 shadow-xl">
                <h2 class="text-base font-semibold text-slate-200 mb-4 flex items-center gap-2">
                    <svg class="w-5 h-5 text-indigo-400" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M12 6V4m0 2a2 2 0 100 4m0-4a2 2 0 110 4m-6 8a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4m6 6v10m6-2a2 2 0 100-4m0 4a2 2 0 110-4m0 4v2m0-6V4"></path></svg>
                    Production Setup
                </h2>

                <div class="space-y-4">
                    <!-- Ratio -->
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Aspect Ratio</label>
                        <div class="grid grid-cols-2 gap-3">
                            <label class="relative flex items-center justify-center p-3 rounded-xl border border-slate-800 cursor-pointer bg-slate-950/60 has-[:checked]:border-indigo-500 has-[:checked]:bg-indigo-500/10 transition-all">
                                <input type="radio" name="ratio" value="9:16" checked class="sr-only">
                                <div class="text-center">
                                    <span class="block font-bold text-sm">9:16 Vertical</span>
                                    <span class="block text-xs text-slate-400">Shorts / Reels (1080x1920)</span>
                                </div>
                            </label>
                            <label class="relative flex items-center justify-center p-3 rounded-xl border border-slate-800 cursor-pointer bg-slate-950/60 has-[:checked]:border-indigo-500 has-[:checked]:bg-indigo-500/10 transition-all">
                                <input type="radio" name="ratio" value="16:9" class="sr-only">
                                <div class="text-center">
                                    <span class="block font-bold text-sm">16:9 Landscape</span>
                                    <span class="block text-xs text-slate-400">YouTube Video (1920x1080)</span>
                                </div>
                            </label>
                        </div>
                    </div>

                    <!-- Topic -->
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Topic</label>
                        <input id="topic-input" type="text" placeholder="Enter topic or leave blank for AI brainstorming..." 
                            class="w-full bg-slate-950 border border-slate-800 rounded-xl px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 transition-colors">
                    </div>

                    <!-- Custom Script -->
                    <div>
                        <label class="block text-xs font-semibold text-slate-400 uppercase tracking-wider mb-2">Custom Script (Optional)</label>
                        <textarea id="script-input" rows="3" placeholder="Paste script paragraphs or JSON array. Leave empty for AI script generation..."
                            class="w-full bg-slate-950 border border-slate-800 rounded-xl p-3 text-xs font-mono focus:outline-none focus:border-indigo-500 transition-colors"></textarea>
                    </div>

                    <!-- Subtitles Toggle -->
                    <div class="flex items-center justify-between pt-1">
                        <span class="text-sm font-medium text-slate-300">Burn Synchronized Captions</span>
                        <input id="subtitles-toggle" type="checkbox" checked class="w-5 h-5 accent-indigo-500 rounded cursor-pointer">
                    </div>

                    <!-- Acceleration & Model Auto-Detection -->
                    <div class="border border-slate-800/80 rounded-xl p-4 bg-slate-950/50 space-y-4">
                        <button type="button" onclick="toggleApiSettings()" class="flex items-center justify-between w-full text-left text-xs font-bold text-indigo-400 uppercase tracking-wider">
                            <span>⚡ Acceleration & Model Selection</span>
                            <span id="api-toggle-icon">▼</span>
                        </button>

                        <div id="api-settings-body" class="space-y-4 hidden pt-2">
                            <!-- LLM Engine -->
                            <div class="space-y-2 border-b border-slate-800 pb-4">
                                <div class="flex items-center justify-between">
                                    <span class="text-xs font-bold text-slate-300">Script LLM Engine</span>
                                    <div class="flex gap-2 text-xs">
                                        <label class="cursor-pointer flex items-center gap-1">
                                            <input type="radio" name="llm_mode" value="local" checked onchange="toggleLlmFields()"> Local Qwen 3B
                                        </label>
                                        <label class="cursor-pointer flex items-center gap-1">
                                            <input type="radio" name="llm_mode" value="cloud" onchange="toggleLlmFields()"> Cloud API
                                        </label>
                                    </div>
                                </div>
                                <div id="llm-cloud-inputs" class="space-y-2 hidden pt-1">
                                    <!-- Preset Provider Selector -->
                                    <div class="flex items-center gap-2">
                                        <label class="text-[11px] text-slate-400 font-medium">Provider:</label>
                                        <select id="llm-provider" onchange="onLlmProviderChange()" class="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-indigo-300 focus:outline-none">
                                            <option value="openai">OpenAI</option>
                                            <option value="groq">Groq</option>
                                            <option value="openrouter">OpenRouter</option>
                                            <option value="deepseek">DeepSeek</option>
                                            <option value="custom">Custom / Self-Hosted</option>
                                        </select>
                                    </div>
                                    <input id="llm-key" type="password" placeholder="API Key (sk-...)" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono focus:border-indigo-500 focus:outline-none">
                                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                                        <input id="llm-url" type="text" placeholder="Base URL" value="https://api.openai.com/v1" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono focus:border-indigo-500 focus:outline-none">
                                        <div class="flex gap-1.5">
                                            <input id="llm-model" list="llm-model-list" type="text" placeholder="Model (type or select)" value="gpt-4o-mini" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono focus:border-indigo-500 focus:outline-none">
                                            <datalist id="llm-model-list"></datalist>
                                            <button type="button" id="btn-fetch-llm" onclick="fetchModels('llm')" class="px-3 bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/40 rounded-lg text-xs font-semibold whitespace-nowrap transition-all">
                                                Detect
                                            </button>
                                        </div>
                                    </div>
                                    <div id="llm-fetch-status" class="text-[11px] text-slate-500 italic"></div>
                                </div>
                            </div>

                            <!-- Image Engine -->
                            <div class="space-y-2">
                                <div class="flex items-center justify-between">
                                    <span class="text-xs font-bold text-slate-300">Image Generation Engine</span>
                                    <div class="flex gap-2 text-xs">
                                        <label class="cursor-pointer flex items-center gap-1">
                                            <input type="radio" name="img_mode" value="local" checked onchange="toggleImgFields()"> Local Dream
                                        </label>
                                        <label class="cursor-pointer flex items-center gap-1">
                                            <input type="radio" name="img_mode" value="cloud" onchange="toggleImgFields()"> Cloud API
                                        </label>
                                    </div>
                                </div>
                                <div id="img-cloud-inputs" class="space-y-2 hidden pt-1">
                                    <!-- Preset Provider Selector -->
                                    <div class="flex items-center gap-2">
                                        <label class="text-[11px] text-slate-400 font-medium">Provider:</label>
                                        <select id="img-provider" onchange="onImgProviderChange()" class="bg-slate-900 border border-slate-800 rounded-lg px-2.5 py-1 text-xs text-indigo-300 focus:outline-none">
                                            <option value="openai">OpenAI (DALL-E)</option>
                                            <option value="together">Together AI (Flux / SDXL)</option>
                                            <option value="custom">Custom / Self-Hosted</option>
                                        </select>
                                    </div>
                                    <input id="img-key" type="password" placeholder="API Key (sk-...)" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono focus:border-indigo-500 focus:outline-none">
                                    <div class="grid grid-cols-1 md:grid-cols-2 gap-2">
                                        <input id="img-url" type="text" placeholder="Base URL" value="https://api.openai.com/v1" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono focus:border-indigo-500 focus:outline-none">
                                        <div class="flex gap-1.5">
                                            <input id="img-model" list="img-model-list" type="text" placeholder="Model (type or select)" value="dall-e-3" class="w-full bg-slate-900 border border-slate-800 rounded-lg px-3 py-2 text-xs font-mono focus:border-indigo-500 focus:outline-none">
                                            <datalist id="img-model-list"></datalist>
                                            <button type="button" id="btn-fetch-img" onclick="fetchModels('img')" class="px-3 bg-indigo-600/30 hover:bg-indigo-600/50 text-indigo-300 border border-indigo-500/40 rounded-lg text-xs font-semibold whitespace-nowrap transition-all">
                                                Detect
                                            </button>
                                        </div>
                                    </div>
                                    <div id="img-fetch-status" class="text-[11px] text-slate-500 italic"></div>
                                </div>
                            </div>
                        </div>
                    </div>

                    <button id="btn-generate" onclick="triggerPipeline()" 
                        class="w-full mt-2 bg-gradient-to-r from-indigo-500 via-purple-600 to-pink-600 hover:from-indigo-400 hover:to-pink-500 text-white font-bold py-3.5 px-6 rounded-xl shadow-lg shadow-indigo-500/20 active:scale-[0.98] transition-all">
                        Render Video
                    </button>
                </div>
            </div>

            <!-- Terminal -->
            <div class="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 shadow-xl">
                <div class="flex items-center justify-between mb-3">
                    <h2 class="text-sm font-semibold text-slate-300 flex items-center gap-2">
                        <span id="log-indicator" class="w-2 h-2 rounded-full bg-slate-600"></span> Live Pipeline Logs
                    </h2>
                    <span id="progress-percent" class="text-xs font-mono font-bold text-indigo-400">0%</span>
                </div>
                <div class="w-full bg-slate-950 rounded-full h-2 mb-3 overflow-hidden border border-slate-800">
                    <div id="progress-bar" class="bg-gradient-to-r from-indigo-500 via-purple-500 to-pink-500 h-2 transition-all duration-300" style="width: 0%"></div>
                </div>
                <div id="console-logs" class="bg-slate-950 border border-slate-800/60 rounded-xl p-4 h-48 overflow-y-auto text-xs font-mono text-slate-300 space-y-1">
                    <div class="text-slate-500 italic">Waiting for pipeline trigger...</div>
                </div>
            </div>
        </div>

        <!-- Preview & Gallery -->
        <div class="lg:col-span-5 space-y-6">
            <div class="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 shadow-xl">
                <h2 class="text-base font-semibold text-slate-200 mb-4">Latest Render</h2>
                <div id="player-container" class="bg-slate-950 border border-slate-800 rounded-xl overflow-hidden flex items-center justify-center min-h-[360px]">
                    <div id="player-placeholder" class="text-center p-6 text-slate-500 text-sm">No video rendered yet this session.</div>
                    <video id="video-player" controls class="w-full h-auto hidden max-h-[500px]"></video>
                </div>
                <div id="video-actions" class="mt-4 hidden">
                    <a id="btn-download" href="#" download class="block text-center w-full bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-bold py-2.5 rounded-lg transition-colors">
                        Download Video
                    </a>
                </div>
            </div>

            <div class="bg-slate-900 border border-slate-800/80 rounded-2xl p-6 shadow-xl">
                <h2 class="text-base font-semibold text-slate-200 mb-3">Gallery (/sdcard/pipeline)</h2>
                <div id="gallery-list" class="space-y-2 max-h-56 overflow-y-auto pr-1 text-xs">
                    <div class="text-slate-500 italic">Scanning directory...</div>
                </div>
            </div>
        </div>
    </main>

    <script>
        let isPolling = false;

        const LLM_PRESETS = {
            openai: { url: "https://api.openai.com/v1", defaultModel: "gpt-4o-mini" },
            groq: { url: "https://api.groq.com/openai/v1", defaultModel: "llama-3.3-70b-versatile" },
            openrouter: { url: "https://openrouter.ai/api/v1", defaultModel: "deepseek/deepseek-chat" },
            deepseek: { url: "https://api.deepseek.com", defaultModel: "deepseek-chat" },
            custom: { url: "", defaultModel: "" }
        };

        const IMG_PRESETS = {
            openai: { url: "https://api.openai.com/v1", defaultModel: "dall-e-3" },
            together: { url: "https://api.together.xyz/v1", defaultModel: "black-forest-labs/FLUX.1-schnell" },
            custom: { url: "", defaultModel: "" }
        };

        function onLlmProviderChange() {
            const val = document.getElementById('llm-provider').value;
            const preset = LLM_PRESETS[val];
            if (preset && val !== 'custom') {
                document.getElementById('llm-url').value = preset.url;
                document.getElementById('llm-model').value = preset.defaultModel;
            }
        }

        function onImgProviderChange() {
            const val = document.getElementById('img-provider').value;
            const preset = IMG_PRESETS[val];
            if (preset && val !== 'custom') {
                document.getElementById('img-url').value = preset.url;
                document.getElementById('img-model').value = preset.defaultModel;
            }
        }

        async function fetchModels(type) {
            const isLlm = (type === 'llm');
            const key = document.getElementById(isLlm ? 'llm-key' : 'img-key').value.trim();
            const url = document.getElementById(isLlm ? 'llm-url' : 'img-url').value.trim();
            const statusEl = document.getElementById(isLlm ? 'llm-fetch-status' : 'img-fetch-status');
            const btn = document.getElementById(isLlm ? 'btn-fetch-llm' : 'btn-fetch-img');
            const datalist = document.getElementById(isLlm ? 'llm-model-list' : 'img-model-list');

            if (!key) {
                statusEl.innerText = "Please enter an API key first.";
                statusEl.className = "text-[11px] text-rose-400";
                return;
            }

            statusEl.innerText = "Querying available models...";
            statusEl.className = "text-[11px] text-indigo-400";
            btn.disabled = true;

            try {
                const res = await fetch('/api/fetch_models', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ base_url: url, api_key: key })
                });
                const data = await res.json();

                if (data.success && data.models.length > 0) {
                    datalist.innerHTML = data.models.map(m => `<option value="${m}"></option>`).join('');
                    statusEl.innerText = `Detected ${data.models.length} models. Click or type in the model box.`;
                    statusEl.className = "text-[11px] text-emerald-400";
                    // If current input is empty, select the first detected model
                    const modelInput = document.getElementById(isLlm ? 'llm-model' : 'img-model');
                    if (!modelInput.value && data.models.length > 0) {
                        modelInput.value = data.models[0];
                    }
                } else {
                    statusEl.innerText = `Could not detect models: ${data.error || 'Empty list returned'}`;
                    statusEl.className = "text-[11px] text-amber-400";
                }
            } catch (err) {
                statusEl.innerText = `Network error: ${err.message}`;
                statusEl.className = "text-[11px] text-rose-400";
            } finally {
                btn.disabled = false;
            }
        }

        function toggleApiSettings() {
            const body = document.getElementById('api-settings-body');
            const icon = document.getElementById('api-toggle-icon');
            if (body.classList.contains('hidden')) {
                body.classList.remove('hidden');
                icon.innerText = '▲';
            } else {
                body.classList.add('hidden');
                icon.innerText = '▼';
            }
        }

        function toggleLlmFields() {
            const mode = document.querySelector('input[name="llm_mode"]:checked').value;
            const container = document.getElementById('llm-cloud-inputs');
            if (mode === 'cloud') container.classList.remove('hidden');
            else container.classList.add('hidden');
        }

        function toggleImgFields() {
            const mode = document.querySelector('input[name="img_mode"]:checked').value;
            const container = document.getElementById('img-cloud-inputs');
            if (mode === 'cloud') container.classList.remove('hidden');
            else container.classList.add('hidden');
        }

        function saveSettings() {
            const settings = {
                llm_mode: document.querySelector('input[name="llm_mode"]:checked').value,
                llm_provider: document.getElementById('llm-provider').value,
                llm_key: document.getElementById('llm-key').value,
                llm_url: document.getElementById('llm-url').value,
                llm_model: document.getElementById('llm-model').value,
                img_mode: document.querySelector('input[name="img_mode"]:checked').value,
                img_provider: document.getElementById('img-provider').value,
                img_key: document.getElementById('img-key').value,
                img_url: document.getElementById('img-url').value,
                img_model: document.getElementById('img-model').value
            };
            localStorage.setItem('aistudio_keys', JSON.stringify(settings));
        }

        function loadSettings() {
            const saved = localStorage.getItem('aistudio_keys');
            if (!saved) return;
            try {
                const s = JSON.parse(saved);
                if (s.llm_mode) document.querySelector(`input[name="llm_mode"][value="${s.llm_mode}"]`).checked = true;
                if (s.llm_provider) document.getElementById('llm-provider').value = s.llm_provider;
                if (s.llm_key) document.getElementById('llm-key').value = s.llm_key;
                if (s.llm_url) document.getElementById('llm-url').value = s.llm_url;
                if (s.llm_model) document.getElementById('llm-model').value = s.llm_model;

                if (s.img_mode) document.querySelector(`input[name="img_mode"][value="${s.img_mode}"]`).checked = true;
                if (s.img_provider) document.getElementById('img-provider').value = s.img_provider;
                if (s.img_key) document.getElementById('img-key').value = s.img_key;
                if (s.img_url) document.getElementById('img-url').value = s.img_url;
                if (s.img_model) document.getElementById('img-model').value = s.img_model;

                toggleLlmFields();
                toggleImgFields();
            } catch (e) {}
        }

        async function checkLocalDream() {
            try {
                const res = await fetch('/api/ld_status');
                const data = await res.json();
                const badge = document.getElementById('ld-badge');
                if (data.active) {
                    badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span> Local Dream: ${data.model} (${data.size}px)`;
                    badge.className = "px-3 py-1.5 rounded-full text-xs font-semibold bg-emerald-950/60 text-emerald-400 border border-emerald-800/60 flex items-center gap-2";
                } else {
                    badge.innerHTML = `<span class="w-2 h-2 rounded-full bg-rose-500"></span> Local Dream: Offline`;
                    badge.className = "px-3 py-1.5 rounded-full text-xs font-semibold bg-rose-950/60 text-rose-400 border border-rose-800/60 flex items-center gap-2";
                }
            } catch (e) {}
        }

        async function triggerPipeline() {
            saveSettings();
            const ratio = document.querySelector('input[name="ratio"]:checked').value;
            const topic = document.getElementById('topic-input').value;
            const customScript = document.getElementById('script-input').value;
            const subtitles = document.getElementById('subtitles-toggle').checked;

            const btn = document.getElementById('btn-generate');
            btn.disabled = true;
            btn.classList.add('opacity-50', 'cursor-not-allowed');

            const payload = {
                aspect_ratio: ratio,
                topic: topic,
                custom_script: customScript,
                subtitles: subtitles,
                llm_mode: document.querySelector('input[name="llm_mode"]:checked').value,
                llm_key: document.getElementById('llm-key').value,
                llm_url: document.getElementById('llm-url').value,
                llm_model: document.getElementById('llm-model').value,
                img_mode: document.querySelector('input[name="img_mode"]:checked').value,
                img_key: document.getElementById('img-key').value,
                img_url: document.getElementById('img-url').value,
                img_model: document.getElementById('img-model').value
            };

            await fetch('/api/generate', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload)
            });

            startPolling();
        }

        function startPolling() {
            if (isPolling) return;
            isPolling = true;
            const interval = setInterval(async () => {
                const res = await fetch('/api/status');
                const data = await res.json();

                document.getElementById('progress-bar').style.width = data.progress + '%';
                document.getElementById('progress-percent').innerText = data.progress + '%';

                const logBox = document.getElementById('console-logs');
                if (data.logs.length > 0) {
                    logBox.innerHTML = data.logs.map(l => `<div class="${l.includes('ERROR') ? 'text-rose-400' : ''}">${l}</div>`).join('');
                    logBox.scrollTop = logBox.scrollHeight;
                }

                const indicator = document.getElementById('log-indicator');
                if (data.running) {
                    indicator.className = "w-2 h-2 rounded-full bg-indigo-400 animate-ping";
                } else {
                    indicator.className = "w-2 h-2 rounded-full bg-slate-600";
                    clearInterval(interval);
                    isPolling = false;
                    const btn = document.getElementById('btn-generate');
                    btn.disabled = false;
                    btn.classList.remove('opacity-50', 'cursor-not-allowed');

                    if (data.last_video) {
                        showVideo(data.last_video);
                        loadGallery();
                    }
                }
            }, 1000);
        }

        function showVideo(filename) {
            const player = document.getElementById('video-player');
            const placeholder = document.getElementById('player-placeholder');
            const actions = document.getElementById('video-actions');
            const dl = document.getElementById('btn-download');

            player.src = `/videos/${filename}`;
            player.classList.remove('hidden');
            placeholder.classList.add('hidden');
            actions.classList.remove('hidden');
            dl.href = `/videos/${filename}`;
            player.load();
        }

        async function loadGallery() {
            const res = await fetch('/api/videos');
            const files = await res.json();
            const container = document.getElementById('gallery-list');
            if (files.length === 0) {
                container.innerHTML = '<div class="text-slate-500 italic">No videos in pipeline folder yet.</div>';
                return;
            }
            container.innerHTML = files.map(f => `
                <div class="flex items-center justify-between p-2.5 rounded-lg bg-slate-950/80 border border-slate-800 hover:border-slate-700 transition-colors">
                    <span class="truncate font-medium text-slate-300 max-w-[240px]">${f}</span>
                    <button onclick="showVideo('${f}')" class="text-indigo-400 hover:text-indigo-300 font-bold ml-2">Preview</button>
                </div>
            `).join('');
        }

        loadSettings();
        checkLocalDream();
        loadGallery();
        setInterval(checkLocalDream, 8000);
    </script>
</body>
</html>
"""

# --- Endpoints ---
@app.route("/")
def index():
    return render_template_string(HTML_TEMPLATE)

@app.route("/api/ld_status")
def ld_status():
    try:
        r = requests.get(LD_STATUS_URL, timeout=2).json()
        return jsonify({
            "active": r.get("state") == "running",
            "model": r.get("serving_model_id", "unknown"),
            "size": r.get("width", 512)
        })
    except Exception:
        return jsonify({"active": False})

@app.route("/api/fetch_models", methods=["POST"])
def fetch_models_endpoint():
    """Queries any standard OpenAI-compatible /models endpoint and returns model IDs."""
    data = request.get_json() or {}
    base_url = data.get("base_url", "https://api.openai.com/v1").rstrip("/")
    api_key = data.get("api_key", "").strip()

    if not api_key:
        return jsonify({"success": False, "error": "API Key is required"}), 400

    endpoint = f"{base_url}/models" if base_url.endswith("/v1") else f"{base_url}/v1/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    try:
        resp = requests.get(endpoint, headers=headers, timeout=10)
        # Fallback if provider expects base without /v1
        if resp.status_code == 404 and not base_url.endswith("/v1"):
            endpoint = f"{base_url}/models"
            resp = requests.get(endpoint, headers=headers, timeout=10)

        resp.raise_for_status()
        res_json = resp.json()

        models = []
        if "data" in res_json and isinstance(res_json["data"], list):
            for item in res_json["data"]:
                if isinstance(item, dict) and "id" in item:
                    models.append(item["id"])
        elif isinstance(res_json, list):
            for item in res_json:
                if isinstance(item, dict) and "id" in item:
                    models.append(item["id"])
                elif isinstance(item, str):
                    models.append(item)

        models = sorted(list(set(models)))
        return jsonify({"success": True, "models": models})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})

@app.route("/api/generate", methods=["POST"])
def generate():
    if job_state["running"]:
        return jsonify({"error": "Pipeline is already running"}), 400
    params = request.get_json() or {}
    t = threading.Thread(target=execute_pipeline, args=(params,))
    t.daemon = True
    t.start()
    return jsonify({"status": "started"})

@app.route("/api/status")
def status():
    return jsonify(job_state)

@app.route("/api/videos")
def list_videos():
    if not os.path.exists(EXPORT_DIR):
        return jsonify([])
    mp4s = [f for f in os.listdir(EXPORT_DIR) if f.endswith(".mp4")]
    mp4s.sort(key=lambda x: os.path.getmtime(os.path.join(EXPORT_DIR, x)), reverse=True)
    return jsonify(mp4s)

@app.route("/videos/<filename>")
def serve_video(filename):
    return send_from_directory(EXPORT_DIR, filename)

if __name__ == "__main__":
    print("\n[✔] Starting AI Studio Server with Model Auto-Detection...")
    print("    Access at: http://localhost:5000\n")
    app.run(host="0.0.0.0", port=5000, debug=False)
