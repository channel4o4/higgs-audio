#!/usr/bin/env python3
import subprocess
import sys
import time
import os
import argparse
from pathlib import Path
import datetime

CONTENT_DIR = Path("404NewsContent")
CONTENT_DIR.mkdir(exist_ok=True)

def find_today_transcript() -> Path | None:
    """Return today's transcript file if it exists."""
    today_prefix = datetime.datetime.now().strftime("Channel404News_%Y-%m-%d")
    for p in CONTENT_DIR.glob(f"{today_prefix}_*.txt"):
        return p.resolve()
    return None

def run_generator(generator_script: str) -> Path:
    """Run transcript generator and return path to new transcript."""
    start_time = time.time()
    print("Running transcript generator...")

    try:
        result = subprocess.run(
            [sys.executable, generator_script],
            capture_output=True,
            text=True,
            check=True,
        )
        print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Generator script failed: {e.stderr}")

    # Find most recent transcript file
    candidates = []
    for p in CONTENT_DIR.glob("Channel404News_*.txt"):
        try:
            if p.stat().st_mtime >= start_time - 1.0:
                candidates.append((p.stat().st_mtime, p))
        except FileNotFoundError:
            continue

    if not candidates:
        raise RuntimeError("No transcript generated. Aborting before TTS.")

    candidates.sort(key=lambda x: x[0], reverse=True)
    transcript_path = candidates[0][1].resolve()
    print(f"Generated transcript: {transcript_path}")
    return transcript_path

def run_tts(tts_script: str, transcript_path: Path) -> Path:
    """Run TTS helper to generate .wav from transcript."""
    if not transcript_path.exists():
        raise RuntimeError(f"Transcript not found: {transcript_path}")

    wav_path = transcript_path.with_suffix(".wav")

    cmd = [
        sys.executable, tts_script,
        "--input", str(transcript_path),
        "--out", str(wav_path),
        "--ref", "SPEAKER0=examples/voice_prompts/bbc_male_reporter.wav",
        "--ref-file", "SPEAKER0=examples/voice_prompts/bbc_male_reporter.txt",
        "--seed", "42",
    ]

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"

    print("Running TTS synthesis...")
    try:
        result = subprocess.run(
            cmd,
            env=env,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            print(result.stdout.strip())
        if result.stderr:
            print(result.stderr.strip(), file=sys.stderr)
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"TTS failed: {e.stderr}")

    if not wav_path.exists():
        raise RuntimeError(f"TTS reported success but no file found: {wav_path}")

    print(f"TTS complete: {wav_path}")
    return wav_path

def main():
    parser = argparse.ArgumentParser(description="Generate and voice Channel 404 AI news.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force regeneration of today's transcript even if one exists.",
    )
    args = parser.parse_args()

    generator_script = "gen_404news_transcript.py"
    tts_script = "higgs_tts_helper.py"

    today_file = find_today_transcript()

    if today_file and not args.force:
        print(f"Today's transcript already exists: {today_file}")
        transcript_path = today_file
    else:
        if today_file and args.force:
            print("--force used, removing existing transcript.")
            today_file.unlink(missing_ok=True)
        transcript_path = run_generator(generator_script)

    wav_path = run_tts(tts_script, transcript_path)

    print("\nAll done.")
    print(f"Transcript: {transcript_path}")
    print(f"Audio:      {wav_path}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)