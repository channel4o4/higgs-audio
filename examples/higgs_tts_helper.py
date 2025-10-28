#!/usr/bin/env python3
"""
Higgs Audio dialogue TTS helper

Reads a transcript with [SPEAKER*] tags, generates speech for each
utterance with a per‑speaker reference voice using Higgs Audio V2,
and stitches it into a single WAV normalized to −14 LUFS for YouTube.

Usage:
  python higgs_audio_dialogue_tts.py \
    --input transcript.txt \
    --out out.wav \
    --ref SPEAKER0=voices/alex.wav --ref SPEAKER1=voices/jordan.wav \
    --model bosonai/higgs-audio-v2-generation-3B-base \
    --tokenizer bosonai/higgs-audio-v2-tokenizer

Optional:
  --ref-text SPEAKER0="short transcript of alex.wav" \
  --max-new-tokens 2048 --temperature 0.9 --top-p 0.95 --top-k 50 \
  --ras-win-len 7 --ras-win-max-repeat 2 \
  --max-chars 0     # if >0, splits long utterances on sentence boundaries

Notes:
- Expects a CUDA GPU if available; will run on CPU if not.
- Sample rate is fixed to 24 kHz (Higgs default).
- Reference audio is embedded once per speaker and reused for all of that speaker’s lines.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Audio IO + loudness
try:
    import soundfile as sf
except Exception as e:  # pragma: no cover
    raise SystemExit("Please `pip install soundfile`.")

try:
    import pyloudnorm as pyln
except Exception:
    raise SystemExit("Please `pip install pyloudnorm` for LUFS normalization.")

# Higgs Audio imports
try:
    from higgs_audio.serve.serve_engine import HiggsAudioServeEngine
    from higgs_audio.data_types import ChatMLSample, AudioContent, Message
except Exception as e:  # pragma: no cover
    raise SystemExit(
        "Could not import Higgs Audio. Did you `pip install -r requirements.txt` and `pip install -e .` in the repo?"
    )

SAMPLE_RATE = 24000  # Higgs V2 default

LINE_RE = re.compile(r"^\s*\[(SPEAKER\d+)\]\s*(.+?)\s*$")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

@dataclass
class Utterance:
    speaker: str
    text: str


def parse_refs(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise argparse.ArgumentTypeError(f"--ref '{p}' must look like SPEAKERX=path.wav")
        k, v = p.split("=", 1)
        out[k.strip().upper()] = v.strip()
    return out


def parse_ref_texts(pairs: List[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in pairs:
        if "=" not in p:
            raise argparse.ArgumentTypeError(f"--ref-text '{p}' must look like SPEAKERX=\"transcript...\"")
        k, v = p.split("=", 1)
        out[k.strip().upper()] = v.strip()
    return out


def read_transcript(path: str) -> List[Utterance]:
    lines = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            m = LINE_RE.match(raw)
            if not m:
                raise ValueError(
                    f"Line doesn't match [SPEAKER*] format: {raw}\n"
                    "Example line: [SPEAKER0] Hello there!"
                )
            spk, text = m.group(1), m.group(2)
            lines.append(Utterance(spk, text))
    return lines


def maybe_chunk(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    for sent in SENT_SPLIT_RE.split(text):
        if not parts:
            parts.append(sent)
        elif len(parts[-1]) + 1 + len(sent) <= max_chars:
            parts[-1] += " " + sent
        else:
            parts.append(sent)
    return [s.strip() for s in parts if s.strip()]


def b64_of_file(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_chatml_sample(
    system_prompt: str,
    ref_audio_b64: Optional[str],
    ref_text: str,
    text: str,
) -> ChatMLSample:
    messages: List[Message] = []
    if system_prompt:
        messages.append(Message(role="system", content=system_prompt))
    if ref_audio_b64:
        # Reference pair: user transcript then assistant audio content
        messages.append(Message(role="user", content=ref_text))
        messages.append(Message(role="assistant", content=[AudioContent(raw_audio=ref_audio_b64, audio_url="")]))
    # Main user text
    messages.append(Message(role="user", content=text))
    return ChatMLSample(messages=messages)


def normalize_to_lufs(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    """Loudness-normalize mono/stereo float32 audio to target LUFS.
    Returns float32 clipped to [-1, 1].
    """
    if audio.ndim == 1:
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        loudness_diff = target_lufs - loudness
        factor = 10 ** (loudness_diff / 20)
        y = audio * factor
    else:
        # stereo: average channel loudness, apply same gain to both
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio.mean(axis=1))
        loudness_diff = target_lufs - loudness
        factor = 10 ** (loudness_diff / 20)
        y = audio * factor
    # Simple hard clip to avoid overflow
    y = np.clip(y, -1.0, 1.0)
    return y.astype(np.float32)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Higgs Audio dialogue TTS")
    p.add_argument("--input", required=True, help="Transcript file with [SPEAKER*] tags")
    p.add_argument("--out", required=True, help="Output WAV path")
    p.add_argument("--ref", action="append", default=[], help="SPEAKERX=path.wav (repeatable)")
    p.add_argument("--ref-text", action="append", default=[], help="SPEAKERX=transcript (repeatable)")
    p.add_argument("--model", default="bosonai/higgs-audio-v2-generation-3B-base")
    p.add_argument("--tokenizer", default="bosonai/higgs-audio-v2-tokenizer")
    p.add_argument("--system-prompt", default="Generate audio following instruction.\n\n<|scene_desc_start|>\nAudio is recorded from a quiet room.\n<|scene_desc_end|>")
    p.add_argument("--max-new-tokens", type=int, default=1024)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--ras-win-len", type=int, default=7)
    p.add_argument("--ras-win-max-repeat", type=int, default=2)
    p.add_argument("--max-chars", type=int, default=0, help="If >0, split long utterances by sentences to this many chars")

    args = p.parse_args(argv)

    refs = parse_refs(args.ref)
    ref_texts = parse_ref_texts(args.ref_text)

    # Pre‑encode ref audio once per speaker
    ref_audio_b64: Dict[str, str] = {}
    for spk, wav_path in refs.items():
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"Reference audio not found for {spk}: {wav_path}")
        ref_audio_b64[spk] = b64_of_file(wav_path)

    lines = read_transcript(args.input)

    # Init engine
    engine = HiggsAudioServeEngine(
        model_name_or_path=args.model,
        audio_tokenizer_name_or_path=args.tokenizer,
        device=("cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") or os.environ.get("FORCE_CUDA") or os.environ.get("NVIDIA_VISIBLE_DEVICES") else "cpu"),
    )

    chunks: List[np.ndarray] = []
    for u in lines:
        speaker_b64 = ref_audio_b64.get(u.speaker)
        speaker_ref_text = ref_texts.get(u.speaker, "")
        parts = maybe_chunk(u.text, args.max_chars)
        for part in parts:
            chatml = build_chatml_sample(
                system_prompt=args.system_prompt,
                ref_audio_b64=speaker_b64,
                ref_text=speaker_ref_text,
                text=part,
            )
            resp = engine.generate(
                chat_ml_sample=chatml,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=(args.top_k if args.top_k > 0 else None),
                top_p=args.top_p,
                stop_strings=["<|end_of_text|>", "<|eot_id|>"],
                ras_win_len=(args.ras_win_len if args.ras_win_len > 0 else None),
                ras_win_max_num_repeat=max(args.ras_win_len, args.ras_win_max_repeat),
            )
            if resp.audio is None:
                print(f"[WARN] No audio returned for {u.speaker} -> '{part[:60]}…'", file=sys.stderr)
                continue
            # resp.audio is float -1..1 at resp.sampling_rate
            if resp.sampling_rate != SAMPLE_RATE:
                # resample on the fly if needed (simple linear)
                import numpy as _np
                import math as _math
                ratio = SAMPLE_RATE / float(resp.sampling_rate)
                new_len = int(_math.ceil(len(resp.audio) * ratio))
                x = _np.linspace(0, 1, len(resp.audio), endpoint=False)
                x_new = _np.linspace(0, 1, new_len, endpoint=False)
                part_audio = _np.interp(x_new, x, resp.audio).astype(np.float32)
            else:
                part_audio = resp.audio.astype(np.float32)
            chunks.append(part_audio)

    if not chunks:
        raise SystemExit("No audio was generated.")

    full = np.concatenate(chunks)

    # Loudness normalize to −14 LUFS for YouTube
    full = normalize_to_lufs(full.astype(np.float32), SAMPLE_RATE, target_lufs=-14.0)

    # Write WAV (float32)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    sf.write(args.out, full, SAMPLE_RATE, subtype="FLOAT")
    print(f"Wrote {args.out} (sr={SAMPLE_RATE}, seconds={len(full)/SAMPLE_RATE:.2f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())