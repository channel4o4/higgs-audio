#!/usr/bin/env python3
"""
Higgs Audio dialogue TTS helper

Reads a transcript with [SPEAKER*] tags, generates speech for each
utterance with a per-speaker reference voice using Higgs Audio V2,
and stitches it into a single WAV normalized to −14 LUFS for YouTube.

Supports reference conditioning in three ways (merged in this order):
1) --ref-text SPEAKERX="text here"
2) --ref-file SPEAKERX=path/to/textfile.txt   # file contains raw text (no SPEAKERX prefix)
3) --ref-file path/to/mapping.txt             # file with one or more lines of either:
       SPEAKERX=raw reference text
       SPEAKERX=path/to/textfile.txt

Also includes:
- Progress output for each line processed.
- Fixed seed passed to engine & generation for consistent results.
"""
from __future__ import annotations

import argparse
import base64
import os
import re
import sys
import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
import soundfile as sf
import pyloudnorm as pyln

try:
    from higgs_audio.serve.serve_engine import HiggsAudioServeEngine
    from higgs_audio.data_types import ChatMLSample, AudioContent, Message
except Exception:
    from boson_multimodal.serve.serve_engine import HiggsAudioServeEngine  # type: ignore
    from boson_multimodal.data_types import ChatMLSample, AudioContent, Message  # type: ignore

SAMPLE_RATE = 24000
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
            raise argparse.ArgumentTypeError(f"--ref-text '{p}' must look like SPEAKERX=\"text\"")
        k, v = p.split("=", 1)
        out[k.strip().upper()] = v.strip()
    return out


def _read_text_file(path: str) -> str:
    with open(path, 'r', encoding='utf-8') as f:
        return f.read().strip()


def load_ref_texts_from_files(entries: List[str]) -> Dict[str, str]:
    """Accepts a list of --ref-file entries that can be:
    - SPEAKERX=/path/to/textfile.txt  (file contains raw text, no prefix)
    - /path/to/mapping.txt            (file with lines SPEAKERX=text or SPEAKERX=/path/to/textfile.txt)
    Returns a dict {SPEAKERX: text}.
    """
    result: Dict[str, str] = {}
    for e in entries:
        if '=' in e and os.path.exists(e.split('=', 1)[1]):
            # Direct mapping: SPEAKERX=path/to/textfile.txt
            spk, path = e.split('=', 1)
            spk = spk.strip().upper()
            txt = _read_text_file(path.strip())
            result[spk] = txt
            continue
        # Otherwise treat as a mapping file path
        path = e
        if not os.path.exists(path):
            raise FileNotFoundError(f"Reference text file not found: {path}")
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if '=' not in line:
                    raise ValueError(f"Invalid line in ref mapping file {path}: {line}")
                k, v = line.split('=', 1)
                spk = k.strip().upper()
                rhs = v.strip()
                if os.path.exists(rhs):
                    # rhs points to a file; read its content
                    result[spk] = _read_text_file(rhs)
                else:
                    # rhs is inline text
                    result[spk] = rhs
    return result


def read_transcript(path: str) -> List[Utterance]:
    lines: List[Utterance] = []
    with open(path, 'r', encoding='utf-8') as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            m = LINE_RE.match(raw)
            if not m:
                raise ValueError(f"Invalid line format: {raw}")
            lines.append(Utterance(m.group(1), m.group(2)))
    return lines


def maybe_chunk(text: str, max_chars: int) -> List[str]:
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]
    parts: List[str] = []
    for sent in SENT_SPLIT_RE.split(text):
        if not parts:
            parts.append(sent)
        elif len(parts[-1]) + 1 + len(sent) <= max_chars:
            parts[-1] += ' ' + sent
        else:
            parts.append(sent)
    return [s.strip() for s in parts if s.strip()]


def b64_of_file(path: str) -> str:
    with open(path, 'rb') as f:
        import base64 as _b64
        return _b64.b64encode(f.read()).decode('utf-8')


def build_chatml_sample(
    system_prompt: str,
    ref_audio_b64: Optional[str],
    ref_text: str,
    text: str,
    history: Sequence[Tuple[str, str]],
) -> ChatMLSample:
    messages = []
    if system_prompt:
        messages.append(Message(role='system', content=system_prompt))
    if ref_audio_b64:
        messages.append(Message(role='user', content=ref_text or 'Reference audio for target voice.'))
        messages.append(Message(role='assistant', content=[AudioContent(raw_audio=ref_audio_b64, audio_url='')]))
    for hist_text, hist_audio_b64 in history:
        messages.append(Message(role='user', content=hist_text))
        messages.append(Message(role='assistant', content=[AudioContent(raw_audio=hist_audio_b64, audio_url='')]))
    messages.append(Message(role='user', content=text))
    return ChatMLSample(messages=messages)


def audio_array_to_b64(audio: np.ndarray, sr: int) -> str:
    import io

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format='WAV', subtype='FLOAT')

    return base64.b64encode(buf.getvalue()).decode('utf-8')


def normalize_to_lufs(audio: np.ndarray, sr: int, target_lufs: float = -14.0) -> np.ndarray:
    meter = pyln.Meter(sr)
    loudness = meter.integrated_loudness(audio if audio.ndim == 1 else audio.mean(axis=1))
    gain = 10 ** ((target_lufs - loudness) / 20)
    audio = np.clip(audio * gain, -1.0, 1.0)
    return audio.astype(np.float32)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description='Higgs Audio dialogue TTS')
    p.add_argument('--input', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--ref', action='append', default=[], help='SPEAKERX=path.wav (repeat)')
    p.add_argument('--ref-text', action='append', default=[], help='SPEAKERX="text" (repeat)')
    p.add_argument('--ref-file', action='append', default=[], help='Either SPEAKERX=path/to/textfile.txt (repeat) or a mapping file path (repeat)')
    p.add_argument('--model', default='bosonai/higgs-audio-v2-generation-3B-base')
    p.add_argument('--tokenizer', default='bosonai/higgs-audio-v2-tokenizer')
    p.add_argument('--system-prompt', default='Generate audio following instruction.')
    p.add_argument('--scene-file', help='Path to a text file for scene description; wrapped in <|scene_desc_start|> ... <|scene_desc_end|>')
    p.add_argument('--max-new-tokens', type=int, default=1024)
    p.add_argument('--temperature', type=float, default=1.0)
    p.add_argument('--top-p', type=float, default=0.95)
    p.add_argument('--top-k', type=int, default=50)
    p.add_argument('--max-chars', type=int, default=0)
    p.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    p.add_argument(
        '--speaker-buffer-size',
        type=int,
        default=2,
        help='Number of previous utterances per speaker to include as audio/text context (0 to disable).',
    )

    args = p.parse_args(argv)

    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except ImportError:
        pass

    refs = parse_refs(args.ref)
    ref_texts = parse_ref_texts(args.ref_text)

    if args.ref_file:
        file_texts = load_ref_texts_from_files(args.ref_file)
        ref_texts.update(file_texts)

    ref_audio_b64 = {spk: b64_of_file(path) for spk, path in refs.items()}
    lines = read_transcript(args.input)

    # Build final system prompt (optionally augmented with scene file)
    system_prompt = args.system_prompt
    if args.scene_file:
        if not os.path.exists(args.scene_file):
            raise FileNotFoundError(f"Scene file not found: {args.scene_file}")
        with open(args.scene_file, 'r', encoding='utf-8') as sfp:
            scene_text = sfp.read().strip()
        if '<|scene_desc_start|>' not in system_prompt:
            system_prompt = system_prompt.rstrip() + "<|scene_desc_start|>" + scene_text + "<|scene_desc_end|>"
        else:
            system_prompt = system_prompt.rstrip() + "" + scene_text

    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    engine = HiggsAudioServeEngine(
        model_name_or_path=args.model,
        audio_tokenizer_name_or_path=args.tokenizer,
        device=device,
    )

    chunks: List[np.ndarray] = []
    speaker_histories: Dict[str, Deque[Tuple[str, str]]] = {}
    total_lines = len(lines)
    for idx, u in enumerate(lines, start=1):
        print(f"[{idx}/{total_lines}] Processing {u.speaker}: {u.text[:80]}...", flush=True)
        speaker_ref_audio = ref_audio_b64.get(u.speaker)
        speaker_ref_text = ref_texts.get(u.speaker, '')
        history_deque: Optional[Deque[Tuple[str, str]]] = None
        if args.speaker_buffer_size > 0:
            history_deque = speaker_histories.setdefault(
                u.speaker, deque(maxlen=args.speaker_buffer_size)
            )
        for part in maybe_chunk(u.text, args.max_chars):
            history_for_prompt: Sequence[Tuple[str, str]] = ()
            if history_deque is not None:
                history_for_prompt = tuple(history_deque)
            chatml = build_chatml_sample(
                system_prompt,
                speaker_ref_audio,
                speaker_ref_text,
                part,
                history_for_prompt,
            )
            resp = engine.generate(
                chatml,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                top_p=args.top_p,
                seed=args.seed,
            )
            # Resample if model returns a different rate
            sr = getattr(resp, 'sampling_rate', SAMPLE_RATE)
            audio = resp.audio if resp.audio is not None else None
            if audio is not None and sr != SAMPLE_RATE:
                import numpy as _np, math as _math
                ratio = SAMPLE_RATE / float(sr)
                new_len = int(_math.ceil(len(audio) * ratio))
                x = _np.linspace(0, 1, len(audio), endpoint=False)
                x_new = _np.linspace(0, 1, new_len, endpoint=False)
                audio = _np.interp(x_new, x, audio).astype(np.float32)
            if audio is not None:
                audio = audio.astype(np.float32)
                chunks.append(audio)
                if history_deque is not None:
                    audio_b64 = audio_array_to_b64(audio, SAMPLE_RATE)
                    history_deque.append((part, audio_b64))

    if not chunks:
        print("[WARN] No audio generated.")
        return 1

    full = np.concatenate(chunks)
    full = normalize_to_lufs(full, SAMPLE_RATE, -14.0)
    sf.write(args.out, full, SAMPLE_RATE, subtype='FLOAT')
    print(f"Wrote {args.out} with {len(full)/SAMPLE_RATE:.2f}s of audio")
    return 0

if __name__ == '__main__':
    raise SystemExit(main())