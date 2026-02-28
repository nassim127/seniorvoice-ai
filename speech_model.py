from __future__ import annotations

import os
import re
import wave
from pathlib import Path

import numpy as np
import whisper

_MODEL = None
TARGET_SR = 16000
_ALLOWED_CHAR = re.compile(r"[A-Za-z?-?؀-ۿ0-9\s'?.,!?;:()\-]")


def _get_cache_path(model_name: str) -> Path:
    return Path.home() / ".cache" / "whisper" / f"{model_name}.pt"


def _load_model() -> whisper.Whisper:
    global _MODEL
    if _MODEL is not None:
        return _MODEL

    model_name = os.getenv("SENIORVOICE_WHISPER_MODEL", "base")
    try:
        _MODEL = whisper.load_model(model_name)
        return _MODEL
    except RuntimeError as err:
        if "checksum" not in str(err).lower():
            raise
        cache_file = _get_cache_path(model_name)
        if cache_file.exists():
            cache_file.unlink()
        _MODEL = whisper.load_model(model_name)
        return _MODEL


def _read_wav_as_float32(path: str) -> tuple[np.ndarray, int]:
    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sample_width = wf.getsampwidth()
        sample_rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if sample_width != 2:
        raise ValueError("Only PCM16 WAV is supported.")

    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)

    return audio, sample_rate


def _resample_linear(audio: np.ndarray, src_sr: int, dst_sr: int = TARGET_SR) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio

    duration = audio.shape[0] / float(src_sr)
    dst_len = int(round(duration * dst_sr))
    if dst_len <= 1:
        return audio

    src_x = np.linspace(0.0, duration, num=audio.shape[0], endpoint=False)
    dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
    return np.interp(dst_x, src_x, audio).astype(np.float32)


def _normalize_peak(audio: np.ndarray, target_peak: float = 0.85) -> np.ndarray:
    if audio.size == 0:
        return audio

    peak = float(np.max(np.abs(audio)))
    if peak < 1e-6:
        return audio

    gain = min(target_peak / peak, 6.0)
    return np.clip(audio * gain, -1.0, 1.0).astype(np.float32)


def _is_repetitive(text: str) -> bool:
    words = [w for w in re.split(r"\s+", text.lower().strip()) if w]
    if len(words) < 3:
        return False

    if len(set(words)) <= 2 and len(words) >= 4:
        return True

    pair = " ".join(words[:2])
    if pair and text.lower().count(pair) >= 3:
        return True

    return False


def _text_quality_score(text: str, result: dict) -> float:
    if not text or not text.strip():
        return -999.0

    length = max(len(text), 1)
    allowed = sum(1 for ch in text if _ALLOWED_CHAR.fullmatch(ch))
    allowed_ratio = allowed / length

    cyrillic = sum(1 for ch in text if "Ѐ" <= ch <= "ӿ")
    cyr_ratio = cyrillic / length

    segments = result.get("segments", [])
    if segments:
        avg_logprob = sum(seg.get("avg_logprob", -1.5) for seg in segments) / len(segments)
        avg_nospeech = sum(seg.get("no_speech_prob", 0.5) for seg in segments) / len(segments)
        avg_compression = sum(seg.get("compression_ratio", 2.0) for seg in segments) / len(segments)
    else:
        avg_logprob = -1.5
        avg_nospeech = 0.5
        avg_compression = 2.0

    repetition_penalty = 1.8 if _is_repetitive(text) else 0.0
    compression_penalty = max(0.0, avg_compression - 2.4)

    return (
        (2.5 * allowed_ratio)
        + avg_logprob
        - (1.3 * avg_nospeech)
        - (0.8 * compression_penalty)
        - (3.2 * cyr_ratio)
        - repetition_penalty
    )


def _transcribe_attempt(model: whisper.Whisper, audio: np.ndarray, language: str | None) -> dict:
    kwargs = {
        "audio": audio,
        "fp16": False,
        "task": "transcribe",
        "temperature": 0.0,
        "condition_on_previous_text": False,
        "beam_size": 5,
        "best_of": 3,
        "no_speech_threshold": 0.6,
        "logprob_threshold": -1.2,
        "compression_ratio_threshold": 2.4,
        "initial_prompt": "Senior tunisien, arabe dialectal tunisien et francais.",
    }
    if language is not None:
        kwargs["language"] = language
    return model.transcribe(**kwargs)


def transcribe_audio(audio_path: str) -> dict:
    if not audio_path.lower().endswith(".wav"):
        raise ValueError("Audio format not supported without ffmpeg. Please send WAV audio.")

    audio, sr = _read_wav_as_float32(audio_path)
    audio = _resample_linear(audio, sr, TARGET_SR)
    audio = _normalize_peak(audio)

    # Too short or too quiet audio often creates hallucinated phrases.
    if audio.size < int(0.7 * TARGET_SR):
        return {"text": "", "confidence": 0.0, "language": "unknown"}

    rms = float(np.sqrt(np.mean(audio ** 2))) if audio.size else 0.0
    if rms < 0.008:
        return {"text": "", "confidence": 0.0, "language": "unknown"}

    model = _load_model()
    attempts = [None, "fr", "ar"]

    best_result = None
    best_score = -10_000.0
    for lang in attempts:
        result = _transcribe_attempt(model, audio, lang)
        text = result.get("text", "").strip()
        score = _text_quality_score(text, result)
        if score > best_score:
            best_score = score
            best_result = result

    result = best_result or {"text": "", "language": "unknown", "segments": []}
    text = result.get("text", "").strip()

    segments = result.get("segments", [])
    if segments:
        avg_no_speech = sum(seg.get("no_speech_prob", 0.5) for seg in segments) / len(segments)
        confidence = max(0.0, min(1.0, 1.0 - avg_no_speech))
    else:
        confidence = 0.4

    if _is_repetitive(text) and confidence < 0.75:
        text = ""
        confidence = min(confidence, 0.35)

    return {
        "text": text,
        "confidence": round(confidence, 3),
        "language": result.get("language", "unknown"),
    }
