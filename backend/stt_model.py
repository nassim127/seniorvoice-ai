from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT_STT_PATH = Path(__file__).resolve().parent.parent / "speech_model.py"
SPEC = importlib.util.spec_from_file_location("seniorvoice_root_speech_model", ROOT_STT_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load speech model from {ROOT_STT_PATH}")

MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

transcribe_audio = MODULE.transcribe_audio

