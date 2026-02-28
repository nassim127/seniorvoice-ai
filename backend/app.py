from __future__ import annotations

import os
import tempfile

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from intent_model import parse_command
from stt_model import transcribe_audio

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})




@app.route("/", methods=["GET"])
def home():
    frontend_dir = os.path.join(os.path.dirname(__file__), "..", "frontend")
    return send_from_directory(frontend_dir, "index.html")

@app.route("/health", methods=["GET"])
def health() -> dict:
    return {"status": "ok"}


@app.route("/process", methods=["POST"])
def process():
    if "audio" not in request.files:
        return jsonify({"error": "audio file is required"}), 400

    audio = request.files["audio"]
    suffix = os.path.splitext(audio.filename or "input.wav")[1] or ".wav"

    fd, temp_path = tempfile.mkstemp(prefix="seniorvoice_", suffix=suffix)
    os.close(fd)
    try:
        audio.save(temp_path)
        stt = transcribe_audio(temp_path)
        parsed = parse_command(stt["text"])
        parsed["confidence"] = stt["confidence"]
        parsed["raw_text"] = stt["text"]
        parsed["language"] = stt["language"]
        return jsonify(parsed)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


if __name__ == "__main__":
    app.run()
