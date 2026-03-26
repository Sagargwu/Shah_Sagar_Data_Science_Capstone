from pathlib import Path
import uuid

from flask import Flask, jsonify, render_template, request, send_from_directory, url_for
from werkzeug.utils import secure_filename

from lane_service import process_video

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"

ALLOWED_EXTENSIONS = {"mp4", "mov", "avi", "mkv"}
MAX_FILE_SIZE_MB = 300

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_FILE_SIZE_MB * 1024 * 1024


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    if "video" not in request.files:
        return jsonify({"ok": False, "message": "No video file was uploaded."}), 400

    file = request.files["video"]

    if file.filename == "":
        return jsonify({"ok": False, "message": "Please choose a video file first."}), 400

    if not allowed_file(file.filename):
        return jsonify({
            "ok": False,
            "message": "Unsupported file type. Please upload MP4, MOV, AVI, or MKV."
        }), 400

    original_name = secure_filename(file.filename)
    file_id = uuid.uuid4().hex[:12]
    upload_name = f"{file_id}_{original_name}"
    output_name = f"lane_detected_{file_id}.mp4"

    upload_path = UPLOAD_DIR / upload_name
    output_path = OUTPUT_DIR / output_name

    try:
        file.save(upload_path)
        stats = process_video(str(upload_path), str(output_path))

        return jsonify({
            "ok": True,
            "message": "Lane detection completed successfully.",
            "download_url": url_for("download_file", filename=output_name),
            "preview_url": url_for("output_file", filename=output_name),
            "output_filename": output_name,
            "stats": stats
        })
    except Exception as exc:
        return jsonify({
            "ok": False,
            "message": f"Processing failed: {str(exc)}"
        }), 500


@app.route("/outputs/<path:filename>")
def output_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=False)


@app.route("/download/<path:filename>")
def download_file(filename: str):
    return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)


if __name__ == "__main__":
    app.run(debug=True)