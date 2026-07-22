import json
import os
from base64 import b64decode
from binascii import Error as Base64DecodeError
from io import BytesIO
from pathlib import Path

from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from google import genai
from google.genai import types
from PIL import Image, UnidentifiedImageError


BASE_DIR = Path(__file__).resolve().parent
PUBLIC_DIR = BASE_DIR.parent / "public"


load_dotenv(BASE_DIR / ".env")

app = Flask(__name__, static_folder=None)
CORS(app, resources={r"/run-ocr": {"origins": os.getenv("CORS_ORIGINS", "*")}})


def get_genai_client():
    if not os.getenv("GEMINI_API_KEY"):
        return None, "GEMINI_API_KEY environment variable is not set on the server."

    try:
        return genai.Client(), None
    except Exception as exc:
        app.logger.exception("Failed to initialize Gemini client")
        return None, f"Error initializing Gemini client: {exc.__class__.__name__}"


def run_gemini_ocr(base64_image_data: str, mime_type: str) -> dict:
    try:
        image_bytes = b64decode(base64_image_data, validate=True)
        image = Image.open(BytesIO(image_bytes))
        image.verify()
        image = Image.open(BytesIO(image_bytes))

        ocr_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "extractedText": types.Schema(
                    type=types.Type.STRING,
                    description=(
                        "The complete, accurately extracted text from the image, "
                        "preserving line breaks. Use \\n for line breaks."
                    ),
                )
            },
            required=["extractedText"],
        )

        config = types.GenerateContentConfig(
            system_instruction=(
                "You are an AI-powered text recognition engine specialized in "
                "high-accuracy OCR, particularly for handwritten text. Extract all "
                "text from the image."
            ),
            response_mime_type="application/json",
            response_schema=ocr_schema,
        )

        client, client_err = get_genai_client()
        if client_err:
            return {"error": client_err}

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                (
                    "Extract all text from the image, including line breaks, and "
                    "return the result as a single JSON object structured exactly "
                    "as requested. Focus on accurate text extraction regardless of "
                    "the text's style (handwritten or printed)."
                ),
                image,
            ],
            config=config,
        )

        text = (response.text or "").strip()
        if text.startswith("```json"):
            text = text[7:-3].strip()
        elif text.startswith("```"):
            text = text[3:-3].strip()

        parsed_model_output = json.loads(text)
        extracted_text = parsed_model_output.get(
            "extractedText", "No text could be extracted."
        )

        return {"extractedText": extracted_text}

    except (Base64DecodeError, UnidentifiedImageError):
        return {"error": "Invalid image data."}
    except genai.errors.APIError as exc:
        app.logger.exception("Gemini API error")
        return {
            "error": (
                f"API Error: {exc.__class__.__name__}. "
                "Check your API key or usage limits."
            )
        }
    except (json.JSONDecodeError, KeyError, TypeError) as exc:
        app.logger.exception("Gemini returned an invalid OCR payload")
        return {"error": f"Invalid OCR response: {exc.__class__.__name__}"}


@app.get("/")
def index():
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.post("/run-ocr")
def run_ocr_endpoint():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    data = request.get_json(silent=True) or {}
    base64_data_with_prefix = data.get("imageBase64")
    mime_type = data.get("fileMimeType")

    if not base64_data_with_prefix or not mime_type:
        return jsonify({"error": "Missing image data or MIME type in request."}), 400

    _, _, base64_data = base64_data_with_prefix.partition(",")
    result = run_gemini_ocr(base64_data or base64_data_with_prefix, mime_type)

    if "error" in result:
        status_code = 400 if result["error"] == "Invalid image data." else 500
        return jsonify(result), status_code

    return jsonify(result)


def main():
    app.run(debug=True)


if __name__ == "__main__":
    main()
