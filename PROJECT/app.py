import os
import json
from flask import Flask, request, jsonify, send_from_directory
# 1. Import CORS
from flask_cors import CORS 
from google import genai
from google.genai import types
from PIL import Image
from io import BytesIO
from base64 import b64decode

# ----------------- CONFIGURATION -----------------
# Check for API Key
if not os.getenv("GEMINI_API_KEY"):
    print("FATAL ERROR: GEMINI_API_KEY environment variable is not set.")
    print("Please set it before running. Example: export GEMINI_API_KEY=\"YOUR_API_KEY\"")
    exit(1)
    
app = Flask(__name__)
# 2. Initialize CORS
# This allows requests from any origin (*), which is safe for local development.
CORS(app) 

# Initialize the Gemini client
try:
    client = genai.Client()
except Exception as e:
    print(f"Error initializing Gemini client: {e}")
    exit(1)


# ----------------- CORE OCR LOGIC -----------------

def run_gemini_ocr(base64_image_data: str, mime_type: str) -> dict:
    """
    Performs OCR using the Gemini-2.5-Flash model on a base64-encoded image.
    The response is formatted as a JSON object containing the extracted text.
    """
    try:
        # Convert Base64 data to a PIL Image object for the SDK
        image_bytes = b64decode(base64_image_data)
        # Use Image.open with BytesIO to read image from memory
        image = Image.open(BytesIO(image_bytes))

        system_instruction = (
            "You are an AI-powered text recognition engine specialized in high-accuracy OCR, "
            "particularly for handwritten text. Extract all text from the image."
        )
        user_query = (
            "Extract all text from the image, including line breaks, and return the result as a "
            "single JSON object structured exactly as requested. Focus on accurate text extraction "
            "regardless of the text's style (handwritten or printed). The extracted text must use "
            "\\n for line breaks for proper JSON escaping."
        )

        # Define the response schema to force JSON output and structure
        ocr_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "extractedText": types.Schema(
                    type=types.Type.STRING,
                    description="The complete, accurately extracted text from the image, preserving line breaks. Use \\n for line breaks."
                )
            }
        )

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=ocr_schema,
        )

        # Call the Gemini API
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=[user_query, image],
            config=config,
        )
        
        # Parse the JSON response text provided by the model
        parsed_model_output = json.loads(response.text)
        extracted_text = parsed_model_output.get("extractedText", "No text could be extracted.")
        
        return {"extractedText": extracted_text}

    except genai.errors.APIError as e:
        print(f"Gemini API Error: {e}")
        return {"error": f"API Error: {e.__class__.__name__}. Check your API key or usage limits."}
    except Exception as e:
        print(f"General Error during OCR: {e}")
        return {"error": f"Internal Server Error: {e.__class__.__name__}. Details: {str(e)}"}

# ----------------- FLASK ROUTES -----------------

@app.route('/')
def index():
    """Route to serve the main HTML file."""
    # The file must be named 'index.html' and reside in the same directory
    return send_from_directory('.', 'index.html')

@app.route('/run-ocr', methods=['POST'])
def run_ocr_endpoint():
    """Route to handle the front-end's image submission and run OCR."""
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
        
    data = request.json
    base64_data_with_prefix = data.get('imageBase64')
    mime_type = data.get('fileMimeType')

    if not base64_data_with_prefix or not mime_type:
        return jsonify({"error": "Missing image data or MIME type in request."}), 400

    # Strip the data URI prefix (e.g., 'data:image/png;base64,')
    try:
        # The split[1] gets the actual base64 data after the comma
        base64_data = base64_data_with_prefix.split(',', 1)[1]
    except IndexError:
        return jsonify({"error": "Invalid base64 format received from client."}), 400

    # Call the core OCR function
    result = run_gemini_ocr(base64_data, mime_type)
    
    # Check if the core function returned an error
    if "error" in result:
        # Return a 500 status code for server-side errors
        return jsonify(result), 500
        
    # Success: Return the extracted text JSON object
    return jsonify(result)

if __name__ == '__main__':
    print("Flask server starting...")
    print("Make sure you have set the GEMINI_API_KEY environment variable.")
    print("\nOpen this link in your browser: http://127.0.0.1:5000/")
    # Run the server on http://127.0.0.1:5000/
    app.run(debug=True)