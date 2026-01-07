
# demo4.py (google.genai) - FIXED: keyword-only Part.from_text

from pathlib import Path
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

HTTP_OPTS = HttpOptions(api_version="v1")

def run_gemini_shelf_inspection(image_path: str, user_instruction: str, model_id: str = "gemini-2.5-flash") -> str:
    """
    Sends a text+image prompt to Gemini and returns the model's textual analysis.
    """
    img_file = Path(image_path)
    if not img_file.exists():
        raise FileNotFoundError(f"Image not found: {img_file}")

    client = genai.Client(http_options=HTTP_OPTS)
    try:
        contents = [
            types.Content(
                role="user",
                parts=[
                    # IMPORTANT: keyword-only
                    types.Part.from_text(text=user_instruction),
                    types.Part.from_bytes(
                        mime_type="image/jpeg" if img_file.suffix.lower() in [".jpg", ".jpeg"] else "image/png",
                        data=img_file.read_bytes(),
                    ),
                ],
            )
        ]

        resp = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=2048,
            ),
        )

        return resp.text or "No text content returned."
    finally:
        client.close()


if __name__ == "__main__":
    image = r"C:\Users\talthi\OneDrive - Capgemini\Documents\Comcast\Tiru\shelf2.png"
    prompt = """You are a robotic shelf inspection system.
   The input is an image of a supermarket rack.
    Each shelf in the image contains a visible text label written on or just below the shelf
    (e.g., "Chips", "Drink Tin Can", "Milk Bottle", "Breads and Biscuits").
    This text label explicitly defines the category of products that are allowed on that shelf.

    Shelf segmentation and product validation rules:
    1. Detect and read the text label associated with each shelf.
    2. The detected text label becomes the expected product category for that shelf.
    3. Identify all visible products on the rack.
    4. For each product, compare its category with the expected category defined by that shelf’s text label.
    5. If a product’s category does not match the shelf’s label, mark it as misplaced.

    Confidence handling:
    - If the shelf label is partially visible, blurry, or ambiguous, mark confidence as "low".
    - If the product identity is unclear, mark confidence as "low".
    - Otherwise, use "high".

    Return ONLY valid JSON in the following format:

    {
      "misplaced_items": [
        {
          "product_name": "",
          "current_shelf_label": "",
          "expected_category": "",
          "confidence": "",
          "reason": ""
        }
      ]
    }"""

    try:
        output = run_gemini_shelf_inspection(image, prompt)
        print("\n--- Gemini Analysis ---\n")
        print(output)
    except Exception as e:
        print(f"ERROR: {e}")
