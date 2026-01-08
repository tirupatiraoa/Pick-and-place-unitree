import os
import sys
import time
import traceback

import cv2
import numpy as np
from google import genai
from google.genai import types
from google.genai.types import HttpOptions

# ---------------- ZED SDK ----------------
try:
    import pyzed.sl as sl
except ImportError:
    print("[FATAL] pyzed (ZED SDK) not installed")
    sys.exit(1)

# ---------------- GEMINI CONFIG ----------------
HTTP_OPTS = HttpOptions(api_version="v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Use a stable vision model for live demos
MODEL_ID = "gemini-2.5-flash"

# ---------------- PROMPT ----------------
PROMPT = """You are a robotic shelf inspection system.
The input is an image of a supermarket rack.

Each shelf in the image contains 
 "Chips", "Drink Tin Can", "Chocolate", "Fruit Juice").

1. Detect and identify image 
2. Based on the image identification category which it self it is a product_name
3. Identify all visible products on the rack.

Return ONLY valid JSON in the following format:
{"product_name": ""}
No explanation and no extra content.
"""

# ---------------- GEMINI CLIENT (GLOBAL) ----------------
_gemini_client = None


def get_gemini_client():
    global _gemini_client
    if _gemini_client is None:
        _gemini_client = genai.Client(
            api_key=GEMINI_API_KEY,
            http_options=HTTP_OPTS,
        )
    return _gemini_client


# ---------------- ZED CAMERA ----------------
def open_zed():
    init = sl.InitParameters()
    init.camera_resolution = sl.RESOLUTION.HD720
    init.camera_fps = 30
    init.depth_mode = sl.DEPTH_MODE.NONE

    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {status}")
    return cam


def grab_frame(cam):
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    if cam.grab(runtime) == sl.ERROR_CODE.SUCCESS:
        cam.retrieve_image(mat, sl.VIEW.LEFT)
        img = mat.get_data()

        # RGBA / RGB → BGR
        if img.shape[2] == 4:
            return img[:, :, :3][:, :, ::-1].copy()
        return img[:, :, ::-1].copy()

    raise RuntimeError("Failed to grab frame from ZED")


# ---------------- GEMINI VISION ----------------
def run_gemini_live(frame: np.ndarray, retries: int = 3) -> str | None:
    if not GEMINI_API_KEY:
        print("[WARN] GEMINI_API_KEY not set")
        return None

    client = get_gemini_client()

    ok, jpg = cv2.imencode(".jpg", frame)
    if not ok:
        print("[WARN] JPEG encoding failed")
        return None

    contents = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=PROMPT),
                types.Part.from_bytes(
                    mime_type="image/jpeg",
                    data=jpg.tobytes(),
                ),
            ],
        )
    ]

    for attempt in range(retries):
        try:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=contents,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=2048,
                ),
            )
            return resp.text or None

        except Exception as e:
            msg = str(e)

            # 503 – model overloaded
            if "503" in msg or "UNAVAILABLE" in msg:
                wait = 2**attempt
                print(f"[WARN] Gemini overloaded (503). Retrying in {wait}s...")
                time.sleep(wait)
                continue

            # 429 – quota exhausted
            if "RESOURCE_EXHAUSTED" in msg or "429" in msg:
                print("[WARN] Gemini quota exhausted. Skipping vision call.")
                return None

            # Any other error → real failure
            raise

    print("[ERROR] Gemini failed after retries.")
    return None


# ---------------- MAIN LOOP ----------------
def main():
    cam = None

    try:
        print("[INFO] Opening ZED 2i...")
        cam = open_zed()
        print("[INFO] ZED camera opened.")

        print("[INFO] Press 'g' to run Gemini shelf inspection")
        print("[INFO] Press 'q' to quit")

        last_call_time = 0
        MIN_INTERVAL = 1.5  # seconds (important to avoid overload)

        while True:
            frame = grab_frame(cam)
            cv2.imshow("ZED Live Shelf View", frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("g"):
                now = time.time()
                if now - last_call_time < MIN_INTERVAL:
                    print("[INFO] Gemini throttled (too frequent).")
                else:
                    last_call_time = now
                    print("[INFO] Running Gemini Vision...")
                    result = run_gemini_live(frame)
                    if result:
                        print("\n--- Gemini Shelf Inspection JSON ---")
                        print(result)
                        print("-----------------------------------\n")
                    else:
                        print("[INFO] Gemini response unavailable.")

            if key == ord("q"):
                break

            time.sleep(0.03)

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")

    except Exception as e:
        print("[ERROR]", e)
        traceback.print_exc()

    finally:
        if cam:
            cam.close()
            print("[INFO] ZED camera closed.")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
