
# Livedemo1_zed.py
# ----------------
# Works with Stereolabs ZED 2i (not Intel RealSense).
# - Uses ZED SDK (pyzed) to open camera and grab color frames (left eye).
# - Migrates from google.generativeai -> google.genai.
# - Graceful checks and cleanup.

import os
import sys
import time
import traceback

import numpy as np

# Optional: preview window or JPEG encoding (comment out if not used)
# import cv2

# --- ZED SDK (pyzed) ---
try:
    import pyzed.sl as sl
except ImportError:
    print("[FATAL] pyzed (ZED SDK Python API) is not available in this interpreter.")
    print("Install the Stereolabs ZED SDK and matching Python API (wheel) for your Python version.")
    sys.exit(1)

# --- Google Gemini (new SDK) ---
try:
    from google import genai
except ImportError:
    print("[FATAL] google-genai is not installed. Install with:")
    print("  python -m pip install google-genai")
    sys.exit(1)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def open_zed_camera(serial_number: int | None = None, resolution=sl.RESOLUTION.HD720, fps: int = 30):
    """
    Open the ZED camera with optional serial filtering.
    Returns the initialized sl.Camera() instance.
    """
    init_params = sl.InitParameters()
    init_params.camera_resolution = resolution
    init_params.camera_fps = fps
    init_params.depth_mode = sl.DEPTH_MODE.NONE  # set to ULTRA/QUALITY/PERFORMANCE if you need depth
    init_params.coordinate_units = sl.UNIT.MILLIMETER  # for depth if enabled
    if serial_number is not None:
        init_params.input.set_from_serial_number(serial_number)

    cam = sl.Camera()
    status = cam.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(f"ZED open failed: {repr(status)}. "
                           "Check USB 3.x port, cable, and that ZED Explorer can see the camera.")
    return cam


def grab_left_image(cam: sl.Camera, timeout_s: int = 5) -> np.ndarray:
    """
    Grabs a single left image from the ZED camera as a numpy array (H,W,3) in BGR.
    """
    runtime_params = sl.RuntimeParameters()
    end_time = time.time() + timeout_s

    mat = sl.Mat()
    while time.time() < end_time:
        if cam.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            cam.retrieve_image(mat, sl.VIEW.LEFT)  # LEFT is in RGB by default
            img = mat.get_data()  # returns numpy array in RGBA or RGB depending on SDK version
            # Convert to BGR if you plan to use with OpenCV
            if img.shape[2] == 4:
                # RGBA -> BGR
                bgr = img[:, :, :3][:, :, ::-1].copy()
            else:
                # RGB -> BGR
                bgr = img[:, :, ::-1].copy()
            return bgr
        time.sleep(0.005)

    raise RuntimeError("Timed out waiting for a frame from ZED camera.")


def main():
    cam = None
    try:
        print("[INFO] Opening ZED 2i...")
        cam = open_zed_camera(resolution=sl.RESOLUTION.HD720, fps=30)
        print("[INFO] ZED camera opened.")

        bgr = grab_left_image(cam, timeout_s=5)
        print(f"[INFO] Captured frame from ZED. Shape: {bgr.shape}")

        # # Optional: show the image using OpenCV
        # cv2.imshow("ZED Left", bgr)
        # cv2.waitKey(1)

        # Example Gemini call to confirm google.genai works
        if client:
            resp = client.models.generate_content(
                model="gemini-1.5-flash",
                contents="Hello from ZED 2i pipeline using google.genai!"
            )
            print("[GEMINI]", resp.text.strip())
        else:
            print("[INFO] Skipping Gemini call: GEMINI_API_KEY not set.")

        # # Optional: send the frame to Gemini Vision (uncomment if needed)
        # ok, jpg = cv2.imencode(".jpg", bgr)
        # if not ok:
        #     raise RuntimeError("Failed to encode JPEG.")
        # from google.genai.types import Part
        # img_part = Part.from_bytes(data=jpg.tobytes(), mime_type="image/jpeg")
        # vis = client.models.generate_content(
        #     model="gemini-1.5-flash",
        #     contents=[img_part, "Describe this scene in one sentence."]
        # )
        # print("[GEMINI-VISION]", vis.text.strip())

    except Exception as e:
        print("\n[ERROR] An exception occurred:\n", e)
        traceback.print_exc()
    finally:
        try:
            if cam is not None:
                cam.close()
        except Exception:
            pass
        # try:
        #     cv2.destroyAllWindows()
        # except Exception:
        #     pass


if __name__ == "__main__":
    main()
``
