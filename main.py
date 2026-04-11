import cv2
import time
import pyttsx3
import threading
import queue
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image

# ==========================================
# 🔑 CONFIGURATION
# ==========================================
CAMERA_INDEX = 1
GEMINI_API_KEY = "AIzaSyAveR8EsstrcpJPJCtXDpgIZSU-ncdu6IQ"

PRINT_COOLDOWN = 1.5
VOICE_COOLDOWN = 2.5

FOCAL_LENGTH = 800  # camera focal length (pixels)

KNOWN_WIDTHS = {
    "person": 0.5,
    "chair": 0.45,
    "bottle": 0.07,
    "laptop": 0.35,
    "car": 1.8,
    "bus": 2.5,
    "backpack": 0.3,     
    "umbrella": 0.9,
    "handbag": 0.35,
    "tie": 0.08,
    "cell phone": 0.075
}

TARGET_OBJECTS = list(KNOWN_WIDTHS.keys())

# ==========================================
# ⚙️ INITIALIZATION
# ==========================================
print("🚀 Initializing S.E.E Project...")

speech_queue = queue.Queue()
last_spoken = {}
last_printed = {}

# ==========================================
# 🔊 VOICE SYSTEM (STABLE)
# ==========================================
def voice_worker():
    while True:
        text = speech_queue.get()
        if text is None:
            break
        try:
            engine = pyttsx3.init()
            engine.setProperty('rate', 170)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
        except Exception as e:
            print(f"🔇 Voice error: {e}")
        speech_queue.task_done()

voice_thread = threading.Thread(target=voice_worker)
voice_thread.start()

def speak_text(text, key=None):
    now = time.time()
    if key:
        if now - last_spoken.get(key, 0) < VOICE_COOLDOWN:
            return
        last_spoken[key] = now
    speech_queue.put(text)

# ==========================================
# 📐 DISTANCE ESTIMATION
# ==========================================
def estimate_distance(real_width, pixel_width):
    if pixel_width <= 0:
        return None
    return (real_width * FOCAL_LENGTH) / pixel_width

# ==========================================
# 🤖 GEMINI
# ==========================================
genai.configure(api_key=GEMINI_API_KEY)
vision_model = genai.GenerativeModel('gemini-2.0-flash')

def describe_scene_with_ai(frame):
    speak_text("Analyzing scene")
    try:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        prompt = (
            "You are a guide for a blind person. "
            "Describe this scene in one short sentence. "
            "Mention the most important obstacle."
        )
        response = vision_model.generate_content([prompt, img])
        if response.text:
            print(f"🤖 Gemini: {response.text}")
            speak_text(response.text)
    except:
        speak_text("AI error occurred")

# ==========================================
# 🎥 YOLO + CAMERA
# ==========================================
model = YOLO("yolov8n.pt")

cap = cv2.VideoCapture(CAMERA_INDEX)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

if not cap.isOpened():
    print("❌ Camera not found")
    exit()

print("✅ System Ready!")

# ==========================================
# 🔁 MAIN LOOP
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        threading.Thread(
            target=describe_scene_with_ai,
            args=(frame.copy(),),
            daemon=True
        ).start()

    results = model(frame, stream=True, verbose=False)

    for r in results:
        for box in r.boxes:
            conf = float(box.conf[0])
            if conf < 0.5:
                continue

            cls = int(box.cls[0])
            name = model.names[cls]

            if name not in TARGET_OBJECTS:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            pixel_width = x2 - x1

            distance = estimate_distance(
                KNOWN_WIDTHS[name],
                pixel_width
            )

            if distance:
                msg = f"{name} at {distance:.1f} meters"
                speak_text(msg, key=name)

                cv2.putText(
                    frame,
                    f"{name} {distance:.1f}m",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 0),
                    2
                )

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow("S.E.E Prototype", frame)

# ==========================================
# 🛑 CLEAN EXIT
# ==========================================
print("🛑 Terminating S.E.E...")
speak_text("S E E terminated")
speech_queue.join()
speech_queue.put(None)
voice_thread.join()

cap.release()
cv2.destroyAllWindows()
