import cv2
import math
import pyttsx3
import threading
import google.generativeai as genai
from ultralytics import YOLO
from PIL import Image 

# ==========================================
# 🔑 CONFIGURATION
# ==========================================
# 1. IP Webcam URL (Keep /video at the end)
URL = "http://10.229.92.160:8080/video" # <--- CHECK THIS IP AGAIN

# 2. Gemini API Key
GEMINI_API_KEY = "AIzaSyAveR8EsstrcpJPJCtXDpgIZSU-ncdu6IQ"

# 3. Target Objects
TARGET_OBJECTS = ['person', 'chair', 'bottle', 'laptop', 'stairs', 'car', 'bus']

# ==========================================
# ⚙️ INITIALIZATION
# ==========================================
print("🚀 Initializing S.E.E Project...")

# 1. Setup Voice
engine = pyttsx3.init()
engine.setProperty('rate', 150)
is_speaking = False

# 2. Setup YOLO
print("Loading YOLOv8...")
model = YOLO('yolov8n.pt')

# 3. Setup Gemini (Using the model from your list)
genai.configure(api_key=GEMINI_API_KEY)
print("Connecting to Gemini 2.0...")

# We use the model name exactly as it appeared in your list
try:
    vision_model = genai.GenerativeModel('gemini-2.0-flash')
    print("✅ Success! Connected to Gemini 2.0 Flash")
except:
    # Fallback just in case
    vision_model = genai.GenerativeModel('gemini-flash-latest')
    print("⚠️ Switched to 'gemini-flash-latest'")

# ==========================================
# 🛠️ FUNCTIONS
# ==========================================
def speak_text(text):
    """Speaks without freezing video"""
    global is_speaking
    def run():
        global is_speaking
        is_speaking = True
        try:
            print(f"🗣️ Speaking: {text}")
            engine.say(text)
            engine.runAndWait()
        except: pass
        is_speaking = False
    
    if not is_speaking:
        threading.Thread(target=run).start()

def describe_scene_with_ai(frame):
    """Sends image to Gemini 2.0"""
    speak_text("Analyzing...")
    try:
        # Convert Frame
        color_converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(color_converted)
        
        # Ask Gemini
        prompt = "You are a guide for a blind person. Describe this scene in one very short sentence. Mention the most important obstacle."
        
        response = vision_model.generate_content([prompt, pil_image])
        
        if response.text:
            speak_text(response.text)
    except Exception as e:
        print(f"❌ AI Error: {e}")
        speak_text("Connection error.")

# ==========================================
# 🎥 MAIN LOOP
# ==========================================
cap = cv2.VideoCapture(URL)

if not cap.isOpened():
    print("❌ ERROR: Check IP Webcam URL.")
    exit()

print("✅ System Ready! Press 's' to Describe Scene.")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Resize
    frame = cv2.resize(frame, (640, 480))
    frame_width = frame.shape[1]
    
    # Controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        threading.Thread(target=describe_scene_with_ai, args=(frame,)).start()

    # YOLO Detection (Safety Mode)
    if not is_speaking:
        results = model(frame, stream=True, verbose=False)
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                name = model.names[int(box.cls[0])]
                
                if name in TARGET_OBJECTS and box.conf[0] > 0.5:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Distance Alert
                    width = x2 - x1
                    if width > (frame_width * 0.45):
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 4)
                        speak_text(f"Stop. {name} very close.")

    cv2.imshow('S.E.E Prototype', frame)

cap.release()
cv2.destroyAllWindows()