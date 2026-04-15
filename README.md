main.py - A blind-assistance vision system that:
- Uses YOLOv8 to detect objects (person, chair, bottle, laptop, car, etc.) via webcam
- Estimates distance to objects using focal length math
- Announces detected objects via text-to-speech (e.g., "person at 2.3 meters")
- Press 's' to get an AI scene description via Gemini
- Press 'q' to quit
  
see.py - A simpler version of the same blind-assistance concept:
- Uses YOLOv8 for object detection via IP webcam (streaming from phone)
- Triggers a stop alert when objects are very close (>45% of frame width)
- Optimized the detection pipeline to achieve 25+ FPS 
- Press 's' to get Gemini AI scene description
- Press 'q' to quit
  
  
modelfiner.py - A utility script that:
- Lists all available Google Gemini models that support generateContent
- Used to check which AI models you can access with your API key
