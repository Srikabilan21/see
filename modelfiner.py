import google.generativeai as genai

# PASTE YOUR API KEY HERE
GEMINI_API_KEY = "AIzaSyAveR8EsstrcpJPJCtXDpgIZSU-ncdu6IQ"

genai.configure(api_key=GEMINI_API_KEY)

print("Checking available models...")
try:
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            print(f"- {m.name}")
except Exception as e:
    print(f"Error: {e}")