import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print(f"Has VideoGenerationModel? {hasattr(genai, 'VideoGenerationModel')}")
print(f"Has ImageGenerationModel? {hasattr(genai, 'ImageGenerationModel')}")
