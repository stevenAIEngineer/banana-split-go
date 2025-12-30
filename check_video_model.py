from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

models = [
    "veo-2.0-generate-001",
    "veo-3.1-generate-preview",
    "veo-3.1-fast-generate-preview"  # Trying this again just in case
]

print("Testing Video Models...\n")

for m in models:
    print(f"Testing {m}...")
    try:
        # Simple text prompt test
        res = client.models.generate_videos(
            model=m,
            prompt="A spinning banana.",
        )
        print(f"SUCCESS: {m}")
        print(res)
        break # Stop on first success
    except Exception as e:
        print(f"FAILED: {m} - {str(e)}")
        print("-" * 20)
