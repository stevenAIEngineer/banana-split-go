from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=api_key)

if hasattr(client, 'operations'):
    print("Has client.operations")
    print(dir(client.operations))
else:
    print("No client.operations")
