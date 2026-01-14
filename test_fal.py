from dotenv import load_dotenv
import os
import fal_client
import base64
import io
from PIL import Image

# Load environment variables
load_dotenv()

def test_fal_api():
    print("Testing FAL API...")
    
    key = os.getenv("FAL_KEY")
    if not key:
        print("❌ Error: FAL_KEY not found in .env")
        return

    print(f"✅ FAL_KEY found: {key[:5]}...{key[-3:]}")

    # Create dummy red image
    print("Creating dummy image...")
    img = Image.new('RGB', (1024, 576), color='red')
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    img_bytes = buf.getvalue()
    b64_img = base64.b64encode(img_bytes).decode('utf-8')
    data_uri = f"data:image/png;base64,{b64_img}"

    print("Submitting to Kling 2.6 (Pro)...")
    try:
        handler = fal_client.submit(
            "fal-ai/kling-video/v2.6/pro/image-to-video",
            arguments={
                "prompt": "A red screen static shot, minimal movement.",
                "start_image_url": data_uri,
                "duration": "5"
            }
        )
        print(f"✅ Submission Successful! Request ID: {handler.request_id}")
        
        print("Waiting for result (this may take a few minutes)...")
        result = fal_client.result("fal-ai/kling-video/v2.6/pro/image-to-video", handler.request_id)
        
        if 'video' in result:
             print(f"✅ Video Generated: {result['video']['url']}")
        else:
             print(f"❌ Unexpected Result: {result}")

    except Exception as e:
        print(f"❌ API Call Failed: {e}")

if __name__ == "__main__":
    test_fal_api()
