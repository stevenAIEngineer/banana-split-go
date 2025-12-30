import pickle
import os

try:
    with open('project_data.pkl', 'rb') as f:
        data = pickle.load(f)
    print("KEYS:", data.keys())
    if 'roster' in data:
        print(f"Roster Length: {len(data['roster'])}")
        # Check first item structure
        if len(data['roster']) > 0:
            item = data['roster'][0]
            print(f"Roster Item 0 keys: {item.keys()}")
            print(f"Name: {item.get('name')}")
            # Check Image type
            img = item.get('image')
            print(f"Image Type: {type(img)}")
            # Try converting to bytes to see if it works
            import io
            if img:
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                print("Image save check: OK")
except Exception as e:
    print(f"Error checking pickle: {e}")
