# Developed by: Steven Lansangan
# Developed by: Steven Lansangan
import google.genai
print("google-genai imported")
try:
    from google.genai import types
    print("types imported")
    print("BatchJobRequest available:", hasattr(types, "BatchJobRequest"))  # Check attribute
    # Print dir to debugging
    # print(dir(types))
except Exception as e:
    print(e)
