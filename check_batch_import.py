# Developed by: Steven Lansangan
# Developed by: Steven Lansangan
import google.genai
print(f"google.genai imported. Version: {google.genai.__version__}")
try:
    from google.genai import types
    print("types imported")
    # In 0.3+, the request objects might be different
    print("BatchJobRequest avail:", hasattr(types, 'BatchJobRequest'))
    print("CreateBatchJobRequest avail:", hasattr(types, 'CreateBatchJobRequest'))
except Exception as e:
    print(f"Error checking types: {e}")
