# Banana Split Studio Logic

## Overview
A lightweight AI movie production dashboard built with Streamlit and Google Gemini.

## Requirements
- Python 3.10+
- Google API Key (with access to Gemini 1.5/2.0/3.0 models)

## Setup
1. Clone repo/navigate to folder.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Create a `.env` file (optional, or input key in UI):
   ```
   GOOGLE_API_KEY=your_key_here
   ```

## Usage
1. Run the app:
   ```bash
   streamlit run app.py
   ```
2. **Style Rig**: Upload an image to extract its style DNA.
3. **Character Rig**: Upload a character image to extract DNA and allow for "Visual Lock" in Pro mode.
4. **Script**: Paste your script and click 'Chunk' to break it into shots.
5. **generate**:
   - **Flash (Draft)**: Fast generation using text prompts.
   - **Pro (Final)**: High-quality generation using Text + Character Image Reference.

## Credits

**Developed by:** Steven Lansangan  
**Date:** 2025-12-29

---
