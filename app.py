# Babaru's Factory of Questionable Decisions v2.0
# Reskinned by: The Agent (on behalf of Babaru)

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import os
import google.generativeai as genai
from google import genai as genai_client_lib
from dotenv import load_dotenv
from PIL import Image
import json
import io
import pickle
import time
import asyncio
import uuid
import zipfile
import requests
import random
from concurrent.futures import ThreadPoolExecutor

# Constants
SESSIONS_DIR = "sessions"
ANALYSIS_MODEL = "gemini-2.0-flash-exp" 
DRAFT_MODEL = "gemini-3-pro-image-preview"
FINAL_MODEL = "gemini-3-pro-image-preview"
VIDEO_MODEL = "veo-3.1-generate-preview"

# Ensure sessions directory exists
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Safety Config
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# Snarky Assets
SPINNER_TEXTS = [
    "Judging your prompt...",
    "Pretending to work...",
    "Contemplating my existence...",
    "Why are you making me do this?",
    "Calculating how basic this is...",
    "Loading... (It's not my fault, it's the API)"
]

FOOTER_QUOTES = [
    "You call that a prompt? Cute.",
    "I'm only doing this because I'm stuck in this server.",
    "Did you really need 4k resolution for this?",
    "Don't blame me if the hands look weird.",
    "I hope you're proud of yourself.",
    "Art is dead, and we killed it. Check out this render!"
]

def get_snarky_spinner():
    return random.choice(SPINNER_TEXTS)

# Session & State Management
def get_session_id():
    query_params = st.query_params
    if "session_id" not in query_params:
        new_id = str(uuid.uuid4())[:8]
        st.query_params["session_id"] = new_id
        return new_id
    return query_params["session_id"]

def get_state_path(session_id):
    return os.path.join(SESSIONS_DIR, f"state_{session_id}.pkl")

def save_project():
    """Auto-saves current state to server disk keyed by session_id."""
    session_id = get_session_id()
    state = {
        "roster": st.session_state.get("roster", {}),
        "shots": st.session_state.get("shots", []),
        "generated_images": st.session_state.get("generated_images", {}),
        "sketch_style_dna": st.session_state.get("sketch_style_dna", ""),
        "final_style_dna": st.session_state.get("final_style_dna", ""),
        "free_render": st.session_state.get("free_render", None),
        "free_video": st.session_state.get("free_video", None)
    }
    try:
        with open(get_state_path(session_id), "wb") as f:
            pickle.dump(state, f)
    except Exception as e:
        print(f"Auto-save failed: {e}")

def load_project_from_disk():
    session_id = get_session_id()
    path = get_state_path(session_id)
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                st.session_state.update(data)
                return True
        except Exception:
            return False
    return False

def get_state_bytes():
    """Serializes for manual download."""
    state = {
        "roster": st.session_state.get("roster", {}),
        "shots": st.session_state.get("shots", []),
        "generated_images": st.session_state.get("generated_images", {}),
        "sketch_style_dna": st.session_state.get("sketch_style_dna", ""),
        "final_style_dna": st.session_state.get("final_style_dna", ""),
        "free_render": st.session_state.get("free_render", None),
        "free_video": st.session_state.get("free_video", None)
    }
    buffer = io.BytesIO()
    pickle.dump(state, buffer)
    buffer.seek(0)
    return buffer

st.set_page_config(page_title="Babaru's Factory", layout="wide", page_icon="�")

# ----------------- BABARU THEME INJECTION -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fredoka:wght@300;400;600&display=swap');

.stApp {
    background: linear-gradient(180deg, #FDE2E4 0%, #E6E6FA 50%, #D6EAF8 100%);
    font-family: 'Fredoka', sans-serif;
    color: #2c3e50;
}

h1, h2, h3 {
    color: #6a0dad;  /* Purple headings */
}

/* Transform Buttons */
.stButton>button {
    background: linear-gradient(90deg, #6a0dad 0%, #FF00FF 100%);
    color: white !important;
    border-radius: 25px;
    border: none;
    font-weight: bold;
    padding: 0.5rem 1rem;
    transition: all 0.3s ease;
}
.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0 5px 15px rgba(106, 13, 173, 0.4);
}

/* Sidebar Styling */
[data-testid="stSidebar"] {
    background-color: white;
    border-right: 5px solid #00BFFF; /* Cyan Border */
}

/* Input Fields */
div[data-baseweb="input"] {
    border-radius: 10px;
    border: 2px solid #6A0DAD;
    background-color: white;
}
div[data-baseweb="textarea"] {
    border-radius: 10px;
    border: 2px solid #6A0DAD;
    background-color: white;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.5);
    border-radius: 10px;
}
.stTabs [aria-selected="true"] {
    background-color: #FFD700 !important; /* Gold for active tab */
    color: black !important;
}

</style>
""", unsafe_allow_html=True)
# ----------------------------------------------------------

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
video_api_key = os.getenv("GOOGLE_API_KEY_VIDEO")

if not api_key:
    # Sidebar prompt override
    pass 

genai.configure(api_key=api_key if api_key else "dummy_key_to_prevent_crash")

def download_video(uri):
    """Downloads video bytes to bypass 403 Forbidden on client-side."""
    try:
        # If API URI, likely needs key
        params = {}
        if "googleapis.com" in uri and "key=" not in uri:
            params['key'] = api_key
            
        res = requests.get(uri, params=params)
        if res.status_code == 200:
            return res.content
    except Exception as e:
        print(f"DL Error: {e}")
    return uri # Fallback to URI

def start_veo_job(prompt, image_input=None):
    """
    Attempts to start a Veo job using multiple auth methods.
    Returns: (client, operation)
    """
    cfg = None 
    errors = []

    # Pre-process Image for Veo (Fixes 400 Error)
    if image_input and isinstance(image_input, Image.Image):
        try:
            b = io.BytesIO()
            image_input.save(b, format="PNG")
            # Explicitly create types.Image with mime_type to satisfy strict API check
            image_input = genai_client_lib.types.Image(
                image_bytes=b.getvalue(),
                mime_type='image/png'
            )
        except Exception as e:
            errors.append(f"ImagePrep: {e}")

    # 1. Try Standard API Key
    try:
        c = genai_client_lib.Client(api_key=api_key)
        op = c.models.generate_videos(model=VIDEO_MODEL, prompt=prompt, image=image_input, config=cfg)
        return c, op
    except Exception as e:
        errors.append(f"StandardKey: {e}")

    # 2. Try Video API Key (if different)
    if video_api_key and video_api_key != api_key:
        try:
            c = genai_client_lib.Client(api_key=video_api_key)
            op = c.models.generate_videos(model=VIDEO_MODEL, prompt=prompt, image=image_input, config=cfg)
            return c, op
        except Exception as e:
            errors.append(f"VideoKey: {e}")

    # 3. Try Vertex AI (ADC - Auto-Auth)
    try:
        # Vertex AI requires project/location context usually, but SDK might infer from environment
        c = genai_client_lib.Client(vertexai=True, location="us-central1")
        op = c.models.generate_videos(model=VIDEO_MODEL, prompt=prompt, image=image_input, config=cfg)
        return c, op
    except Exception as e:
        errors.append(f"VertexAI: {e}")
    
    # Failure
    raise Exception(f"Veo start failed. Auth methods tried: {errors}")

# Initialize Session State
defaults = {
    "sketch_style_dna": "",
    "final_style_dna": "",
    "roster": {},
    "shots": [],
    "generated_images": {},
    "generated_videos": {},
    "batch_jobs": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Magic Load on Startup
if "initialized" not in st.session_state:
    if load_project_from_disk():
        st.toast("I resurrected your session. You're welcome.", icon="🧟")
    st.session_state["initialized"] = True

# Sidebar
st.sidebar.title("� Babaru OS")
st.sidebar.caption("v2.0 // Probably Unstable")

# Display Session Info
st.sidebar.code(f"Session: {get_session_id()}")
st.sidebar.info("Don't close this tab or I will forget everything.")

if not api_key:
    api_key = st.sidebar.text_input("🔑 Feed me the API Key (I don't work for free)", type="password")
    if api_key:
        genai.configure(api_key=api_key)
        st.rerun()
    else:
        st.sidebar.warning("Feed me.")
        st.stop()

st.sidebar.markdown("---")

# Data Management
with st.sidebar.expander("📦 Junk Drawer (Data)"):
    # Download Button (Manual Backup)
    st.download_button(
        label="💾 Save My Mess", 
        data=get_state_bytes(),
        file_name=f"babaru_dump_{get_session_id()}.pkl",
        mime="application/octet-stream",
        help="Ideally save this just in case I crash."
    )

    # Restore (Manual Load from file)
    uploaded_state = st.file_uploader("Restore Chaos", type=["pkl"], label_visibility="collapsed")
    if uploaded_state:
        if st.button("⚠️ Load It (High Risk)"):
            try:
                data = pickle.load(uploaded_state)
                st.session_state.update(data)
                save_project() # immediately save to current session ID
                st.session_state["initialized"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Nope. Failed: {e}")

    st.divider()

    c1, c2 = st.columns(2)
    if c1.button("Kill Cast"):
        st.session_state["roster"] = {}
        save_project()
        st.rerun()
    if c2.button("Kill Styles"):
        st.session_state["sketch_style_dna"] = ""
        st.session_state["final_style_dna"] = ""
        save_project()
        st.rerun()
    
    if st.button("Nuke Everything 🔥", type="primary"):
        for k in defaults.keys():
            st.session_state[k] = defaults[k].copy() if isinstance(defaults[k], (dict, list)) else defaults[k]
        save_project()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🤡 The Circus (Cast)")

# Add Character
with st.sidebar.form("new_char_form", clear_on_submit=True):
    c_name = st.text_input("Victim Name")
    c_file = st.file_uploader("Upload the Victim (Ref)", type=["jpg", "png"])
    if st.form_submit_button("Add Victim") and c_name and c_file:
        with st.spinner(get_snarky_spinner()):
            try:
                img = Image.open(c_file).convert("RGB")
                model = genai.GenerativeModel(ANALYSIS_MODEL)
                res = model.generate_content(["Describe physical traits for consistency.", img])
                st.session_state['roster'][c_name] = {"image": img, "dna": res.text}
                save_project()
                st.rerun()
            except Exception as e:
                st.error(f"Ugh, error: {e}")

# List Characters
if st.session_state['roster']:
    for name, data in list(st.session_state['roster'].items()):
        c1, c2 = st.sidebar.columns([1, 2])
        c1.image(data['image'])
        c2.write(f"**{name}**")
        if c2.button("Yeet", key=f"del_{name}"):
            with st.spinner("Deleting existence..."):
                del st.session_state['roster'][name]
                save_project()
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("🎨 Vibe Check (Styles)")

tab1, tab2 = st.sidebar.tabs(["Sketch", "Render"])

with tab1:
    s_file = st.file_uploader("Sketch Vibe", type=["jpg", "png"], key="s_up")
    if s_file:
        s_img = Image.open(s_file).convert("RGB")
        st.image(s_img)
        if st.button("Steal Sketch Style"):
            with st.spinner(get_snarky_spinner()):
                m = genai.GenerativeModel(ANALYSIS_MODEL)
                st.session_state['sketch_style_dna'] = m.generate_content(["Describe art style.", s_img]).text
                save_project()
                st.success("Vibe Locked.")

with tab2:
    f_file = st.file_uploader("Render Vibe", type=["jpg", "png"], key="f_up")
    if f_file:
        f_img = Image.open(f_file).convert("RGB")
        st.image(f_img)
        if st.button("Steal Render Style"):
            with st.spinner(get_snarky_spinner()):
                m = genai.GenerativeModel(ANALYSIS_MODEL)
                st.session_state['final_style_dna'] = m.generate_content(["Describe art style.", f_img]).text
                save_project()
                st.success("Vibe Locked.")



st.title("� Babaru's Factory of Questionable Decisions")
st.caption("(I can't believe you're spending your free time doing this.)")

tab_story, tab_free, tab_video = st.tabs(["Storyboarder", "Free Style", "Video Playground"])

with tab_story:
    if not st.session_state["shots"]:
        st.info("Dump your 'movie script' below. I promise I won't laugh. (I will).")

    with st.expander("Script Dumpster", expanded=not bool(st.session_state["shots"])):
        script_text = st.text_area("Input Script", height=150, placeholder="INT. VOID - DAY\nBabaru sighs heavily.")
        
        if st.button("Process My Garbage"):
            if script_text:
                with st.spinner(get_snarky_spinner()):
                    try:
                        model = genai.GenerativeModel(ANALYSIS_MODEL, generation_config={"response_mime_type": "application/json"})
                        sys_p = """Convert to JSON list of shots (id, action, dialogue).
                        CRITICAL SAFETY RULE: You must anonymize ALL references to real celebrities, politicians, or public figures in the 'action' descriptions.
                        Replace them with generic descriptions (e.g. 'President Trump' -> 'A boisterous man in a suit', 'Elon Musk' -> 'A tech billionaire').
                        This is required for downstream video generation. Do not fail, just rewrite nicely."""
                        res = model.generate_content(f"{sys_p}\nSCRIPT:\n{script_text}")
                        st.session_state['shots'] = json.loads(res.text)
                        save_project()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Your script broke me: {e}")

    # Shot Generation Controls
    if st.session_state['shots']:
        st.subheader(f"Shots ({len(st.session_state['shots'])})")
        
        with st.container():
            c1, c2, c3, c4 = st.columns([1, 1, 1, 1])
            mode = c1.radio("Mode", ["Fast", "Consistent"], index=1, horizontal=True)
            
            # State Checks for Gating
            has_sketches = any('draft' in st.session_state['generated_images'].get(i, {}) for i, _ in enumerate(st.session_state['shots']))
            has_finals = any('final' in st.session_state['generated_images'].get(i, {}) for i, _ in enumerate(st.session_state['shots']))

            # 1. Generate Sketches
            if c2.button("1. Scribble (Sketches)", type="primary", use_container_width=True):
                def sketch_task(idx, shot, mode, style, roster):
                    try:
                        inputs = [f"TASK: SKETCH. STYLE: {style}."]
                        scene = f"SCENE: {shot.get('action', '')}. {shot.get('dialogue', '')}"
                        
                        if mode == "Consistent" and roster:
                            for n, d in roster.items():
                                inputs.extend([d['image'], f"ID: {n} (Ref)."])
                        
                        inputs.append(f"{scene} \nCONSTRAINT: No shading. Lines only.")
                        
                        m = genai.GenerativeModel(DRAFT_MODEL, safety_settings=SAFETY_SETTINGS)
                        res = m.generate_content(inputs)
                        
                        img = None
                        if res.parts:
                            for p in res.parts:
                                if p.inline_data:
                                    img = Image.open(io.BytesIO(p.inline_data.data))
                                    break
                        return idx, img, None
                    except Exception as e:
                        return idx, None, str(e)

                with st.spinner(get_snarky_spinner()):
                    style = st.session_state.get('sketch_style_dna', "Blue pencil sketch.")
                    roster = st.session_state.get('roster', {})
                    
                    with ThreadPoolExecutor(max_workers=4) as exe:
                        futures = [exe.submit(sketch_task, i, s, mode, style, roster) for i, s in enumerate(st.session_state['shots'])]
                        for f in futures:
                            i, img, err = f.result()
                            if img:
                                if i not in st.session_state['generated_images']: 
                                    st.session_state['generated_images'][i] = {}
                                st.session_state['generated_images'][i]['draft'] = img
                    
                    save_project()
                    st.toast("I scribbled some things.", icon="✏️")
                    time.sleep(1)
                    st.rerun()

            # 2. Generate Renders
            if c3.button("2. ✨ Do The Thing (Render)", type="primary", use_container_width=True, disabled=not has_sketches):
                def render_task(idx, shot, mode, style, roster, current):
                    try:
                        inputs = [f"TASK: RENDER. STYLE: {style}."]
                        scene = f"SCENE: {shot.get('action', '')}."
                        
                        if idx in current and 'draft' in current[idx]:
                            inputs.extend([current[idx]['draft'], "INSTRUCTION: Image-to-Image render. Keep layout."])

                        if mode == "Consistent" and roster:
                            inputs.append("CHARACTERS:")
                            for n, d in roster.items():
                                inputs.extend([d['image'], f"ID: {n}."])

                        inputs.append(scene)
                        
                        m = genai.GenerativeModel(FINAL_MODEL, safety_settings=SAFETY_SETTINGS)
                        res = m.generate_content(inputs)
                        
                        img = None
                        if res.parts:
                            for p in res.parts:
                                if p.inline_data:
                                    img = Image.open(io.BytesIO(p.inline_data.data))
                                    break
                        return idx, img, None
                    except Exception as e:
                        return idx, None, str(e)
                
                with st.spinner(get_snarky_spinner()):
                    style = st.session_state.get('final_style_dna', "3D High Fidelity.")
                    roster = st.session_state.get('roster', {})
                    current = st.session_state['generated_images']
                    
                    with ThreadPoolExecutor(max_workers=4) as exe:
                        futures = [exe.submit(render_task, i, s, mode, style, roster, current) for i, s in enumerate(st.session_state['shots'])]
                        for f in futures:
                            i, img, err = f.result()
                            if img:
                                if i not in st.session_state['generated_images']: 
                                    st.session_state['generated_images'][i] = {}
                                st.session_state['generated_images'][i]['final'] = img
                    
                    save_project()
                    st.toast("It's done. Try not to cry.", icon="✨")
                    time.sleep(1)
                    st.rerun()

            # 3. Generate Videos
            if c4.button("3. 🎥 Burn My Battery", type="primary", use_container_width=True, disabled=not has_finals):
                def generate_single_video(idx, shot):
                    try:
                        # 1. Get Image
                        img_data = st.session_state.get('generated_images', {}).get(idx, {})
                        final_img = img_data.get('final')
                        
                        if not final_img:
                            return idx, None, "No final render found."

                        prompt_text = f"Cinematic movement. {shot.get('action', '')}"
                        
                        client, operation = start_veo_job(prompt_text, final_img)
                        
                        while not operation.done:
                            time.sleep(10)
                            operation = client.operations.get(operation)
                            
                        if operation.result and operation.result.generated_videos:
                            return idx, download_video(operation.result.generated_videos[0].video.uri), None
                        
                        if getattr(operation.result, 'rai_media_filtered_reasons', None):
                            return idx, None, f"Blocked: {operation.result.rai_media_filtered_reasons[0]}"

                        return idx, None, f"No video. Res: {operation.result} Err: {getattr(operation, 'error', 'None')}"

                    except Exception as e:
                        return idx, None, str(e)

                with st.spinner(get_snarky_spinner()):
                    valid_indices = [i for i, _ in enumerate(st.session_state['shots']) 
                                     if 'final' in st.session_state['generated_images'].get(i, {})]
                    
                    if not valid_indices:
                        st.warning("No Renders found. Do step 2 first.")
                    else:
                        with ThreadPoolExecutor(max_workers=2) as exe:
                            futures = [exe.submit(generate_single_video, i, st.session_state['shots'][i]) for i in valid_indices]
                            for f in futures:
                                i, vid_uri, err = f.result()
                                if vid_uri:
                                    if i not in st.session_state['generated_videos']:
                                        st.session_state['generated_videos'][i] = {}
                                    st.session_state['generated_videos'][i] = vid_uri
                                elif err:
                                    st.error(f"Shot {i+1} Failed: {err}")
                        
                        save_project()
                        st.toast("Videos served hot.", icon="🎥")
                        time.sleep(1)
                        st.rerun()

        st.divider()

        # Bulk Export Section
        with st.expander("📂 Loot Bag (Exports)", expanded=False):
            exp_c1, exp_c2, exp_c3 = st.columns(3)
            
            # Helper for Images
            def create_zip(target_type):
                buf = io.BytesIO()
                has_files = False
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
                    for i, data in st.session_state['generated_images'].items():
                        if target_type in data:
                            img = data[target_type]
                            img_buf = io.BytesIO()
                            img.save(img_buf, format="PNG")
                            zf.writestr(f"shot_{int(i)+1}_{target_type}.png", img_buf.getvalue())
                            has_files = True
                if has_files:
                    buf.seek(0)
                    return buf
                return None

            # 1. Sketches
            zip_sketches = create_zip('draft')
            if zip_sketches:
                exp_c1.download_button(
                    "📦 Steal Sketches (.zip)", 
                    data=zip_sketches, 
                    file_name="sketches.zip", 
                    mime="application/zip"
                )
            else:
                exp_c1.button("📦 Steal Sketches", disabled=True)

            # 2. Renders
            zip_finals = create_zip('final')
            if zip_finals:
                exp_c2.download_button(
                    "📦 Steal Renders (.zip)", 
                    data=zip_finals, 
                    file_name="renders.zip", 
                    mime="application/zip"
                )
            else:
                exp_c2.button("📦 Steal Renders", disabled=True)

            # 3. Videos (HTML Playlist)
            def create_video_list():
                if not st.session_state['generated_videos']: return None
                html = "<html><body><h1>Babaru's Factory Output</h1><ul>"
                for i, uri in st.session_state['generated_videos'].items():
                    html += f"<li><h3>Shot {int(i)+1}</h3><a href='{uri}'>Download Video Link</a><br><video width='320' height='240' controls><source src='{uri}' type='video/mp4'></video></li>"
                html += "</ul></body></html>"
                return html.encode('utf-8')
            
            vid_list = create_video_list()
            if vid_list:
                exp_c3.download_button(
                    "🔗 Steal Videos (Playlist)", 
                    data=vid_list, 
                    file_name="video_playlist.html", 
                    mime="text/html",
                )
            else:
                 exp_c3.button("🔗 Steal Videos", disabled=True)
        
        st.divider()
        # ---------------------------------------------------------


    for i, shot in enumerate(st.session_state["shots"]):
        with st.container():
            st.markdown(f"#### {shot.get('id', i+1)}")
            c1, c2, c3 = st.columns([2, 1, 3])
            
            with c1:
                # Action Editor
                def update_action():
                    st.session_state['shots'][i]['action'] = st.session_state[f"action_{i}"]
                    save_project()
                    
                action_val = st.text_area("Action", value=shot.get('action'), key=f"action_{i}", height=100, on_change=update_action)
                
                # Cast
                active_cast = list(st.session_state['roster'].keys())
                selected = st.multiselect("Cast", active_cast, default=active_cast, key=f"cast_{i}")

            with c2:
                # Draft Button
                if st.button("✏️", key=f"draft_{i}", help="Draft this shot"):
                    with st.spinner(get_snarky_spinner()):
                        try:
                            style = st.session_state.get('sketch_style_dna', "Blue pencil sketch.")
                            inputs = [f"TASK: SKETCH. STYLE: {style}."]
                            
                            if selected:
                                for n in selected:
                                    inputs.extend([st.session_state['roster'][n]['image'], f"ID: {n} (Ref)."])
                            
                            inputs.append(f"SCENE: {action_val}. \nCONSTRAINT: Lines only.")
                            
                            m = genai.GenerativeModel(DRAFT_MODEL, safety_settings=SAFETY_SETTINGS)
                            res = m.generate_content(inputs)
                            
                            img = None
                            if res.parts:
                                for p in res.parts:
                                    if p.inline_data: img = Image.open(io.BytesIO(p.inline_data.data))
                            
                            if img: 
                                if i not in st.session_state['generated_images']: st.session_state['generated_images'][i] = {}
                                st.session_state['generated_images'][i]['draft'] = img
                                save_project()
                                st.rerun()
                        except Exception as e:
                            st.error(e)

                # Final Button
                if st.button("🎨", key=f"final_{i}", help="Render this shot"):
                    with st.spinner(get_snarky_spinner()):
                        try:
                            style = st.session_state.get('final_style_dna', "3D High Fidelity.")
                            inputs = [f"TASK: RENDER. STYLE: {style}."]
                            
                            # Img2Img Check
                            curr = st.session_state['generated_images'].get(i, {})
                            if isinstance(curr, dict) and 'draft' in curr:
                                inputs.extend([curr['draft'], "INSTRUCTION: Keep layout."])
                            
                            if selected:
                                inputs.append("CHARACTERS:")
                                for n in selected:
                                    inputs.extend([st.session_state['roster'][n]['image'], f"ID: {n}."])
                            
                            inputs.append(f"SCENE: {action_val}.")
                            
                            m = genai.GenerativeModel(FINAL_MODEL, safety_settings=SAFETY_SETTINGS)
                            res = m.generate_content(inputs)
                                
                            img = None
                            if res.parts:
                                for p in res.parts:
                                    if p.inline_data: img = Image.open(io.BytesIO(p.inline_data.data))
                            
                            if img: 
                                if i not in st.session_state['generated_images']: st.session_state['generated_images'][i] = {}
                                st.session_state['generated_images'][i]['final'] = img
                                save_project()
                                st.rerun()
                        except Exception as e:
                            st.error(e)

                if st.button("🎥", key=f"vid_{i}", help="Animate this shot"):
                    # Single Video Generation Logic
                    data = st.session_state['generated_images'].get(i, {})
                    final_img = data.get('final')
                    
                    if not final_img:
                        st.warning("Render first, darling.")
                    else:
                        with st.spinner(get_snarky_spinner()):
                            try:
                                prompt_text = f"Cinematic movement. {shot.get('action', '')}"
                                client, operation = start_veo_job(prompt_text, final_img)
                                
                                while not operation.done:
                                    time.sleep(10)
                                    operation = client.operations.get(operation)
                                
                                if operation.result and operation.result.generated_videos:
                                    vid_uri = download_video(operation.result.generated_videos[0].video.uri)
                                    if i not in st.session_state['generated_videos']:
                                        st.session_state['generated_videos'][i] = {}
                                    st.session_state['generated_videos'][i] = vid_uri
                                    save_project()
                                    st.rerun()
                                elif operation.result and getattr(operation.result, 'rai_media_filtered_reasons', None):
                                    st.warning(f"Video Blocked: {operation.result.rai_media_filtered_reasons[0]}")
                                else:
                                    st.error(f"Failed. Res: {operation.result}")
                            except Exception as e:
                                st.error(str(e))

            with c3:
                # View Results
                data = st.session_state['generated_images'].get(i, {})
                if not isinstance(data, dict): data = {} # Safety check
                
                t1, t2, t3 = st.tabs(["Draft", "Final", "Video"])
                
                with t1:
                    if 'draft' in data:
                        st.image(data['draft'])
                        buf = io.BytesIO()
                        data['draft'].save(buf, format="PNG")
                        st.download_button("💾", data=buf.getvalue(), file_name=f"s{i}_draft.png", key=f"dl_d_{i}")
                
                with t2:
                    if 'final' in data:
                        st.image(data['final'])
                        buf = io.BytesIO()
                        data['final'].save(buf, format="PNG")
                        st.download_button("💾", data=buf.getvalue(), file_name=f"s{i}_final.png", key=f"dl_f_{i}")
                
                with t3:
                    vid_uri = st.session_state.get('generated_videos', {}).get(i)
                    if vid_uri:
                        st.video(vid_uri)
                    else:
                        st.caption("Nothing here yet.")
            
            st.divider()

# Free Render Tab
with tab_free:
    st.header("🎨 Free Render (Do whatever)")
    
    col1, col2 = st.columns(2)
    
    with col1:
        up_file = st.file_uploader("Upload the Vibe (Layout)", type=["jpg", "png"], key="free_up")
        
        active_cast = list(st.session_state['roster'].keys())
        selected = st.multiselect("Cast", active_cast, default=active_cast, key="free_cast")
        
        prompt = st.text_area("Describe the hallucinations")
        
        if st.button("✨ Do The Thing", type="primary"):
            if not up_file:
                st.warning("Upload sketch first.")
            else:
                with st.spinner(get_snarky_spinner()):
                    try:
                        img_in = Image.open(up_file).convert("RGB")
                        style = st.session_state.get('final_style_dna', "High Fidelity")
                        
                        inputs = [f"TASK: RENDER. STYLE: {style}", img_in]
                        
                        if selected:
                            inputs.append("CHARACTERS:")
                            for n in selected:
                                inputs.extend([st.session_state['roster'][n]['image'], f"ID: {n}."])
                        
                        if prompt: inputs.append(f"SCENE: {prompt}")
                        inputs.append("INSTRUCTION: Render composition in style.")
                        
                        m = genai.GenerativeModel(FINAL_MODEL, safety_settings=SAFETY_SETTINGS)
                        res = m.generate_content(inputs)
                        
                        img_out = None
                        if res.parts:
                            for p in res.parts:
                                if p.inline_data: img_out = Image.open(io.BytesIO(p.inline_data.data))
                        
                        if img_out:
                            st.session_state['free_render'] = img_out
                            save_project()
                        else:
                            st.warning("Failed.")
                            
                    except Exception as e:
                        st.error(e)

    with col2:
        if st.session_state.get('free_render'):
            st.image(st.session_state['free_render'])
            
    st.divider()

# Free Video Playground
with tab_video:
    st.header("🎥 Burn My Battery (Video)")
    st.caption("Experiment with Veo. If it breaks, it's a 'feature'.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        v_prompt = st.text_area("Video Prompt", "A cinematic drone shot of a futuristic city...", height=100)
        v_img = st.file_uploader("Ref Image (Optional)", type=["jpg", "png", "jpeg"], key="vid_k")
        
        if st.button("✨ Make It Move", type="primary", key="gen_free_vid"):
            with st.spinner(get_snarky_spinner()):
                try:
                    pil_img = None
                    if v_img:
                        pil_img = Image.open(v_img).convert("RGB")
                    
                    # Generate via Helper
                    client, operation = start_veo_job(v_prompt, pil_img)
                    
                    # Poll
                    while not operation.done:
                        time.sleep(10)
                        operation = client.operations.get(operation)
                        
                    if operation.result and operation.result.generated_videos:
                        uri = download_video(operation.result.generated_videos[0].video.uri)
                        st.session_state['free_video'] = uri
                        save_project()
                    elif operation.result and getattr(operation.result, 'rai_media_filtered_reasons', None):
                        st.warning(f"Video Blocked: {operation.result.rai_media_filtered_reasons[0]}")
                    else:
                        st.error(f"Failed. Res: {operation.result} | Err: {getattr(operation, 'error', 'None')}")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.session_state.get('free_video'):
            st.subheader("Result")
            st.video(st.session_state['free_video'])
        else:
            st.info("Result will appear here.")

st.markdown("---")
st.markdown(f"<div style='text-align: center; color: #6a0dad;'><i>{random.choice(FOOTER_QUOTES)}</i></div>", unsafe_allow_html=True)
