# Banana Split Studio v1.6
# Developed by: Steven Lansangan

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

st.set_page_config(page_title="Banana Split Studio", layout="wide", page_icon="🍌")
load_dotenv()

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
        st.toast("Welcome back! Session restored. 🍌")
    st.session_state["initialized"] = True

# Sidebar
st.sidebar.title("🍌 Banana Split")

# Display Session Info
st.sidebar.caption(f"ID: `{get_session_id()}`")
st.sidebar.info("Bookmark this URL to return to this session!")

env_key = os.getenv("GOOGLE_API_KEY")
api_key = env_key
if not api_key:
    api_key = st.sidebar.text_input("🔑 API Key", type="password")
    if not api_key: st.stop()

try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.sidebar.error(f"Key Error: {e}")

st.sidebar.markdown("---")

# Data Management
with st.sidebar.expander("⚙️ Manage Data"):
    # Download Button (Manual Backup)
    st.download_button(
        label="💾 Download Backup", 
        data=get_state_bytes(),
        file_name=f"banana_{get_session_id()}.pkl",
        mime="application/octet-stream",
        help="Ideally save this just in case the server restarts!"
    )

    # Restore (Manual Load from file)
    uploaded_state = st.file_uploader("Restore Backup", type=["pkl"], label_visibility="collapsed")
    if uploaded_state:
        if st.button("⚠️ Load Restore File"):
            try:
                data = pickle.load(uploaded_state)
                st.session_state.update(data)
                save_project() # immediately save to current session ID
                st.session_state["initialized"] = False
                st.rerun()
            except Exception as e:
                st.error(f"Restore failed: {e}")

    st.divider()

    c1, c2 = st.columns(2)
    if c1.button("Clear Cast"):
        st.session_state["roster"] = {}
        save_project()
        st.rerun()
    if c2.button("Clear Styles"):
        st.session_state["sketch_style_dna"] = ""
        st.session_state["final_style_dna"] = ""
        save_project()
        st.rerun()
    
    if st.button("Clear All Data", type="primary"):
        for k in defaults.keys():
            st.session_state[k] = defaults[k].copy() if isinstance(defaults[k], (dict, list)) else defaults[k]
        save_project()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Cast Roster")

# Add Character
with st.sidebar.form("new_char_form", clear_on_submit=True):
    c_name = st.text_input("Name")
    c_file = st.file_uploader("Reference Image", type=["jpg", "png"])
    if st.form_submit_button("Add") and c_name and c_file:
        with st.spinner(f"Analyzing {c_name}..."):
            try:
                img = Image.open(c_file).convert("RGB")
                model = genai.GenerativeModel(ANALYSIS_MODEL)
                res = model.generate_content(["Describe physical traits for consistency.", img])
                st.session_state['roster'][c_name] = {"image": img, "dna": res.text}
                save_project()
                st.rerun()
            except Exception as e:
                st.error(e)

# List Characters
if st.session_state['roster']:
    for name, data in list(st.session_state['roster'].items()):
        c1, c2 = st.sidebar.columns([1, 2])
        c1.image(data['image'])
        c2.write(f"**{name}**")
        if c2.button("Remove", key=f"del_{name}"):
            with st.spinner("Removing..."):
                del st.session_state['roster'][name]
                save_project()
                st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Style Config")

tab1, tab2 = st.sidebar.tabs(["Sketch", "Render"])

with tab1:
    s_file = st.file_uploader("Sketch Ref", type=["jpg", "png"], key="s_up")
    if s_file:
        s_img = Image.open(s_file).convert("RGB")
        st.image(s_img)
        if st.button("Analyze Sketch Style"):
            with st.spinner("Analyzing Sketch Style..."):
                m = genai.GenerativeModel(ANALYSIS_MODEL)
                st.session_state['sketch_style_dna'] = m.generate_content(["Describe art style.", s_img]).text
                save_project()
                st.success("Style Locked!")

with tab2:
    f_file = st.file_uploader("Render Ref", type=["jpg", "png"], key="f_up")
    if f_file:
        f_img = Image.open(f_file).convert("RGB")
        st.image(f_img)
        if st.button("Analyze Render Style"):
            with st.spinner("Analyzing Render Style..."):
                m = genai.GenerativeModel(ANALYSIS_MODEL)
                st.session_state['final_style_dna'] = m.generate_content(["Describe art style.", f_img]).text
                save_project()
                st.success("Style Locked!")



st.title("🎬 Storyboard Production")

tab_story, tab_free, tab_video = st.tabs(["Storyboard", "Free Render", "Video"])

with tab_story:
    if not st.session_state["shots"]:
        st.info("Start by pasting your script below.")

    with st.expander("Script Editor", expanded=not bool(st.session_state["shots"])):
        script_text = st.text_area("Input Script", height=150)
        
        if st.button("Process Script"):
            if script_text:
                with st.spinner("Breaking down script into shots..."):
                    try:
                        model = genai.GenerativeModel(ANALYSIS_MODEL, generation_config={"response_mime_type": "application/json"})
                        sys_p = "Convert to JSON list of shots (id, action, dialogue)."
                        res = model.generate_content(f"{sys_p}\nSCRIPT:\n{script_text}")
                        st.session_state['shots'] = json.loads(res.text)
                        save_project()
                        st.rerun()
                    except Exception as e:
                        st.error(e)

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
            if c2.button("1. Generate Sketches", type="primary", use_container_width=True):
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

                with st.spinner("Generating Sketches..."):
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
                    st.toast("✅ Batch Sketches Complete!", icon="✏️")
                    time.sleep(1)
                    st.rerun()

            # 2. Generate Renders (Disabled if no sketches)
            if c3.button("2. Generate Renders", type="primary", use_container_width=True, disabled=not has_sketches):
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
                
                with st.spinner("Rendering Finals..."):
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
                    st.toast("✅ Batch Renders Complete!", icon="🎨")
                    time.sleep(1)
                    st.rerun()

            # 3. Generate Videos (Disabled if no renders)
            if c4.button("3. Generate Videos", type="primary", use_container_width=True, disabled=not has_finals):
                def generate_single_video(idx, shot):
                    try:
                        # 1. Get Image
                        img_data = st.session_state.get('generated_images', {}).get(idx, {})
                        final_img = img_data.get('final')
                        
                        if not final_img:
                            return idx, None, "No final render found."

                        prompt_text = f"Cinematic movement. {shot.get('action', '')}"
                        client = genai_client_lib.Client(api_key=api_key)
                        
                        operation = client.models.generate_videos(
                            model=VIDEO_MODEL,
                            prompt=prompt_text,
                            image=final_img,
                            config={"duration_seconds": 5} 
                        )
                        
                        while not operation.done:
                            time.sleep(10)
                            operation = client.operations.get(operation)
                            
                        if operation.result and operation.result.generated_videos:
                            return idx, operation.result.generated_videos[0].video.uri, None
                        return idx, None, "No video returned."

                    except Exception as e:
                        return idx, None, str(e)

                with st.spinner("Generating Videos... (Processing valid renders only)"):
                    valid_indices = [i for i, _ in enumerate(st.session_state['shots']) 
                                     if 'final' in st.session_state['generated_images'].get(i, {})]
                    
                    if not valid_indices:
                        st.warning("No Final Renders found to animate.")
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
                                    st.error(f"Shot {i+1}: {err}")
                        
                        save_project()
                        st.toast("✅ Batch Videos Complete!", icon="🎥")
                        time.sleep(1)
                        st.rerun()

        st.divider()

        # Bulk Export Section
        with st.expander("📂 Bulk Exports (ZIP/Links)", expanded=False):
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
                    "📦 All Sketches (.zip)", 
                    data=zip_sketches, 
                    file_name="sketches.zip", 
                    mime="application/zip"
                )
            else:
                exp_c1.button("📦 All Sketches", disabled=True)

            # 2. Renders
            zip_finals = create_zip('final')
            if zip_finals:
                exp_c2.download_button(
                    "📦 All Renders (.zip)", 
                    data=zip_finals, 
                    file_name="renders.zip", 
                    mime="application/zip"
                )
            else:
                exp_c2.button("📦 All Renders", disabled=True)

            # 3. Videos (HTML Playlist)
            def create_video_list():
                if not st.session_state['generated_videos']: return None
                html = "<html><body><h1>Generated Videos</h1><ul>"
                for i, uri in st.session_state['generated_videos'].items():
                    html += f"<li><h3>Shot {int(i)+1}</h3><a href='{uri}'>Download Video Link</a><br><video width='320' height='240' controls><source src='{uri}' type='video/mp4'></video></li>"
                html += "</ul></body></html>"
                return html.encode('utf-8')
            
            vid_list = create_video_list()
            if vid_list:
                exp_c3.download_button(
                    "🔗 All Videos (Playlist)", 
                    data=vid_list, 
                    file_name="video_playlist.html", 
                    mime="text/html",
                    help="Downloads an HTML file with links to all your generated videos."
                )
            else:
                 exp_c3.button("🔗 All Videos", disabled=True)
        
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
                if st.button("Draft", key=f"draft_{i}"):
                    with st.spinner("Sketching..."):
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
                if st.button("Final", key=f"final_{i}"):
                    with st.spinner("Rendering..."):
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

                if st.button("Video", key=f"vid_{i}"):
                    # Single Video Generation Logic
                    data = st.session_state['generated_images'].get(i, {})
                    final_img = data.get('final')
                    
                    if not final_img:
                        st.warning("Generate Final Render first!")
                    else:
                        with st.spinner("Generating Video..."):
                            try:
                                prompt_text = f"Cinematic movement. {shot.get('action', '')}"
                                client = genai_client_lib.Client(api_key=api_key)
                                
                                operation = client.models.generate_videos(
                                    model=VIDEO_MODEL,
                                    prompt=prompt_text,
                                    image=final_img,
                                    config={"duration_seconds": 5} 
                                )
                                
                                while not operation.done:
                                    time.sleep(10)
                                    operation = client.operations.get(operation)
                                
                                if operation.result and operation.result.generated_videos:
                                    vid_uri = operation.result.generated_videos[0].video.uri
                                    if i not in st.session_state['generated_videos']:
                                        st.session_state['generated_videos'][i] = {}
                                    st.session_state['generated_videos'][i] = vid_uri
                                    save_project()
                                    st.rerun()
                                else:
                                    st.error("No video returned.")
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
                        st.download_button("Download", data=buf.getvalue(), file_name=f"s{i}_draft.png", key=f"dl_d_{i}")
                
                with t2:
                    if 'final' in data:
                        st.image(data['final'])
                        buf = io.BytesIO()
                        data['final'].save(buf, format="PNG")
                        st.download_button("Download", data=buf.getvalue(), file_name=f"s{i}_final.png", key=f"dl_f_{i}")
                
                with t3:
                    vid_uri = st.session_state.get('generated_videos', {}).get(i)
                    if vid_uri:
                        st.video(vid_uri)
                        st.caption("Right click > Save Video to download")
                    else:
                        st.caption("No video generated yet.")
            
            st.divider()

# Free Render Tab
with tab_free:
    st.header("🎨 Free Render Mode")
    
    col1, col2 = st.columns(2)
    
    with col1:
        up_file = st.file_uploader("Upload Layout", type=["jpg", "png"], key="free_up")
        
        active_cast = list(st.session_state['roster'].keys())
        selected = st.multiselect("Active Cast", active_cast, default=active_cast, key="free_cast")
        
        prompt = st.text_area("Description")
        
        if st.button("Render", type="primary"):
            if not up_file:
                st.warning("Upload sketch first.")
            else:
                with st.spinner("Rendering..."):
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
            
    # Video Section Placeholder (Simplified)
    st.divider()
    st.caption("Video Tools (Coming Soon)")

# Free Video Playground
with tab_video:
    st.header("🎥 Free Video Playground")
    st.caption("Experiment with Text-to-Video and Image-to-Video generation using Veo.")
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        v_prompt = st.text_area("Video Prompt", "A cinematic drone shot of a futuristic city...", height=100)
        v_img = st.file_uploader("Reference Image (Optional)", type=["jpg", "png", "jpeg"], key="vid_k")
        
        if st.button("Generate Video", type="primary", key="gen_free_vid"):
            with st.spinner("Generating Video... (Approx. 1-2 mins)"):
                try:
                    client = genai_client_lib.Client(api_key=api_key)
                    
                    pil_img = None
                    if v_img:
                        pil_img = Image.open(v_img).convert("RGB")
                    
                    # Generate
                    operation = client.models.generate_videos(
                        model=VIDEO_MODEL,
                        prompt=v_prompt,
                        image=pil_img,
                        config={"duration_seconds": 5} 
                    )
                    
                    # Poll
                    while not operation.done:
                        time.sleep(10)
                        operation = client.operations.get(operation)
                        
                    if operation.result and operation.result.generated_videos:
                        uri = operation.result.generated_videos[0].video.uri
                        st.session_state['free_video'] = uri
                        save_project()
                    else:
                        st.error("No video returned.")
                        
                except Exception as e:
                    st.error(f"Error: {e}")

    with col2:
        if st.session_state.get('free_video'):
            st.subheader("Result")
            st.video(st.session_state['free_video'])
        else:
            st.info("Result will appear here.")
