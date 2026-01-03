# Banana Split Studio
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
from concurrent.futures import ThreadPoolExecutor

# Constants
STATE_FILE = "banana_state.pkl"
ANALYSIS_MODEL = "gemini-2.0-flash-exp" 
DRAFT_MODEL = "gemini-3-pro-image-preview"
FINAL_MODEL = "gemini-3-pro-image-preview"
VIDEO_MODEL = "veo-3.1-generate-preview"

# Safety Config
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

def save_project():
    state = {
        "roster": st.session_state.get("roster", {}),
        "shots": st.session_state.get("shots", []),
        "generated_images": st.session_state.get("generated_images", {}),
        "sketch_style_dna": st.session_state.get("sketch_style_dna", ""),
        "final_style_dna": st.session_state.get("final_style_dna", ""),
        "free_render": st.session_state.get("free_render", None),
        "free_video": st.session_state.get("free_video", None)
    }
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state, f)

def load_project():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "rb") as f:
                data = pickle.load(f)
                st.session_state.update(data)
        except Exception as e:
            st.warning(f"Error loading state: {e}")

st.set_page_config(page_title="Banana Split Studio", layout="wide", page_icon="🍌")
load_dotenv()

# Initialize State
defaults = {
    "sketch_style_dna": "",
    "final_style_dna": "",
    "roster": {},
    "shots": [],
    "generated_images": {},
    "batch_jobs": []
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

if "initialized" not in st.session_state:
    load_project()
    st.session_state["initialized"] = True




# Sidebar
st.sidebar.title("🍌 Banana Split")

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
        st.session_state["roster"] = {}
        st.session_state["shots"] = []
        st.session_state["generated_images"] = {}
        st.session_state["sketch_style_dna"] = ""
        st.session_state["final_style_dna"] = ""
        save_project()
        st.rerun()

st.sidebar.markdown("---")
st.sidebar.subheader("Cast Roster")

# Add Character
with st.sidebar.form("new_char_form", clear_on_submit=True):
    c_name = st.text_input("Name")
    c_file = st.file_uploader("Reference Image", type=["jpg", "png"])
    if st.form_submit_button("Add") and c_name and c_file:
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
            c1, c2, c3 = st.columns([1, 1, 2])
            mode = c1.radio("Mode", ["Fast", "Consistent"], index=1, horizontal=True)
            
            # Generate Sketches
            if c2.button("Generate Sketches", type="primary", use_container_width=True):
                
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

                with st.status("Sketching...", expanded=True) as status:
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
                                st.write(f"✅ Shot {i+1}")
                    
                    save_project()
                    status.update(label="Done!", state="complete", expanded=False)
                    st.rerun()

            # Generate Renders
            if c3.button("Generate Renders", use_container_width=True):
                
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
                
                with st.status("Rendering...", expanded=True) as status:
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
                                st.write(f"✅ Shot {i+1}")
                    
                    save_project()
                    status.update(label="Done!", state="complete", expanded=False)
                    st.rerun()

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
                     st.info("Video coming soon!")

            with c3:
                # View Results
                data = st.session_state['generated_images'].get(i, {})
                if not isinstance(data, dict): data = {} # Safety check
                
                t1, t2 = st.tabs(["Draft", "Final"])
                
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

# Video Tab (Placeholder)
with tab_video:
    st.header("🎥 Video Generation")
    st.info("Feature under construction.")
