# Banana Split Studio
# Multi-Style Generative Storyboard Application
# Developed by: Steven Lansangan
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import streamlit as st
import os
import google.generativeai as genai
from google import genai as genai_client_lib
from google.genai import types
from dotenv import load_dotenv
from PIL import Image

import json
import io
import pickle
import time

# State Persistence
STATE_FILE = "banana_state.pkl"

def save_project():
    """Saves critical session state to disk."""
    state_data = {
        "roster": st.session_state.get("roster", {}),
        "shots": st.session_state.get("shots", []),
        "generated_images": st.session_state.get("generated_images", {}),
        "sketch_style_dna": st.session_state.get("sketch_style_dna", ""),
        "final_style_dna": st.session_state.get("final_style_dna", ""),
        "free_render": st.session_state.get("free_render", None),
        "free_video": st.session_state.get("free_video", None),
        "batch_jobs": st.session_state.get("batch_jobs", [])
    }
    with open(STATE_FILE, "wb") as f:
        pickle.dump(state_data, f)

def load_project():
    """Loads session state from disk if exists."""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, "rb") as f:
                state_data = pickle.load(f)
                st.session_state["roster"] = state_data.get("roster", {})
                st.session_state["shots"] = state_data.get("shots", [])
                st.session_state["generated_images"] = state_data.get("generated_images", {})
                st.session_state["sketch_style_dna"] = state_data.get("sketch_style_dna", "")
                st.session_state["final_style_dna"] = state_data.get("final_style_dna", "")
                st.session_state["free_render"] = state_data.get("free_render", None)
                st.session_state["free_video"] = state_data.get("free_video", None)
                st.session_state["batch_jobs"] = state_data.get("batch_jobs", [])
        except Exception as e:
            st.warning(f"Could not load previous session: {e}")

# Config
st.set_page_config(page_title="Banana Split Studio", layout="wide", page_icon="🍌")

# Load Env
load_dotenv()
env_key = os.getenv("GOOGLE_API_KEY")

# Session State Init
if "sketch_style_dna" not in st.session_state:
    st.session_state["sketch_style_dna"] = ""
if "final_style_dna" not in st.session_state:
    st.session_state["final_style_dna"] = ""
if "roster" not in st.session_state:
    st.session_state["roster"] = {} # Format: { "Name": {"image": PIL_Img, "dna": "desc"} }
if "shots" not in st.session_state:
    st.session_state["shots"] = []
if "generated_images" not in st.session_state:
    st.session_state["generated_images"] = {}
if "batch_jobs" not in st.session_state:
    st.session_state["batch_jobs"] = []
if "initialized" not in st.session_state:
    load_project()
    st.session_state["initialized"] = True

# Constants
ANALYSIS_MODEL = "gemini-2.0-flash-exp" 
DRAFT_MODEL = "gemini-3-pro-image-preview"
FINAL_MODEL = "gemini-3-pro-image-preview"
VIDEO_MODEL = "veo-3.1-generate-preview"

# Safety Settings (Standard)
SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]

# ---------------------------------------------------------


# Batch Processing
def submit_batch_job(job_type, api_key):
    """Submits a Batch Job using JSONL upload."""
    try:
        client = genai_client_lib.Client(api_key=api_key)
        
        # Prepare Requests (Lines for JSONL)
        jsonl_lines = []
        shots = st.session_state["shots"]
        roster = st.session_state.get("roster", {})
        
        for i, shot in enumerate(shots):
            req_id = f"{job_type}_{i}"
            request_body = {}
            
            if job_type == "sketch":
                s_style = st.session_state.get('sketch_style_dna', "Rough professional storyboard sketch.")
                prompt_text = f"TASK: STORYBOARD SKETCH. STYLE: {s_style}.\n"
                if roster:
                    prompt_text += "CHARACTERS: " + ", ".join([f"{n}" for n in roster.keys()]) + ".\n"
                prompt_text += f"SCENE: {shot.get('action', '')}. {shot.get('dialogue', '')}"
                
                # Request Body
                request_body = {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generation_config": {"temperature": 0.5}
                }
                model_id = DRAFT_MODEL

            elif job_type == "final":
                 # Text-only description for stability in batch mode
                 style_ref = st.session_state.get('final_style_dna', "3D Pixar Style.")
                 prompt_text = f"TASK: FINAL RENDER. STYLE: {style_ref}. SCENE: {shot.get('action', '')}."
                 request_body = {
                    "contents": [{"parts": [{"text": prompt_text}]}],
                    "generation_config": {"temperature": 0.3}
                 }
                 model_id = FINAL_MODEL

            # Full JSONL Object
            json_line = {
                "custom_id": req_id,
                "request": request_body
            }
            jsonl_lines.append(json.dumps(json_line))
            
        if not jsonl_lines:
            return None, "No tasks."
            
        # Write to temp JSONL file
        temp_filename = f"batch_{job_type}_{int(time.time())}.jsonl"
        with open(temp_filename, "w") as f:
            f.write("\n".join(jsonl_lines))
            
        # Upload to Gemini
        batch_file = client.files.upload(file=temp_filename, config={"mime_type": "application/json"})
        
        # Create Job
        job = client.batches.create(
            model=model_id,
            src=batch_file.name,
            config={"display_name": f"BananaSplit_{job_type}"}
        )
        
        # Cleanup temp file
        os.remove(temp_filename)
        
        return job, None

    except Exception as e:
        return None, str(e)

def check_batch_updates(api_key):
    """Checks and applies results for active batches."""
    if not st.session_state["batch_jobs"]:
        return 0 
        
    client = genai_client_lib.Client(api_key=api_key)
    completed_count = 0
    
    active_jobs = st.session_state["batch_jobs"][:] # Copy to iterate
    
    for job_info in active_jobs:
        job_id = job_info['id']
        try:
            # Poll Status
            refresh_job = client.batches.get(name=job_id)
            
            if refresh_job.state == "SUCCEEDED":
                # Download Results
                # Placeholder for manual check
                job_info['status'] = 'COMPLETED (Download Manually)'
                
                st.toast(f"Batch {job_id} Completed!", icon="✅")
                st.session_state["batch_jobs"].remove(job_info)
                completed_count += 1
                
            elif refresh_job.state == "FAILED":
                job_info['status'] = f"FAILED: {refresh_job.error.message}"
                st.session_state["batch_jobs"].remove(job_info) # Remove failed
                st.error(f"Batch {job_id} Failed.")
                
            else:
                job_info['status'] = refresh_job.state # Updating status (RUNNING/etc)
                
        except Exception as e:
            print(f"Batch Check Error: {e}")
            
    save_project()
    return completed_count

st.sidebar.title("🍌 Banana Split")

# API Key
api_key = env_key
if not api_key:
    api_key = st.sidebar.text_input("🔑 API Key", type="password")
    if not api_key:
        st.sidebar.warning("Key required.")
        st.stop()
try:
    genai.configure(api_key=api_key)
except Exception as e:
    st.sidebar.error(f"Key Error: {e}")

st.sidebar.markdown("---")

# Sidebar UI
with st.sidebar.expander("⚙️ Manage Data", expanded=False):
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
    
    c3, c4 = st.columns(2)
    if c3.button("Clear Storyboard"):
        st.session_state["shots"] = []
        st.session_state["generated_images"] = {}
        save_project()
        st.rerun()
    if c4.button("Clear Free Render"):
        st.session_state["free_render"] = None
        st.session_state["free_video"] = None
        save_project()
        st.rerun()
        
    if st.button("⚠️ Factory Reset (All)", type="primary"):
        st.session_state["roster"] = {}
        st.session_state["shots"] = []
        st.session_state["generated_images"] = {}
        st.session_state["sketch_style_dna"] = ""
        st.session_state["final_style_dna"] = ""
        st.session_state["free_render"] = None
        st.session_state["free_video"] = None
        save_project()
        st.rerun()

st.sidebar.markdown("---")

# --- ROSTER UI ---
st.sidebar.header("👥 Cast Roster")

# Add Character Form
with st.sidebar.form("new_char_form", clear_on_submit=True):
    st.caption("➕ Add a new character")
    c_name = st.text_input("Name", placeholder="e.g. Babaru")
    c_file = st.file_uploader("Reference Image", type=["jpg", "png"])
    submitted = st.form_submit_button("Add to Cast")
    
    if submitted and c_name and c_file:
        with st.spinner("Analyzing Traits..."):
            try:
                img = Image.open(c_file).convert("RGB")
                model = genai.GenerativeModel(ANALYSIS_MODEL)
                dna_prompt = "Analyze physical traits (species, colors, clothes) for character consistency. Concise."
                res = model.generate_content([dna_prompt, img])
                
                st.session_state['roster'][c_name] = {"image": img, "dna": res.text}
                save_project()
                st.toast(f"✅ {c_name} added!")
                st.rerun()
            except Exception as e:
                st.error(f"Failed: {e}")

# Cast Grid Display
if st.session_state['roster']:
    st.sidebar.caption("Active Cast")
    # Custom CSS for compact grid could go here, but we use columns for simplicity
    for name, data in list(st.session_state['roster'].items()):
        c1, c2, c3 = st.sidebar.columns([1, 2, 1])
        c1.image(data['image'], use_container_width=True)
        c2.write(f"**{name}**")
        if c3.button("×", key=f"del_{name}", help="Remove"):
            del st.session_state['roster'][name]
            save_project()
            st.rerun()
else:
    st.sidebar.info("Roster is empty. Add a character above!")

st.sidebar.markdown("---")
st.sidebar.header("🎨 Style Configuration")

tab_s1, tab_s2 = st.sidebar.tabs(["Draft (Sketch)", "Final (Render)"])

with tab_s1:
    st.caption("Style for Sketches (Draft)")
    s_file = st.file_uploader("Sketch Ref", type=["jpg", "png"], key="s_up")
    if s_file:
        s_img = Image.open(s_file).convert("RGB")
        st.image(s_img, use_container_width=True)
        if st.button("Analyze Sketch Style"):
            with st.spinner("Analyzing..."):
                m = genai.GenerativeModel(ANALYSIS_MODEL)
                st.session_state['sketch_style_dna'] = m.generate_content(["Describe art style.", s_img]).text
                save_project()
                st.success("Sketch Style Locked!")

with tab_s2:
    st.caption("Style for Finals (Render)")
    f_file = st.file_uploader("Render Ref", type=["jpg", "png"], key="f_up")
    if f_file:
        f_img = Image.open(f_file).convert("RGB")
        st.image(f_img, use_container_width=True)
        if st.button("Analyze Render Style"):
            with st.spinner("Analyzing..."):
                m = genai.GenerativeModel(ANALYSIS_MODEL)
                st.session_state['final_style_dna'] = m.generate_content(["Describe art style.", f_img]).text
                save_project()
                st.success("Final Style Locked!")



st.title("🎬 Storyboard Production")

tab_story, tab_free, tab_video = st.tabs(["🎬 Storyboard Mode", "🎨 Free Render / Revise", "🎥 Free Video Studio"])

# Main Tabs
with tab_story:
    if not st.session_state["shots"]:
        st.info("Paste your script below to convert it into shots.")


    with st.expander("📝 Script Editor", expanded=not bool(st.session_state["shots"])):
        script_text = st.text_area("Input Script", height=150, placeholder="INT. KITCHEN - DAY\nBABARU eats a banana...")
        
        if st.button("Chunk Script"):
            if script_text:
                with st.spinner("Processing..."):
                    try:
                        model = genai.GenerativeModel(ANALYSIS_MODEL, generation_config={"response_mime_type": "application/json"})
                        sys_p = "Convert to JSON list of shots (id, action). If 'Donald/Trump', rename to 'The President' (caricature, heavy suit)."
                        res = model.generate_content(f"{sys_p}\nSCRIPT:\n{script_text}")
                        st.session_state['shots'] = json.loads(res.text)
                        save_project()
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # GALLERY
    if st.session_state['shots']:
        st.subheader(f"Shot List ({len(st.session_state['shots'])})")
        
        # Async Batching
        with st.expander("⚡ Batch Operations (Async)", expanded=False):
            st.caption("Submit all shots to Google's Batch API (Time-insensitive).")
            b1, b2, b3 = st.columns([1,1,2])
            
            with b1:
                if st.button("✨ Batch Sketch All", use_container_width=True):
                    with st.spinner("Submitting Sketch Job..."):
                        job, err = submit_batch_job("sketch", api_key)
                        if job:
                            st.session_state["batch_jobs"].append({
                                "id": job.name,
                                "type": "sketch",
                                "status": "PENDING",
                                "timestamp": time.time()
                            })
                            save_project()
                            st.success(f"Job Sent! ID: {job.name.split('/')[-1]}")
                        else:
                            st.error(f"Failed: {err}")

            with b2:
                if st.button("🎬 Batch Render All", use_container_width=True):
                    with st.spinner("Submitting Render Job..."):
                        job, err = submit_batch_job("final", api_key)
                        if job:
                            st.session_state["batch_jobs"].append({
                                "id": job.name,
                                "type": "final",
                                "status": "PENDING",
                                "timestamp": time.time()
                            })
                            save_project()
                            st.success(f"Job Sent! ID: {job.name.split('/')[-1]}")
                        else:
                            st.error(f"Failed: {err}")
            
            with b3:
                if st.session_state["batch_jobs"]:
                    st.write(f"**Active Jobs: {len(st.session_state['batch_jobs'])}**")
                    if st.button("🔄 Check Status"):
                         with st.spinner("Checking cloud..."):
                             done = check_batch_updates(api_key)
                             if done > 0: st.rerun()
                             else: st.info("No jobs completed yet.")
                    
                    for j in st.session_state["batch_jobs"]:
                        st.text(f"{j['type'].upper()}: {j['status']}")
                else:
                    st.info("No active batch jobs.")
        
        st.markdown("---")


        for i, shot in enumerate(st.session_state["shots"]):
            with st.container():
                st.markdown(f"#### {shot.get('id', i+1)}")
                c1, c2, c3 = st.columns([2, 1, 3])
                
                with c1:
                    # Editable Action Prompt
                    def update_action():
                        st.session_state['shots'][i]['action'] = st.session_state[f"action_{i}"]
                        save_project()
                        
                    action_val = st.text_area("Action", value=shot.get('action'), key=f"action_{i}", height=100, on_change=update_action)
                    
                    # Cast Selector per shot
                    active_cast = list(st.session_state['roster'].keys())
                    selected = st.multiselect("Cast", active_cast, default=active_cast, key=f"cast_{i}", label_visibility="collapsed")

                with c2:
                    # 🖊️ DRAFT
                    if st.button("Draft", key=f"draft_{i}", help="Generate rough sketch"):
                        with st.spinner("Sketching..."):
                            try:
                                # Style Configuration
                                s_style = st.session_state.get('sketch_style_dna', "")
                                if not s_style: s_style = "Rough blue pencil sketch, loose, dynamic, 2D animation layout."
                                
                                inputs = [f"TASK: SKETCH. STYLE: {s_style}. \nCHARACTERS:"]
                                if selected:
                                    for n in selected:
                                        inputs.append(st.session_state['roster'][n]['image'])
                                        inputs.append(f"ID: {n} (Use as reference).")
                                else:
                                    inputs.append("Generic characters fitting the scene.")
                                
                                inputs.append(f"\nSCENE: {action_val}. \nCONSTRAINT: Follow the style: {s_style}. No shading. Just lines.")

                                m = genai.GenerativeModel(DRAFT_MODEL, safety_settings=SAFETY_SETTINGS, generation_config={"temperature": 0.5})
                                r = m.generate_content(inputs)
                                
                                img = None
                                if r.parts:
                                    for p in r.parts:
                                        if p.inline_data: img = Image.open(io.BytesIO(p.inline_data.data))
                                
                                if img: 
                                    if i not in st.session_state['generated_images']: st.session_state['generated_images'][i] = {}
                                    elif not isinstance(st.session_state['generated_images'][i], dict):
                                        st.session_state['generated_images'][i] = {'final': st.session_state['generated_images'][i]}
                                        
                                    st.session_state['generated_images'][i]['draft'] = img
                                    save_project()
                                    st.rerun()
                                else: st.warning("No sketch.")
                            except Exception as e:
                                st.error(str(e))

                    # 🎬 FINAL
                    if st.button("Final", key=f"final_{i}", help="Generate final render"):
                        with st.spinner("Rendering..."):
                            try:
                                # Style Configuration
                                style_ref = st.session_state.get('final_style_dna', "")
                                if not style_ref:
                                    style_ref = "3D Pixar Animation Style. Cute, expressive, volumetric lighting, high fidelity CGI."
                                    
                                inputs = [f"TASK: FINAL RENDER. STYLE: {style_ref}.\n"]
                                
                                has_draft = i in st.session_state['generated_images'] and isinstance(st.session_state['generated_images'][i], dict) and 'draft' in st.session_state['generated_images'][i]
                                if has_draft:
                                    inputs.append(st.session_state['generated_images'][i]['draft'])
                                    inputs.append("INSTRUCTION: turn this detailed sketch into a high-fidelity render matching the STYLE. Keep composition exactly.")
                                
                                if selected:
                                    inputs.append("CHARACTERS (Texture/Identity Ref):")
                                    for n in selected:
                                        inputs.append(st.session_state['roster'][n]['image'])
                                        inputs.append(f"ID: {n}.")
                                
                                scene = f"\nSCENE: {action_val}. \nLIGHTING: Cinematic volumetric."
                                inputs.append(scene)
                                
                                m = genai.GenerativeModel(FINAL_MODEL, safety_settings=SAFETY_SETTINGS, generation_config={"temperature": 0.3})
                                r = m.generate_content(inputs)
                                    
                                img = None
                                if r.parts:
                                    for p in r.parts:
                                        if p.inline_data: img = Image.open(io.BytesIO(p.inline_data.data))
                                
                                if img: 
                                    if i not in st.session_state['generated_images']: st.session_state['generated_images'][i] = {}
                                    elif not isinstance(st.session_state['generated_images'][i], dict):
                                        st.session_state['generated_images'][i] = {'final': st.session_state['generated_images'][i]}
                                        
                                    st.session_state['generated_images'][i]['final'] = img
                                    save_project()
                                    st.rerun()
                                else: st.warning("No render.")
                            except Exception as e:
                                st.error(str(e))

                    # 🎥 ACTION (VIDEO)
                    if st.button("Action!", key=f"vid_{i}", help="Generate video (Coming Soon)"):
                         st.toast("🎬 The AI director is in their trailer. Video generation coming soon!", icon="⛔")
                         st.info("Video generation is currently on coffee break. Check back later!")

                with c3:
                    # Multi-View with Downloads
                    current_data = st.session_state['generated_images'].get(i, {})
                    if not isinstance(current_data, dict):
                        current_data = {'final': current_data}
                        st.session_state['generated_images'][i] = current_data
                    
                    tabs_display = st.tabs(["Draft", "Final"])
                    
                    with tabs_display[0]:
                        if 'draft' in current_data:
                            st.image(current_data['draft'], use_container_width=True)
                            buf = io.BytesIO()
                            current_data['draft'].save(buf, format="PNG")
                            st.download_button("⬇️ Draft", data=buf.getvalue(), file_name=f"s{i+1}_draft.png", mime="image/png", key=f"dl_d_{i}")
                    
                    with tabs_display[1]:
                        if 'final' in current_data:
                            st.image(current_data['final'], use_container_width=True)
                            buf = io.BytesIO()
                            current_data['final'].save(buf, format="PNG")
                            st.download_button("⬇️ Final", data=buf.getvalue(), file_name=f"s{i+1}_final.png", mime="image/png", key=f"dl_f_{i}")
                            
                st.divider()

# Free Render Tab
with tab_free:
    st.header("🎨 Free Render Mode")
    st.caption("Upload any sketch (or revised draft) and render it in your Locked Style.")
    
    col_input, col_output = st.columns(2)
    
    with col_input:
        uploaded_sketch = st.file_uploader("Upload Sketch/Layout", type=["jpg", "png", "jpeg"], key="free_up")
        
        # Cast Selection
        active_cast = list(st.session_state['roster'].keys())
        selected_free = st.multiselect("Active Cast (for Texture/Identity)", active_cast, default=active_cast, key="free_cast")
        
        # Prompt
        free_prompt = st.text_area("Scene Description", placeholder="Describe the action/lighting...", height=100)
        
        if st.button("Render", type="primary"):
            if not uploaded_sketch:
                st.warning("Please upload a sketch first.")
            else:
                with st.spinner("Rendering..."):
                    try:
                        sketch_img = Image.open(uploaded_sketch).convert("RGB")
                        
                        # Style Configuration
                        style_ref = st.session_state.get('final_style_dna', "")
                        if not style_ref:
                            style_ref = "3D Pixar Animation Style. Cute, expressive, volumetric lighting, high fidelity CGI."

                        inputs = [f"TASK: FINAL RENDER FROM SKETCH. STYLE: {style_ref}\n"]
                        inputs.append(sketch_img)
                        inputs.append(f"INSTRUCTION: This is the visual layout. Render this composition in the specified STYLE: {style_ref}.")
                        
                        if selected_free:
                            inputs.append("CHARACTERS (Texture/Identity Ref):")
                            for n in selected_free:
                                inputs.append(st.session_state['roster'][n]['image'])
                                inputs.append(f"ID: {n}.")
                        
                        if free_prompt:
                            inputs.append(f"SCENE DESCRIPTION: {free_prompt}")
                        
                        m = genai.GenerativeModel(FINAL_MODEL, safety_settings=SAFETY_SETTINGS, generation_config={"temperature": 0.3})
                        r = m.generate_content(inputs)
                        
                        img = None
                        if r.parts:
                            for p in r.parts:
                                if p.inline_data: img = Image.open(io.BytesIO(p.inline_data.data))
                        
                        if img:
                            st.session_state['free_render'] = img
                            save_project()
                        else:
                            st.warning("No render produced.")
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                        
    with col_output:
        if st.session_state.get('free_render'):
            st.image(st.session_state['free_render'], caption="Free Render Result", use_container_width=True)
            
            # Download
            buf = io.BytesIO()
            st.session_state['free_render'].save(buf, format="PNG")
            st.download_button(label="⬇️ Download Result", data=buf.getvalue(), file_name="free_render.png", mime="image/png", key="dl_free")
        else:
            st.info("Render result will appear here.")

# =========================================================
# TAB 3: FREE VIDEO STUDIO (VEO)
# =========================================================
with tab_video:
    st.header("🎥 Free Video Studio")
    
    st.info("🚧 UNDER CONSTRUCTION 🚧")
    st.markdown("""
    ### 🎬 Where's the camera?
    
    Our AI video director is currently **refusing to work**. 
    
    *   **"My artistic vision cannot be rushed!"** — *The AI Model*
    *   **"I am not a content farm!"** — *Also The AI Model*
    
    We are currently negotiating with the GPUs. Please check back in a future update when everyone has had their coffee.
    """)
    
    st.image("https://media.giphy.com/media/l0HlSi3AIOM3fAhX2/giphy.gif", caption="Live footage of our dev team fixing this.", width=400)
