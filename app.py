import os
import warnings
import logging

# ==========================================
# 0. KONFIGURASI SISTEM
# ==========================================
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OUT_OUT"] = "true"
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import plotly.express as px
import re
from datetime import datetime
from PIL import Image
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from supabase import create_client, Client 

# ==========================================
# 1. API KEY & LLM & DATABASE
# ==========================================
# Mengambil dari Secrets Streamlit Cloud
API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY.strip()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Inisialisasi Supabase
SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SB_URL, SB_KEY)

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.1, 
    max_output_tokens=4000 
)
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. SISTEM DATA & NORMALISASI
# ==========================================
def init_state():
    defaults = {
        'audit_stage': 'input',
        'q_index': 0,
        'chat_history': [],
        'initial_tasks': "",
        'initial_evidence': "",
        'data_saved': False
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

def process_images(files):
    descriptions = []
    for f in files:
        img = Image.open(f)
        response = vision_model.generate_content(["Identifikasi fakta teknis dan data objektif.", img])
        descriptions.append(response.text)
    return " | ".join(descriptions)

def save_audit_to_db(user_input, audit_result, nickname):
    if st.session_state.data_saved: return 
    
    # Tanpa try-except agar error Supabase terlihat jelas
    score_match = re.search(r"SKOR_FINAL\s*:\s*(?:\[)?([\d.]+)(?:\])?", audit_result)
    raw_score = float(score_match.group(1)) if score_match else 0.0
    score = raw_score / 10 if raw_score > 10 else raw_score
    
    supabase.table("audit_log").insert({
        "user_id": nickname,
        "score": score,
        "audit_report": audit_result,
        "input_preview": user_input[:100]
    }).execute()
    
    st.session_state.data_saved = True

def extract_and_save_tasks(audit_result, nickname):
    """V8.9: Ekstraksi Tanpa Filter Error untuk Debugging."""
    match = re.search(r"### ACTION_ITEMS\s*(.*?)(?:\n###|$)", audit_result, re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1)
        lines = re.findall(r"[-*]\s*(.+)", content)
        for line in lines:
            clean_task = re.split(r"[:]", line)[0].strip()
            clean_task = clean_task.replace("*", "").replace("_", "").replace("#", "").strip()
            clean_task = re.sub(r"^(Kamu harus |Anda perlu |Pastikan |Lakukan |Segera |Harap |Silakan )", "", clean_task, flags=re.IGNORECASE)
            clean_task = clean_task[0].upper() + clean_task[1:] if len(clean_task) > 0 else clean_task
            
            if 5 < len(clean_task) < 120:
                supabase.table("pending_tasks").insert({
                    "user_id": nickname,
                    "task_name": clean_task,
                    "status": "Pending"
                }).execute()

# ==========================================
# 3. AGENT SETUP
# ==========================================
consultant = Agent(
    role='Expert Strategy Consultant',
    goal='Memberikan observasi teknis dan meminta data pendukung untuk solusi.',
    backstory="""Kamu konsultan senior. Analisa celah mereka dan minta foto/data spesifik agar kamu bisa menghitung solusi. Gunakan 'kamu'.""",
    llm=llm_gemini,
    allow_delegation=False
)

architect = Agent(
    role='Solution Architect',
    goal='Memberikan blueprint solusi teknis dan langkah aksi nyata.',
    backstory="""Gunakan SKOR_FINAL: [0.0 - 10.0]. Wajib ada section '### ACTION_ITEMS' berisi daftar solusi perintah singkat.""",
    llm=llm_gemini,
    allow_delegation=False
)

# ==========================================
# 4. TAMPILAN WEB
# ==========================================
st.set_page_config(page_title="Strategic Auditor V8.9", layout="wide")
user_nickname = st.sidebar.text_input("Identitas:", value="Guest").strip()
page = st.sidebar.radio("Navigasi:", ["Audit & Konsultasi", "Dashboard"])

# Sidebar Checklist (Tanpa try-except)
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Checklist: {user_nickname}")

res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").execute()
pending = res_tasks.data

if pending:
    for task in pending:
        if st.sidebar.checkbox(task['task_name'], key=f"db_{task['id']}"):
            supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", task['id']).execute()
            st.rerun()
else:
    st.sidebar.success("Tidak ada tugas pending. 🚀")

if page == "Audit & Konsultasi":
    st.title(f"Strategic Consultant AI: {user_nickname}")
    st.markdown("---")

    if st.session_state.audit_stage == 'input':
        st.warning("### **🛠️ Panduan Operasional Konsultasi**")
        u_in = st.text_area("Apa tantangan teknis atau rencana yang ingin kamu audit?", height=120)
        u_files = st.file_uploader("Upload Bukti Visual/Data (Opsional)", accept_multiple_files=True)
        
        if st.button("Mulai Analisis"):
            if len(u_in) > 15:
                with st.spinner("Menganalisa data awal..."):
                    st.session_state.initial_evidence = process_images(u_files) if u_files else ""
                    st.session_state.initial_tasks = u_in
                    st.session_state.audit_stage = 'interrogation'
                    st.session_state.q_index = 1
                    st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"🔍 Analisis Consultant ({st.session_state.q_index}/3)")
        with st.spinner("Konsultan sedang menyusun observasi..."):
            hist_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
            task_q = Task(
                description=f"Masalah: {st.session_state.initial_tasks}. History: {hist_str}.",
                agent=consultant,
                expected_output="Satu permintaan data spesifik."
            )
            current_q = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(current_q)
        u_ans = st.text_area("Input Jawaban:", key=f"ans_{st.session_state.q_index}")
        u_img = st.file_uploader("Lampirkan Bukti", accept_multiple_files=True, key=f"img_{st.session_state.q_index}")
        
        if st.button("Kirim Data"):
            img_data = f" [Foto: {process_images(u_img)}]" if u_img else ""
            st.session_state.chat_history.append({"q": current_q, "a": u_ans + img_data})
            if st.session_state.q_index < 3:
                st.session_state.q_index += 1
            else:
                st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_state.audit_stage == 'report':
        with st.spinner("Membangun Blueprint Solusi..."):
            full_hist = f"Input: {st.session_state.initial_tasks}\n" + "\n".join([f"Q{i+1}: {h['q']}\nA: {h['a']}" for i, h in enumerate(st.session_state.chat_history)])
            task_fin = Task(
                description=f"Wajib SKOR_FINAL: [angka] dan section '### ACTION_ITEMS'. History: {full_hist}",
                agent=architect,
                expected_output="Laporan blueprint solusi strategis."
            )
            res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw)
            st.markdown(res)
            
            save_audit_to_db(st.session_state.initial_tasks, res, user_nickname)
            extract_and_save_tasks(res, user_nickname)
            
            if st.button("Selesai & Reset"):
                st.session_state.audit_stage = 'input'
                st.session_state.chat_history = []
                st.session_state.data_saved = False
                st.rerun()

elif page == "Dashboard":
    st.title(f"📊 Intelligence Tracking: {user_nickname}")
    # Tanpa try-except untuk melihat error grafik
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", descending=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
        st.dataframe(df.sort_values(by='created_at', ascending=False))
    else:
        st.info("Belum ada data audit untuk identitas ini.")