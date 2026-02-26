import os
import warnings
import re
from datetime import datetime, timedelta
from PIL import Image

# ==========================================
# 0. KONFIGURASI SISTEM
# ==========================================
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OUT_OUT"] = "true"
warnings.filterwarnings("ignore", category=FutureWarning)

import streamlit as st
import pandas as pd
import plotly.express as px
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from supabase import create_client, Client 
from fpdf import FPDF

# ==========================================
# 1. API & DATABASE INITIALIZATION
# ==========================================
API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY.strip()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SB_URL, SB_KEY)

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)

# ==========================================
# 2. CORE FUNCTIONS
# ==========================================
def init_state():
    if 'audit_stage' not in st.session_state: st.session_state.audit_stage = 'input'
    if 'q_index' not in st.session_state: st.session_state.q_index = 0
    if 'chat_history' not in st.session_state: st.session_state.chat_history = []
    if 'current_user' not in st.session_state: st.session_state.current_user = None
    if 'data_saved' not in st.session_state: st.session_state.data_saved = False

init_state()

def clean_txt(text):
    return text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").replace("*", "-").encode('ascii', 'ignore').decode('ascii')

def generate_pdf(nickname, report_text, score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, txt=f"STRATEGIC AUDIT: {clean_txt(nickname)}", ln=True, align='C')
    pdf.set_font("Arial", size=10); pdf.cell(0, 10, txt=f"Date: {datetime.now().strftime('%d %b %Y')}", ln=True, align='C'); pdf.ln(10)
    pdf.set_fill_color(230, 230, 230); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt=f"SCORE: {score}/10", ln=True, fill=True); pdf.ln(5)
    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 5, txt=clean_txt(report_text))
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. AGENT SETUP (ORIGINAL 7.9 TONE)
# ==========================================
consultant = Agent(
    role='Strategic Consultant',
    goal='Mendiagnosa hambatan operasional melalui interogasi tajam.',
    backstory="""Kamu adalah auditor yang dingin dan objektif. Gunakan 'saya' dan 'kamu'. 
    Tugasmu adalah bertanya secara progresif (4 pertanyaan) untuk membongkar kelemahan user.""",
    llm=llm_gemini, allow_delegation=False
)

architect = Agent(
    role='Solutions Architect',
    goal='Menyusun blueprint solusi dan skor final.',
    backstory="""Kamu ahli strategi. Wajib memberikan format: SKOR_FINAL: [0-10], ### DIAGNOSA_AWAL, ### ACTION_ITEMS, ### CONTINUITY_PROTOCOL.""",
    llm=llm_gemini, allow_delegation=False
)

# ==========================================
# 4. LOGIN GATE (TENDAH LAYAR - LOCKED)
# ==========================================
st.set_page_config(page_title="Strategic Auditor 7.9", layout="wide")

if st.session_state.current_user is None:
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("<h1 style='text-align: center;'>🔐 Access Control</h1>", unsafe_allow_html=True)
        u_name = st.text_input("Nickname:")
        u_pass = st.text_input("Password:", type="password")
        if st.button("Masuk / Daftar", use_container_width=True):
            if u_name and u_pass:
                res = supabase.table("user_access").select("*").eq("username", u_name).execute()
                if res.data:
                    if res.data[0]['password'] == u_pass:
                        st.session_state.current_user = u_name; st.rerun()
                    else: st.error("Password salah.")
                else:
                    supabase.table("user_access").insert({"username": u_name, "password": u_pass}).execute()
                    st.success("Akun baru terdaftar!"); st.rerun()
    st.stop()

# --- SIDEBAR (CHECKLIST) ---
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
st.sidebar.subheader("📋 Pending Checklist")

res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").execute()
for t in res_tasks.data:
    if st.sidebar.button(f"✅ {t['task_name'][:30]}...", key=f"t_{t['id']}", use_container_width=True):
        supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()

if st.sidebar.button("Keluar"): st.session_state.current_user = None; st.rerun()

# ==========================================
# 5. MAIN CONTENT (4 QUESTIONS)
# ==========================================
tabs = st.tabs(["🔍 Audit", "📊 Dashboard"])

with tabs[0]:
    if st.session_state.audit_stage == 'input':
        u_in = st.text_area("Apa masalah teknis/strategis yang ingin diaudit?", height=150)
        if st.button("Mulai Audit"):
            if len(u_in) > 5:
                st.session_state.initial_tasks = u_in
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"Pertanyaan {st.session_state.q_index} dari 4")
        hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        task_q = Task(
            description=f"History: {hist}. Masalah: {st.session_state.initial_tasks}. Berikan satu pertanyaan investigasi baru. Gunakan 'kamu'.",
            agent=consultant, expected_output="Satu pertanyaan tajam."
        )
        with st.spinner("Berpikir..."):
            q_text = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(q_text)
        u_ans = st.text_area("Jawaban:", key=f"ans_{st.session_state.q_index}")
        if st.button("Kirim"):
            st.session_state.chat_history.append({"q": q_text, "a": u_ans})
            if st.session_state.q_index < 4: st.session_state.q_index += 1
            else: st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_stage == 'report' or st.session_state.audit_stage == 'report':
        st.subheader("🏁 Laporan Akhir")
        with st.spinner("Menyusun laporan..."):
            full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
            task_f = Task(description=f"History: {full_hist}.", agent=architect, expected_output="Laporan lengkap dengan SKOR_FINAL.")
            res = str(Crew(agents=[architect], tasks=[task_f]).kickoff().raw)
        
        st.markdown(res)
        
        # Save Logic
        score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
        f_score = float(score_match.group(1)) if score_match else 0.0
        
        if not st.session_state.data_saved:
            supabase.table("audit_log").insert({"user_id": user_nickname, "score": f_score, "audit_report": res}).execute()
            # Simple Task Extraction
            lines = res.split("\n")
            for line in lines:
                if "**" in line:
                    supabase.table("pending_tasks").insert({"user_id": user_nickname, "task_name": line.replace("*", "").strip(), "status": "Pending"}).execute()
            st.session_state.data_saved = True; st.rerun()

        pdf_data = generate_pdf(user_nickname, res, f_score)
        st.download_button("📥 Download PDF", data=pdf_data, file_name=f"Audit_{user_nickname}.pdf")
        
        if st.button("Reset"):
            st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False; st.rerun()

# ==========================================
# 6. DASHBOARD
# ==========================================
with tabs[1]:
    res_dash = supabase.table("audit_log").select("created_at, score").eq("user_id", user_nickname).order("created_at").execute()
    if res_dash.data:
        df = pd.DataFrame(res_dash.data)
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]))
    else: st.info("Belum ada data.")