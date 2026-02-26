import os
import warnings
import re
from datetime import datetime
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
# 1. API KEY & DATABASE
# ==========================================
API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY.strip()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SB_URL, SB_KEY)

llm_gemini = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash", 
    temperature=0.2, 
    max_output_tokens=4000 
)
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. SISTEM DATA & FUNGSI FOTO (RESTORED)
# ==========================================
def init_state():
    defaults = {
        'audit_stage': 'input', 'q_index': 0, 'chat_history': [],
        'initial_tasks': "", 'initial_evidence': "", 'data_saved': False, 'current_user': None
    }
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val

init_state()

def process_images(files):
    """Fungsi untuk membaca data dari foto/screenshot."""
    if not files: return ""
    descriptions = []
    for f in files:
        img = Image.open(f)
        res = vision_model.generate_content(["Ekstrak fakta teknis dan data objektif dari gambar ini.", img])
        descriptions.append(res.text)
    return " | ".join(descriptions)

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
# 3. AGENT SETUP (DEEP ANALYTICS)
# ==========================================
consultant = Agent(
    role='Lead Strategic Copilot',
    goal='Mendiagnosa hambatan sistemik melalui analisa mendalam dan bukti visual.',
    backstory="""Kamu Auditor senior yang dingin namun luwes. Gunakan 'Saya' dan 'Kamu'.
    WAJIB memberikan analisa mendalam (minimal 2-3 paragraf) sebelum bertanya.
    Gunakan fakta dari foto (jika ada) untuk memperkuat diagnosamu.""",
    llm=llm_gemini
)

architect = Agent(
    role='Solutions Architect',
    goal='Memberikan blueprint solusi strategis.',
    backstory="""Berikan laporan: SKOR_FINAL: [0-10], ### DIAGNOSA_AWAL, ### ACTION_ITEMS (Format **Nama Tugas**: Deskripsi), ### CONTINUITY_PROTOCOL.""",
    llm=llm_gemini
)

# ==========================================
# 4. LOGIN TENGAH (LOCKED)
# ==========================================
st.set_page_config(page_title="Strategic Copilot V9.7", layout="wide")

if st.session_state.current_user is None:
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("<h1 style='text-align: center;'>🔐 Secure Access</h1>", unsafe_allow_html=True)
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
                    st.success("Terdaftar! Klik masuk lagi."); st.rerun()
    st.stop()

# --- SIDEBAR: CHECKLIST RAPI (EXPANDER) ---
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
st.sidebar.subheader("📋 Action Items")

res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
if res_tasks.data:
    for t in res_tasks.data:
        title = t['task_name'].split("|")[0].strip()
        desc = t['task_name'].split("|")[1].strip() if "|" in t['task_name'] else "Segera eksekusi."
        with st.sidebar.expander(f"📌 {title}"):
            st.write(desc)
            if st.button("Selesaikan", key=f"btn_{t['id']}", use_container_width=True):
                supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()

if st.sidebar.button("Keluar"): st.session_state.current_user = None; st.rerun()

# ==========================================
# 5. MAIN CONTENT (AUDIT WITH PHOTO)
# ==========================================
tab_audit, tab_dash = st.tabs(["🔍 Sesi Audit", "📊 Dashboard"])

with tab_audit:
    st.error("### ⚠️ INSTRUKSI: Masukkan masalah & lampirkan bukti foto untuk analisa mendalam.")
    
    if st.session_state.audit_stage == 'input':
        u_in = st.text_area("Apa tantangan strategis/operasionalmu?", height=150)
        u_f = st.file_uploader("Upload Bukti Visual (Screenshot/Data)", accept_multiple_files=True)
        if st.button("Mulai Audit"):
            if len(u_in) > 10:
                st.session_state.initial_evidence = process_images(u_f) if u_f else ""
                st.session_state.initial_tasks = u_in
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"Fase Interogasi {st.session_state.q_index}/4")
        hist_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        task_q = Task(
            description=f"Konteks: {st.session_state.initial_tasks}. Bukti Foto: {st.session_state.initial_evidence}. History: {hist_str}. Tahap: {st.session_state.q_index}/4.",
            agent=consultant, expected_output="Analisa mendalam (2-3 paragraf) dan pertanyaan baru."
        )
        with st.spinner("Menganalisa..."):
            current_q = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(current_q)
        u_ans = st.text_area("Jawaban kamu:", key=f"ans_{st.session_state.q_index}")
        u_img = st.file_uploader("Tambah Bukti Foto Baru", accept_multiple_files=True, key=f"img_{st.session_state.q_index}")
        
        if st.button("Kirim Analisa"):
            img_info = f" [Bukti Baru: {process_images(u_img)}]" if u_img else ""
            st.session_state.chat_history.append({"q": current_q, "a": u_ans + img_info})
            if st.session_state.q_index < 4: st.session_state.q_index += 1
            else: st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_state.audit_stage == 'report':
        with st.spinner("Menyusun Laporan..."):
            full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
            task_fin = Task(description=f"History: {full_hist}.", agent=architect, expected_output="Laporan blueprint solusi.")
            res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw); st.markdown(res)
            
            score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
            f_score = float(score_match.group(1)) if score_match else 0.0
            
            if not st.session_state.data_saved:
                supabase.table("audit_log").insert({"user_id": user_nickname, "score": f_score, "audit_report": res, "input_preview": st.session_state.initial_tasks[:100]}).execute()
                # Task Extraction
                action_section = re.search(r"### ACTION_ITEMS\s*(.*?)(?:\n###|$)", res, re.DOTALL | re.IGNORECASE)
                if action_section:
                    tasks = re.findall(r"\*\*(.+?)\*\*[:\-]\s*(.+)", action_section.group(1))
                    for title, desc in tasks:
                        supabase.table("pending_tasks").insert({"user_id": user_nickname, "task_name": f"{title.strip()} | {desc.strip()}", "status": "Pending"}).execute()
                st.session_state.data_saved = True; st.rerun()

            st.download_button("📥 Download PDF", data=generate_pdf(user_nickname, res, f_score), file_name=f"Audit_{user_nickname}.pdf")
            if st.button("Reset Sesi"):
                st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False; st.rerun()

with tab_dash:
    st.title("📊 Dashboard Performa")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data)
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
        st.dataframe(df.sort_values(by='created_at', ascending=False), use_container_width=True)