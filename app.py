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
# 1. API KEY & LLM & DATABASE
# ==========================================
API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY.strip()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SB_URL, SB_KEY)

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. FUNGSI UTILITAS
# ==========================================
def init_state():
    defaults = {'audit_stage': 'input', 'q_index': 0, 'chat_history': [], 'initial_tasks': "", 'data_saved': False, 'current_user': None}
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val

init_state()

def process_images(files):
    if not files: return ""
    descriptions = []
    for f in files:
        img = Image.open(f)
        res = vision_model.generate_content(["Ekstrak fakta teknis/data objektif dari gambar ini.", img])
        descriptions.append(res.text)
    return " | ".join(descriptions)

def clean_txt(text):
    return text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").replace("*", "-").encode('ascii', 'ignore').decode('ascii')

def generate_pdf(nickname, report_text, score, tasks):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, txt=f"REPORT: {clean_txt(nickname)}", ln=True, align='C')
    pdf.set_font("Arial", size=10); pdf.cell(0, 10, txt=f"Date: {datetime.now().strftime('%d/%b/%Y')}", ln=True, align='C'); pdf.ln(10)
    pdf.set_fill_color(230, 230, 230); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt=f"SCORE: {score}/10", ln=True, fill=True); pdf.ln(5)
    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 5, txt=clean_txt(report_text))
    if tasks:
        pdf.ln(10); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt="ACTION ITEMS:", ln=True)
        for i, t in enumerate(tasks, 1): pdf.multi_cell(0, 5, txt=f"{i}. [ ] {clean_txt(t['task_name'].split('|')[0])}")
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. AGENT SETUP
# ==========================================
consultant = Agent(
    role='Lead Strategic Auditor',
    goal='Mendiagnosa bottleneck operasional melalui interogasi progresif.',
    backstory="""Kamu auditor dingin. Gunakan 'saya' dan 'kamu'. JANGAN gunakan kata 'Anda'.
    TUGAS: Ajukan 1 pertanyaan baru berdasarkan history. Jangan pernah memberikan kesimpulan atau skor di tahap ini.""",
    llm=llm_gemini, allow_delegation=False
)

architect = Agent(
    role='Solutions Architect',
    goal='Memberikan blueprint solusi strategis.',
    backstory="""Kamu arsitek kaku. Wajib format: SKOR_FINAL: [0-10], ### DIAGNOSA_AWAL, ### ACTION_ITEMS, ### CONTINUITY_PROTOCOL.""",
    llm=llm_gemini, allow_delegation=False
)

# ==========================================
# 4. APLIKASI UTAMA
# ==========================================
st.set_page_config(page_title="Strategic Auditor V14", layout="wide")

if st.session_state.current_user is None:
    u_name = st.sidebar.text_input("Nickname:")
    u_pass = st.sidebar.text_input("Password:", type="password")
    if st.sidebar.button("Login"):
        res = supabase.table("user_access").select("*").eq("username", u_name).execute()
        if res.data and res.data[0]['password'] == u_pass: st.session_state.current_user = u_name; st.rerun()
    st.title("🔐 Silakan Login di Sidebar"); st.stop()

# SIDEBAR CHECKLIST
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
res_t = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").execute()
st.sidebar.markdown("### 📋 Tasks")
for t in res_t.data:
    if st.sidebar.button(f"✅ {t['task_name'].split('|')[0]}", key=f"t_{t['id']}"):
        supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()

# --- ALUR AUDIT ---
if page == "Audit & Konsultasi":
    st.title("Strategic Auditor V14")
    
    if st.session_state.audit_stage == 'input':
        st.warning("Panduan: Input tantangan -> Jawab 4 pertanyaan (sertakan bukti foto) -> Ambil PDF.")
        u_in = st.text_area("Apa tantangan strategis atau operasionalmu hari ini?")
        u_f = st.file_uploader("Upload PDF Audit Sebelumnya / Bukti", accept_multiple_files=True)
        if st.button("Mulai Audit"):
            if len(u_in) > 5:
                st.session_state.initial_tasks = u_in + " [Data Gambar: " + process_images(u_f) + "]"
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"🔍 Tahap Interogasi ({st.session_state.q_index}/4)")
        hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        # LOGIKA: Pertanyaan 1-4 murni pertanyaan
        task_q = Task(
            description=f"History: {hist}. Awal: {st.session_state.initial_tasks}. Tahap: {st.session_state.q_index}/4. Berikan analisa singkat 2 kalimat dan 1 pertanyaan baru. Gunakan 'kamu'. DILARANG memberikan skor atau diagnosa final sekarang.",
            agent=consultant, expected_output="Pertanyaan investigatif baru."
        )
        with st.spinner("Auditor sedang berpikir..."):
            q_text = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(q_text)
        u_ans = st.text_area("Jawaban kamu:", key=f"ans_{st.session_state.q_index}")
        u_img = st.file_uploader("📷 Lampirkan Bukti Foto/Screenshot", accept_multiple_files=True, key=f"img_{st.session_state.q_index}")
        
        if st.button("Kirim Jawaban"):
            if len(u_ans) > 2:
                img_info = process_images(u_img)
                st.session_state.chat_history.append({"q": q_text, "a": u_ans + f" [Visual: {img_info}]"})
                if st.session_state.q_index < 4: 
                    st.session_state.q_index += 1
                else: 
                    st.session_state.audit_stage = 'report' # Pindah ke stage report HANYA setelah jawaban ke-4
                st.rerun()

    elif st.session_state.audit_stage == 'report':
        st.subheader("📊 Hasil Audit Strategis")
        with st.spinner("Menyusun Laporan & Link Download..."):
            full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
            task_f = Task(description=f"Audit History Lengkap: {full_hist}.", agent=architect, expected_output="Laporan kaku dengan SKOR_FINAL.")
            res = str(Crew(agents=[architect], tasks=[task_f]).kickoff().raw)
            st.markdown(res)
            
            # Parsing Skor
            score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
            f_score = score_match.group(1) if score_match else "0.0"
            
            if not st.session_state.data_saved:
                # Simpan ke DB
                supabase.table("audit_log").insert({"user_id": user_nickname, "score": float(f_score), "audit_report": res}).execute()
                # Simpan Tasks ke Sidebar
                for line in res.split("\n"):
                    if "**" in line and ":" in line:
                        supabase.table("pending_tasks").insert({"user_id": user_nickname, "task_name": line.replace("*", "").strip(), "status": "Pending"}).execute()
                st.session_state.data_saved = True
                st.rerun() 
            
            # Tombol Download PDF yang ASLI
            p_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").execute().data
            pdf_b = generate_pdf(user_nickname, res, f_score, p_tasks)
            st.download_button("📥 UNDUH LAPORAN PDF SEKARANG", data=pdf_b, file_name=f"Audit_{user_nickname}.pdf", use_container_width=True)
            
            if st.button("Selesai & Reset Sesi"):
                st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved, st.session_state.q_index = 'input', [], False, 0
                st.rerun()

elif page == "Dashboard":
    st.title("📊 Performa")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data); df['created_at'] = pd.to_datetime(df['created_at'])
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
    else: st.info("Belum ada data.")