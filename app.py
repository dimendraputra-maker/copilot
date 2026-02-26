import os
import warnings
import re
import io
from datetime import datetime, timedelta
from PIL import Image

# ==========================================
# 0. KONFIGURASI SISTEM & SUPPRESS WARNINGS
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
# 1. API KEY & LLM & DATABASE INITIALIZATION
# ==========================================
API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY.strip()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SB_URL, SB_KEY)

# LLM Konfigurasi - Suhu rendah untuk akurasi audit
llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.1)
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. FUNGSI LOGIKA DATA & PDF
# ==========================================
def init_state():
    keys = {
        'audit_stage': 'input', 
        'q_index': 0, 
        'chat_history': [], 
        'initial_tasks': "", 
        'data_saved': False, 
        'current_user': None
    }
    for key, val in keys.items():
        if key not in st.session_state: st.session_state[key] = val

init_state()

def clean_txt(text):
    """Pembersih teks untuk mencegah FPDF Latin-1 Error."""
    return text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").replace("*", "-").encode('ascii', 'ignore').decode('ascii')

def process_images(files):
    """Mengekstrak data dari gambar/screenshot menggunakan Vision AI."""
    if not files: return ""
    descriptions = []
    for f in files:
        img = Image.open(f)
        res = vision_model.generate_content(["Sebutkan fakta teknis dan angka objektif dari gambar ini.", img])
        descriptions.append(res.text)
    return " | ".join(descriptions)

def generate_pdf(nickname, report_text, score, tasks):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, txt=f"STRATEGIC AUDIT: {clean_txt(nickname)}", ln=True, align='C')
    pdf.set_font("Arial", size=10); pdf.cell(0, 10, txt=f"Date: {datetime.now().strftime('%d %b %Y')}", ln=True, align='C'); pdf.ln(10)
    pdf.set_fill_color(230, 230, 230); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt=f"PERFORMANCE SCORE: {score}/10", ln=True, fill=True); pdf.ln(5)
    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 5, txt=clean_txt(report_text))
    if tasks:
        pdf.ln(10); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt="PENDING ACTION ITEMS:", ln=True)
        for i, t in enumerate(tasks, 1):
            pdf.multi_cell(0, 5, txt=f"{i}. [ ] {clean_txt(t['task_name'].split('|')[0])}")
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. AGENT DEFINITION (UNIVERSAL AUDITOR)
# ==========================================
consultant = Agent(
    role='Lead Strategic Auditor',
    goal='Mendiagnosa bottleneck operasional melalui interogasi progresif.',
    backstory="""Kamu auditor senior yang dingin. Gunakan 'saya' dan 'kamu'. DILARANG KERAS kata 'Anda'.
    Tugasmu: Berikan analisa teknis tajam dan tanya 1 hal baru. Jangan menceramahi psikologi.""",
    llm=llm_gemini, allow_delegation=False
)

architect = Agent(
    role='Solutions Architect',
    goal='Memberikan blueprint solusi strategis dan skor performa.',
    backstory="""Kamu arsitek sistem kaku. Wajib format: SKOR_FINAL: [0-10], ### DIAGNOSA_AWAL, ### ACTION_ITEMS, ### CONTINUITY_PROTOCOL.""",
    llm=llm_gemini, allow_delegation=False
)

# ==========================================
# 4. UI FLOW & SIDEBAR
# ==========================================
st.set_page_config(page_title="Strategic Auditor V13.1", layout="wide")

# Login Logic
if st.session_state.current_user is None:
    with st.sidebar:
        st.title("🔐 Login")
        u_name = st.text_input("Nickname:")
        u_pass = st.text_input("Password:", type="password")
        if st.button("Masuk"):
            res = supabase.table("user_access").select("*").eq("username", u_name).execute()
            if res.data and res.data[0]['password'] == u_pass:
                st.session_state.current_user = u_name; st.rerun()
            else: st.error("Ditolak.")
    st.title("Strategic Auditor AI"); st.info("Silakan Login di Sidebar untuk memulai."); st.stop()

# --- SIDEBAR: CHECKLIST & STATS ---
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")

# Dashboard Stats di Sidebar
try:
    res_stats = supabase.table("audit_log").select("score").eq("user_id", user_nickname).execute()
    avg_score = sum([d['score'] for d in res_stats.data])/len(res_stats.data) if res_stats.data else 0
    st.sidebar.metric("Average Score", f"{avg_score:.1f}/10")
except: pass

st.sidebar.markdown("### 📋 Pending Tasks")
res_t = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").execute()
for t in res_t.data:
    if st.sidebar.button(f"Selesai: {t['task_name'].split('|')[0]}", key=f"t_{t['id']}", use_container_width=True):
        supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()

if st.sidebar.button("Keluar Systems", color="red"): st.session_state.current_user = None; st.rerun()

# --- MAIN NAVIGATION ---
menu = st.tabs(["🔍 Audit", "📊 Dashboard"])

# ==========================================
# 5. TAB: AUDIT & KONSULTASI
# ==========================================
with menu[0]:
    st.warning("Panduan: Input Masalah -> Jawab 4 Pertanyaan (Lampirkan Foto) -> Ambil PDF.")
    
    if st.session_state.audit_stage == 'input':
        u_in = st.text_area("Apa tantangan teknis atau strategismu hari ini?", height=150)
        u_f = st.file_uploader("Upload Bukti/PDF Audit Sebelumnya", accept_multiple_files=True)
        if st.button("Mulai Audit"):
            if len(u_in) > 5:
                img_data = process_images(u_f)
                st.session_state.initial_tasks = u_in + f" [Context Image: {img_data}]"
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"Interogasi Auditor ({st.session_state.q_index}/4)")
        hist_text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        # CrewAI Task - Anti Looping
        task_q = Task(
            description=f"History: {hist_text}. Masalah: {st.session_state.initial_tasks}. Kamu di tahap {st.session_state.q_index}/4. Analisa jawaban terakhir dan tanya 1 hal baru yang teknis. Gunakan 'kamu'.",
            agent=consultant, expected_output="Satu analisa pola dan satu pertanyaan baru."
        )
        with st.spinner("Auditor sedang berpikir..."):
            q_text = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(q_text)
        u_ans = st.text_area("Jawaban kamu:", key=f"ans_{st.session_state.q_index}")
        u_img = st.file_uploader("📷 Lampirkan Bukti Screenshot (Opsional)", accept_multiple_files=True, key=f"img_{st.session_state.q_index}")
        
        if st.button("Kirim Jawaban"):
            visual_info = process_images(u_img)
            st.session_state.chat_history.append({"q": q_text, "a": u_ans + f" [Visual Evidence: {visual_info}]"})
            if st.session_state.q_index < 4: 
                st.session_state.q_index += 1; st.rerun()
            else: 
                st.session_state.audit_stage = 'report'; st.rerun()

    elif st.session_state.audit_stage == 'report':
        st.subheader("🏁 Final Strategic Report")
        with st.spinner("Menyusun blueprint solusi..."):
            full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
            task_f = Task(description=f"History: {full_hist}.", agent=architect, expected_output="Laporan kaku dengan SKOR_FINAL.")
            report_res = str(Crew(agents=[architect], tasks=[task_f]).kickoff().raw)
        
        st.markdown(report_res)
        
        # Extraction & Database Saving
        score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", report_res, re.IGNORECASE)
        final_score = float(score_match.group(1)) if score_match else 0.0
        
        if not st.session_state.data_saved:
            # Simpan Log Audit
            supabase.table("audit_log").insert({"user_id": user_nickname, "score": final_score, "audit_report": report_res}).execute()
            # Simpan Action Items ke Sidebar
            for line in report_res.split("\n"):
                if "**" in line and ":" in line:
                    supabase.table("pending_tasks").insert({"user_id": user_nickname, "task_name": line.replace("*", "").strip(), "status": "Pending"}).execute()
            st.session_state.data_saved = True
            st.rerun() # Refresh untuk update sidebar
            
        # PDF Generation & Download
        current_pending = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").execute().data
        pdf_data = generate_pdf(user_nickname, report_res, final_score, current_pending)
        
        st.download_button("📥 UNDUH LAPORAN PDF (MEMORI)", data=pdf_data, file_name=f"Audit_{user_nickname}_{datetime.now().strftime('%Y%m%d')}.pdf", use_container_width=True)
        
        if st.button("Reset & Mulai Audit Baru"):
            st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False; st.rerun()

# ==========================================
# 6. TAB: DASHBOARD
# ==========================================
with menu[1]:
    st.subheader("Tren Performa Strategis")
    res_dash = supabase.table("audit_log").select("score, created_at").eq("user_id", user_nickname).order("created_at").execute()
    if res_dash.data:
        df = pd.DataFrame(res_dash.data)
        df['created_at'] = pd.to_datetime(df['created_at'])
        fig = px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10], title="Progress Skor Audit")
        st.plotly_chart(fig, use_container_width=True)
        st.table(df.sort_values(by='created_at', ascending=False))
    else:
        st.info("Belum ada data historis.")