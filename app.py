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
# 1. API & DATABASE
# ==========================================
API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY.strip()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SB_URL, SB_KEY)

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2)
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. CORE FUNCTIONS (VISION & PDF)
# ==========================================
def init_state():
    defaults = {
        'audit_stage': 'input', 'q_index': 0, 'chat_history': [],
        'initial_tasks': "", 'initial_evidence': "", 'data_saved': False, 'current_user': None
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()

def process_vision(files):
    """Membaca screenshot data atau PDF laporan audit lama sebagai memori."""
    if not files: return ""
    descriptions = []
    for f in files:
        try:
            img = Image.open(f)
            res = vision_model.generate_content(["Sebutkan poin strategis, skor terakhir, dan action items dari dokumen/gambar ini.", img])
            descriptions.append(res.text)
        except: continue
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
# 3. AGENT SETUP (DEEP & ADAPTIVE)
# ==========================================
consultant = Agent(
    role='Lead Strategic Copilot',
    goal='Mendiagnosa hambatan sistemik dengan memadukan input baru dan memori dari PDF/foto yang diunggah.',
    backstory="""Kamu auditor senior yang luwes namun tajam. Gunakan 'Saya' dan 'Kamu'. Gunakan bahasa yang mudah dipahami
    ANALISA: Wajib berikan analisa 2-3 paragraf mendalam sebelum bertanya.
    MEMORY BRIDGE: Jika user mengunggah PDF laporan lama, bedah isinya, bandingkan dengan kondisi sekarang, dan jangan tanyakan lagi hal yang sudah tuntas di laporan tersebut.
    TONE: Profesional, strategis, dan tidak kaku.""",
    llm=llm_gemini
)

architect = Agent(
    role='Solutions Architect',
    goal='Memberikan blueprint solusi strategis.',
    backstory="""Berikan laporan kaku: SKOR_FINAL: [0-10], ### DIAGNOSA_AWAL, ### ACTION_ITEMS (Format **Nama Tugas**: Deskripsi), ### CONTINUITY_PROTOCOL.""",
    llm=llm_gemini
)

# ==========================================
# 4. LOGIN & SIDEBAR
# ==========================================
st.set_page_config(page_title="Strategic Copilot V9.9", layout="wide")

if st.session_state.current_user is None:
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("<h1 style='text-align: center;'>🔐 Access Control</h1>", unsafe_allow_html=True)
        u_name = st.text_input("Nickname:")
        u_pass = st.text_input("Password:", type="password")
        if st.button("Masuk / Daftar", use_container_width=True):
            res = supabase.table("user_access").select("*").eq("username", u_name).execute()
            if res.data and res.data[0]['password'] == u_pass:
                st.session_state.current_user = u_name; st.rerun()
            elif not res.data:
                supabase.table("user_access").insert({"username": u_name, "password": u_pass}).execute()
                st.success("Terdaftar! Klik masuk."); st.rerun()
            else: st.error("Salah password.")
    st.stop()

# Sidebar Checklist
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
st.sidebar.markdown("### 📋 Pending Tasks")
res_t = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
for t in res_tasks := res_t.data:
    title = t['task_name'].split("|")[0].strip()
    with st.sidebar.expander(f"📌 {title}"):
        st.write(t['task_name'].split("|")[1] if "|" in t['task_name'] else "Eksekusi.")
        if st.button("Selesaikan", key=f"btn_{t['id']}", use_container_width=True):
            supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()

if st.sidebar.button("Keluar"): st.session_state.current_user = None; st.rerun()

# ==========================================
# 5. AUDIT PAGE
# ==========================================
page = st.tabs(["🔍 Sesi Audit", "📊 Dashboard"])

with page[0]:
    st.error("### ⚠️ INSTRUKSI: Masukkan tantangan baru & lampirkan PDF Laporan Lama untuk memori audit.")
    
    if st.session_state.audit_stage == 'input':
        u_in = st.text_area("Apa tantangan strategis/teknismu hari ini?", height=150)
        u_f = st.file_uploader("Upload Foto/PDF Laporan Lama (Memory Bridge)", accept_multiple_files=True)
        if st.button("Mulai Sesi"):
            if len(u_in) > 10:
                st.session_state.initial_evidence = process_vision(u_f)
                st.session_state.initial_tasks = u_in
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"Fase Interogasi {st.session_state.q_index}/4")
        hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        task_q = Task(
            description=f"Masalah: {st.session_state.initial_tasks}. Memori Dokumen: {st.session_state.initial_evidence}. History: {hist}. Tahap: {st.session_state.q_index}/4.",
            agent=consultant, expected_output="Analisa mendalam (2-3 paragraf) dan pertanyaan baru."
        )
        with st.spinner("Menganalisa..."):
            current_q = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(current_q)
        u_ans = st.text_area("Jawaban kamu:", key=f"ans_{st.session_state.q_index}")
        u_f_new = st.file_uploader("Upload Bukti Baru", accept_multiple_files=True, key=f"f_{st.session_state.q_index}")
        if st.button("Kirim Data"):
            new_ev = process_vision(u_f_new)
            st.session_state.chat_history.append({"q": current_q, "a": u_ans + f" [Visual: {new_ev}]"})
            if st.session_state.q_index < 4: st.session_state.q_index += 1
            else: st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_state.audit_stage == 'report':
        with st.spinner("Menyusun Laporan..."):
            full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
            task_fin = Task(description=f"History: {full_hist}.", agent=architect, expected_output="Blueprint solusi.")
            res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw); st.markdown(res)
            
            score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
            f_score = float(score_match.group(1)) if score_match else 0.0
            
            if not st.session_state.data_saved:
                supabase.table("audit_log").insert({"user_id": user_nickname, "score": f_score, "audit_report": res, "input_preview": st.session_state.initial_tasks[:100]}).execute()
                # Extraction Fixed
                action_section = re.search(r"### ACTION_ITEMS\s*(.*?)(?:\n###|$)", res, re.DOTALL | re.IGNORECASE)
                if action_section:
                    tasks = re.findall(r"\*\*(.+?)\*\*[:\-]\s*(.+)", action_section.group(1))
                    for title, desc in tasks:
                        supabase.table("pending_tasks").insert({"user_id": user_nickname, "task_name": f"{title.strip()} | {desc.strip()}", "status": "Pending"}).execute()
                st.session_state.data_saved = True; st.rerun()

            st.download_button("📥 Download PDF Laporan", data=generate_pdf(user_nickname, res, f_score), file_name=f"Audit_{user_nickname}.pdf")
            if st.button("Selesai & Reset"):
                st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False; st.rerun()

with page[1]:
    st.title("📈 Dashboard")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data)
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
        st.dataframe(df.sort_values(by='created_at', ascending=False), use_container_width=True)