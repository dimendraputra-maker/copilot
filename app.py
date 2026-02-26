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
        'data_saved': False,
        'current_user': None
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_state()

def get_user_context(nickname):
    try:
        res_audit = supabase.table("audit_log").select("score, audit_report, created_at").eq("user_id", nickname).order("created_at", desc=True).limit(3).execute()
        res_tasks = supabase.table("pending_tasks").select("status").eq("user_id", nickname).execute()
        total_tasks = len(res_tasks.data)
        completed = len([t for t in res_tasks.data if t['status'] == 'Completed'])
        pending = total_tasks - completed
        history_summary = "\n".join([f"- {d['created_at'][:10]} (Skor: {d['score']})" for d in res_audit.data])
        return f"HISTORI AUDIT: {history_summary if res_audit.data else 'Nol'}. STATS: Selesai {completed}, Pending {pending}."
    except: return "Konteks gagal dimuat."

def generate_pdf(nickname, report_text, score, tasks):
    pdf = FPDF()
    pdf.add_page()
    def clean_text(text):
        text = text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").replace("*", "-")
        return text.encode('ascii', 'ignore').decode('ascii')
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, txt=f"AUDIT REPORT: {clean_text(nickname)}", ln=True, align='C')
    pdf.set_font("Arial", size=10); pdf.cell(0, 10, txt=f"Gen: {datetime.now().strftime('%d %b %Y')}", ln=True, align='C'); pdf.ln(10)
    pdf.set_fill_color(230, 230, 230); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt=f"SCORE: {score}/10", ln=True, fill=True); pdf.ln(5)
    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 5, txt=clean_text(report_text))
    if tasks:
        pdf.ln(10); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt="ACTION ITEMS:", ln=True)
        for i, t in enumerate(tasks, 1):
            name = t['task_name'].split("|")[0] if "|" in t['task_name'] else t['task_name']
            pdf.multi_cell(0, 5, txt=f"{i}. [ ] {clean_text(name)}")
    return pdf.output(dest='S').encode('latin-1')

def process_images(files):
    descriptions = []
    for f in files:
        img = Image.open(f)
        response = vision_model.generate_content(["Ekstrak fakta teknis/angka.", img])
        descriptions.append(response.text)
    return " | ".join(descriptions)

def save_audit_to_db(user_input, audit_result, nickname):
    if st.session_state.data_saved: return 
    score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", audit_result, re.IGNORECASE)
    raw_score = float(score_match.group(1)) if score_match else 0.0
    score = raw_score / 10 if raw_score > 10 else raw_score
    supabase.table("audit_log").insert({"user_id": nickname, "score": score, "audit_report": audit_result, "input_preview": user_input[:100]}).execute()
    st.session_state.data_saved = True

def extract_and_save_tasks(audit_result, nickname):
    match = re.search(r"### ACTION_ITEMS\s*(.*?)(?:\n###|$)", audit_result, re.DOTALL | re.IGNORECASE)
    if not match: return
    content = match.group(1)
    valid_tasks = re.findall(r"\*\*(.+?)\*\*[:\-]\s*(.+)", content)
    for title, desc in valid_tasks[:6]:
        if len(title.strip()) > 5:
            supabase.table("pending_tasks").insert({"user_id": nickname, "task_name": f"{title.strip()} | {desc.strip()}", "status": "Pending"}).execute()

# ==========================================
# 3. AGENT SETUP (V10.2 - STATE-AWARE)
# ==========================================
consultant = Agent(
    role='Lead Strategic Auditor',
    goal='Mendiagnosa bottleneck operasional melalui interogasi progresif.',
    backstory="""Kamu adalah auditor tingkat tinggi yang dingin dan tajam. 
    WAJIB: Gunakan kata 'saya' dan 'kamu'. JANGAN PERNAH gunakan 'Anda' atau 'Saudara'.

    STANDAR INVESTIGASI:
    1. PROGRESIF: Jangan pernah mengulang pertanyaan yang sudah diajukan. Cek history dengan teliti.
    2. NO TEACHING: Berhenti menceramahi user. Fokus pada BUKTI operasional (checklist, jurnal, log).
    3. DINGIN: Jangan berikan apresiasi. Langsung bedah pola dalam 1 kalimat sebelum bertanya.
    
    PROTOKOL FASE:
    - Q1-Q3: Investigasi satu variabel baru per pertanyaan. 
    - Q4 (FINAL): Micro-Audit dan jalankan Continuity Protocol (PDF & Jadwal Kembali).""",
    llm=llm_gemini,
    allow_delegation=False
)

architect = Agent(
    role='High-Leverage Solutions Architect',
    goal='Memberikan blueprint solusi operasional kaku.',
    backstory="""Kamu arsitek yang benci inefisiensi. Wajib memberikan format:
    1. SKOR_FINAL: [0.0 - 10.0]
    2. ### DIAGNOSA_AWAL: Analisa kegagalan prosedur.
    3. ### ACTION_ITEMS: Tugas format **Nama Tugas**: Deskripsi teknis.
    4. ### CONTINUITY_PROTOCOL: Instruksi kembali dalam 7 hari dengan membawa PDF hari ini.""",
    llm=llm_gemini,
    allow_delegation=False
)

# ==========================================
# 4. TAMPILAN WEB & DASHBOARD
# ==========================================
st.set_page_config(page_title="Strategic Auditor V10.2", layout="wide")

if st.session_state.current_user is None:
    l, m, r = st.columns([1, 2, 1])
    with m:
        st.markdown("<h1 style='text-align: center;'>🔐 Access</h1>", unsafe_allow_html=True)
        u_name = st.text_input("Nickname:")
        u_pass = st.text_input("Password:", type="password")
        if st.button("Login", use_container_width=True):
            res = supabase.table("user_access").select("*").eq("username", u_name).execute()
            if res.data and res.data[0]['password'] == u_pass:
                st.session_state.current_user = u_name; st.rerun()
            else: st.error("Akses Ditolak.")
    st.stop()

# SIDEBAR (RESTORED & STABLE)
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
page = st.sidebar.radio("Navigasi:", ["Audit", "Dashboard"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Operational Checklist")
if st.sidebar.button("🗑️ Bersihkan Checklist"):
    supabase.table("pending_tasks").delete().eq("user_id", user_nickname).eq("status", "Pending").execute(); st.rerun()

res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
if res_tasks.data:
    for t in res_tasks.data:
        title = t['task_name'].split("|")[0]
        st.sidebar.caption(f"**{title}**")
        if st.sidebar.button("Selesaikan", key=f"t_{t['id']}", use_container_width=True):
            supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()

# --- HALAMAN AUDIT ---
if page == "Audit":
    st.title("Strategic Auditor V10.2")
    if st.session_state.audit_stage == 'input':
        st.info("Input tantanganmu. Gunakan PDF audit sebelumnya jika ada.")
        u_in = st.text_area("Apa yang ingin diaudit hari ini?", height=120)
        u_files = st.file_uploader("Upload PDF Audit Sebelumnya (Memori)", accept_multiple_files=True)
        if st.button("Mulai Audit"):
            if len(u_in) > 10:
                st.session_state.initial_evidence = process_images(u_files) if u_files else ""
                st.session_state.initial_tasks = u_in
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"🔍 Interogasi Progresif ({st.session_state.q_index}/4)")
        ret_date = (datetime.now() + timedelta(days=7)).strftime('%d %B %Y')
        
        # LOGIKA ANTI-LOOPING (V10.2)
        history_text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        if st.session_state.q_index < 4:
            exp_out = "Satu analisa singkat dan satu pertanyaan investigasi BARU. Gunakan 'kamu'."
            desc_task = f"""
            TUGAS: Kamu di Pertanyaan ke-{st.session_state.q_index}. 
            HISTORY PERCAKAPAN: {history_text}.
            
            PERINTAH KETAT:
            1. JANGAN PERNAH menanyakan hal yang sudah ditanyakan di history.
            2. Bedah jawaban terakhir user. 
            3. Jika user tidak punya checklist (sesuai jawaban sebelumnya), tanya tentang kriteria konfirmasi atau cara dia mereview kegagalan.
            4. Gunakan 'kamu'. Jangan menceramahi.
            """
        else:
            exp_out = f"Micro-Audit Final dan instruksi penutup. Gunakan 'kamu'."
            desc_task = f"HISTORY: {history_text}. Berikan diagnosa final. Perintahkan unduh PDF dan kembali {ret_date}."

        with st.spinner("Auditor sedang membedah data..."):
            task_q = Task(description=desc_task, agent=consultant, expected_output=exp_out)
            current_q = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(current_q)
        u_ans = st.text_area("Jawaban kamu:", key=f"ans_{st.session_state.q_index}")
        if st.button("Kirim"):
            st.session_state.chat_history.append({"q": current_q, "a": u_ans})
            if st.session_state.q_index < 4: st.session_state.q_index += 1
            else: st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_state.audit_stage == 'report':
        with st.spinner("Menyusun Blueprint..."):
            full_hist = "\n".join([f"Q{i+1}: {h['q']}\nA: {h['a']}" for i, h in enumerate(st.session_state.chat_history)])
            task_fin = Task(description=f"HISTORY: {full_hist}.", agent=architect, expected_output="SKOR_FINAL, ### DIAGNOSA_AWAL, ### ACTION_ITEMS, ### CONTINUITY_PROTOCOL.")
            res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw); st.markdown(res)
            save_audit_to_db(st.session_state.initial_tasks, res, user_nickname); extract_and_save_tasks(res, user_nickname)
            try:
                score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
                f_score = score_match.group(1) if score_match else "0.0"
                p_tasks = supabase.table("pending_tasks").select("task_name").eq("user_id", user_nickname).eq("status", "Pending").execute().data
                pdf_bytes = generate_pdf(user_nickname, res, f_score, p_tasks)
                st.download_button("📥 Simpan PDF (Memori Sesi Depan)", data=pdf_bytes, file_name=f"Audit_{user_nickname}.pdf")
            except: st.error("PDF Gagal.")
            if st.button("Selesai & Reset"): st.session_state.audit_stage, st.session_state.chat_history = 'input', []; st.rerun()

elif page == "Dashboard":
    st.title("📊 Performa")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data); df['created_at'] = pd.to_datetime(df['created_at'])
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
    else: st.info("Belum ada data.")