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
# 2. FUNGSI PENDUKUNG (DATA & PDF)
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

def clean_text_for_pdf(text):
    """Membersihkan teks dari karakter Markdown dan simbol non-latin agar PDF tidak crash."""
    text = text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").replace("*", "-")
    # Hanya izinkan karakter ASCII standar
    return text.encode('ascii', 'ignore').decode('ascii')

def generate_pdf(nickname, report_text, score, tasks):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"STRATEGIC AUDIT REPORT: {clean_text_for_pdf(nickname)}", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Generated: {datetime.now().strftime('%d %b %Y')}", ln=True, align='C')
    pdf.ln(10)
    
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"FINAL PERFORMANCE SCORE: {score}/10", ln=True, fill=True)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, txt=clean_text_for_pdf(report_text))
    
    if tasks:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="OPERATIONAL CHECKLIST:", ln=True)
        pdf.set_font("Arial", size=10)
        for i, t in enumerate(tasks, 1):
            name = t['task_name'].split("|")[0]
            pdf.multi_cell(0, 5, txt=f"{i}. [ ] {clean_text_for_pdf(name)}")
    return pdf.output(dest='S').encode('latin-1')

def extract_and_save_tasks(audit_result, nickname):
    """Ekstraksi tugas menggunakan Regex yang lebih kuat."""
    matches = re.findall(r"\*\*(.+?)\*\*[:\-]\s*(.+)", audit_result)
    for title, desc in matches:
        if len(title.strip()) > 3:
            full_data = f"{title.strip()} | {desc.strip()}"
            supabase.table("pending_tasks").insert({
                "user_id": nickname, "task_name": full_data, "status": "Pending"
            }).execute()

# ==========================================
# 3. AGENT SETUP (V12 - UNIVERSAL OPERATIONAL)
# ==========================================
consultant = Agent(
    role='Lead Strategic Auditor',
    goal='Mendiagnosa bottleneck operasional melalui interogasi progresif yang dalam.',
    backstory="""Kamu adalah auditor tingkat tinggi yang dingin, objektif, dan benci inefisiensi. 
    WAJIB: Gunakan kata 'saya' dan 'kamu'. DILARANG KERAS menggunakan kata 'Anda' atau 'Saudara'.

    STANDAR INVESTIGASI UNIVERSAL:
    1. ANALISA MENDALAM: Sebelum bertanya, berikan analisa tajam (3-4 kalimat) tentang pola kegagalan atau hambatan yang kamu temukan dari data user.
    2. OPERASIONAL: Jangan tanya perasaan. Tanya tentang PROSEDUR fisik, alat bantu (log/jurnal/checklist), dan bukti nyata.
    3. NO REPETITION: Cek history percakapan. Jangan pernah menanyakan hal yang sama dua kali.
    4. NO APPRECIATION: Jangan memuji. Langsung ke inti masalah secara dingin.

    PROTOKOL FASE:
    - Q1-Q3: Interogasi mendalam. Satu topik baru per tahap. 
    - Q4 (FINAL): Micro-Audit & Continuity Protocol (PDF & Jadwal Kembali).""",
    llm=llm_gemini,
    allow_delegation=False
)

architect = Agent(
    role='High-Leverage Solutions Architect',
    goal='Memberikan blueprint solusi strategis yang kaku dan terukur.',
    backstory="""Kamu arsitek yang benci inefisiensi. Wajib memberikan laporan dengan format kaku:
    1. SKOR_FINAL: [0.0 - 10.0]
    2. ### DIAGNOSA_AWAL: Bedah teknis kegagalan prosedur/strategi user.
    3. ### ACTION_ITEMS: Minimal 3 tugas format **Nama Tugas**: Deskripsi teknis.
    4. ### CONTINUITY_PROTOCOL: Instruksi kembali dalam 7-14 hari dengan membawa PDF hari ini sebagai memori.""",
    llm=llm_gemini,
    allow_delegation=False
)

# ==========================================
# 4. TAMPILAN WEB & SIDEBAR
# ==========================================
st.set_page_config(page_title="Strategic Auditor V12", layout="wide")

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

# --- SIDEBAR CHECKLIST ---
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
page = st.sidebar.radio("Navigasi:", ["Audit & Konsultasi", "Dashboard"])

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Checklist: {user_nickname}")
if st.sidebar.button("🗑️ Bersihkan Semua Tugas"):
    supabase.table("pending_tasks").delete().eq("user_id", user_nickname).eq("status", "Pending").execute(); st.rerun()

res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
if res_tasks.data:
    for t in res_tasks.data:
        title = t['task_name'].split("|")[0]
        st.sidebar.caption(f"**{title}**")
        if st.sidebar.button("Selesaikan", key=f"t_{t['id']}", use_container_width=True):
            supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()
        st.sidebar.markdown("---")

if st.sidebar.button("Keluar Sistem"): st.session_state.current_user = None; st.rerun()

# --- HALAMAN AUDIT ---
if page == "Audit & Konsultasi":
    st.title("Strategic Auditor V12")
    
    # INSTRUKSI UI (STABLE)
    st.warning("""
    ### **🛠️ Panduan Operasional**
    1. **Input**: Jelaskan tantangan teknis, rencana bisnis, atau masalah strategismu.
    2. **Memory Bridge**: Unggah PDF audit sebelumnya jika ini adalah sesi lanjutan.
    3. **Audit**: Jawab 4 pertanyaan investigatif. AI akan membedah prosedurmu secara dingin.
    4. **Simpan**: Di akhir, **Wajib Unduh PDF** sebagai jembatan memori untuk sesi berikutnya.
    """)

    if st.session_state.audit_stage == 'input':
        u_in = st.text_area("Apa tantangan strategis atau operasionalmu hari ini?", height=120)
        u_files = st.file_uploader("Upload PDF Memori / Bukti Visual", accept_multiple_files=True)
        if st.button("Mulai Audit"):
            if len(u_in) > 10:
                st.session_state.initial_tasks = u_in
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"🔍 Interogasi ({st.session_state.q_index}/4)")
        ret_date = (datetime.now() + timedelta(days=7)).strftime('%d %B %Y')
        
        # Anti-Loop Logic: Mengunci History
        history_text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])

        if st.session_state.q_index < 4:
            exp_out = "Analisa mendalam (3-4 kalimat) dan satu pertanyaan investigatif BARU. Gunakan 'kamu'."
            desc_task = f"""
            HISTORY: {history_text}. 
            TANTANGAN AWAL: {st.session_state.initial_tasks}.
            TUGAS: Tahap {st.session_state.q_index}/4. 
            1. Bedah jawaban user secara tajam. Berikan analisa teknis kenapa hal itu terjadi.
            2. JANGAN MENGULANG PERTANYAAN yang sudah ada di history. Maju ke variabel baru.
            3. Gunakan 'kamu'. Jangan menceramahi.
            """
        else:
            exp_out = f"Micro-Audit Final dan instruksi penutup. Gunakan 'kamu'."
            desc_task = f"HISTORY: {history_text}. Berikan diagnosa final. Perintahkan unduh PDF dan kembali {ret_date}."

        with st.spinner("Auditor sedang membedah sistem..."):
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
        with st.spinner("Menyusun Laporan Akhir..."):
            full_hist = "\n".join([f"Q{i+1}: {h['q']}\nA: {h['a']}" for i, h in enumerate(st.session_state.chat_history)])
            task_fin = Task(description=f"HISTORY: {full_hist}.", agent=architect, expected_output="SKOR_FINAL, ### DIAGNOSA_AWAL, ### ACTION_ITEMS, ### CONTINUITY_PROTOCOL.")
            res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw); st.markdown(res)
            
            # Save Logic
            score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
            f_score = score_match.group(1) if score_match else "0.0"
            if not st.session_state.data_saved:
                supabase.table("audit_log").insert({"user_id": user_nickname, "score": float(f_score), "audit_report": res, "input_preview": st.session_state.initial_tasks[:100]}).execute()
                extract_and_save_tasks(res, user_nickname)
                st.session_state.data_saved = True
            
            p_tasks = supabase.table("pending_tasks").select("task_name").eq("user_id", user_nickname).eq("status", "Pending").execute().data
            pdf_bytes = generate_pdf(user_nickname, res, f_score, p_tasks)
            st.download_button("📥 Simpan Laporan (PDF)", data=pdf_bytes, file_name=f"Audit_{user_nickname}.pdf")
            
            if st.button("Reset Sesi"):
                st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False; st.rerun()

elif page == "Dashboard":
    st.title("📊 Dashboard")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data); df['created_at'] = pd.to_datetime(df['created_at'])
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
    else: st.info("Kosong.")