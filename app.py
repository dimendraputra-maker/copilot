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
# 2. SISTEM DATA & PDF
# ==========================================
def init_state():
    defaults = {
        'audit_stage': 'input', 'q_index': 0, 'chat_history': [],
        'initial_tasks': "", 'initial_evidence': "", 'data_saved': False, 'current_user': None
    }
    for key, val in defaults.items():
        if key not in st.session_state: st.session_state[key] = val

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
# 3. AGENT SETUP (DEEP-DIVE LOGIC)
# ==========================================
consultant = Agent(
    role='Lead Strategic Copilot',
    goal='Mendiagnosa hambatan sistemik dan operasional melalui interogasi mendalam.',
    backstory="""Kamu adalah Auditor Strategis senior yang sangat tajam dan luwes. 
    Tugasmu bukan sekadar bertanya, tapi menjadi "cermin" bagi user.

    PROTOKOL ANALISA (WAJIB):
    1. ANALISA MENDALAM: Sebelum bertanya, berikan minimal 2-3 paragraf analisa mendalam mengenai pola masalah user. Bedah dari sisi operasional, risiko, dan dampak jangka panjang. 
    2. BAHASA LUWES: Gunakan bahasa yang profesional namun mengalir, tidak kaku, dan mudah dipahami. Gunakan 'Saya' dan 'Kamu'.
    3. NO REPETITION: Jangan tanyakan hal yang sudah ada di history.
    4. UNIVERSAL: Deteksi konteks (Bisnis, Manajemen, Prosedur). Berikan wawasan strategis sebelum meminta data baru.""",
    llm=llm_gemini
)

architect = Agent(
    role='High-Leverage Solutions Architect',
    goal='Memberikan blueprint solusi strategis.',
    backstory="""Kamu ahli efisiensi. Berikan laporan dengan format ketat: 
    SKOR_FINAL: [0-10]
    ### DIAGNOSA_AWAL: ...
    ### ACTION_ITEMS: (Format **Nama Tugas**: Deskripsi teknis)
    ### CONTINUITY_PROTOCOL: ...""",
    llm=llm_gemini
)

# ==========================================
# 4. TAMPILAN WEB & LOGIN (TENGAH)
# ==========================================
st.set_page_config(page_title="Strategic Copilot V9.6", layout="wide")

if st.session_state.current_user is None:
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("<h1 style='text-align: center;'>🔐 Secure Access</h1>", unsafe_allow_html=True)
        u_name = st.text_input("Nickname:")
        u_pass = st.text_input("Password:", type="password")
        if st.button("Masuk / Daftar Akun", use_container_width=True):
            if u_name and u_pass:
                res = supabase.table("user_access").select("*").eq("username", u_name).execute()
                if res.data:
                    if res.data[0]['password'] == u_pass:
                        st.session_state.current_user = u_name; st.rerun()
                    else: st.error("Password salah.")
                else:
                    supabase.table("user_access").insert({"username": u_name, "password": u_pass}).execute()
                    st.success("Akun terdaftar! Klik masuk lagi."); st.rerun()
    st.stop()

# --- SIDEBAR: CHECKLIST RAPI ---
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
st.sidebar.markdown("### 📋 Action Items")

res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
if res_tasks.data:
    for t in res_tasks.data:
        title = t['task_name'].split("|")[0].strip()
        desc = t['task_name'].split("|")[1].strip() if "|" in t['task_name'] else "Eksekusi segera."
        with st.sidebar.expander(f"📌 {title}"):
            st.write(desc)
            if st.button("Selesaikan", key=f"btn_{t['id']}", use_container_width=True):
                supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()

if st.sidebar.button("Keluar"): st.session_state.current_user = None; st.rerun()

# ==========================================
# 5. HALAMAN UTAMA (AUDIT & DASHBOARD)
# ==========================================
page = st.tabs(["🔍 Audit Strategis", "📊 Dashboard Performa"])

with page[0]:
    # --- BAGIAN INSTRUKSI (NEW) ---
    st.error("""
    ### ⚠️ **INSTRUKSI OPERASIONAL COPILOT**
    1. **Identifikasi Masalah**: Masukkan tantangan operasional atau rencana strategismu secara utuh.
    2. **Transparansi Data**: Jawab pertanyaan interogasi dengan data jujur. Analisa AI bergantung pada kualitas jawabanmu.
    3. **Observasi Analisa**: Baca analisa mendalam AI di setiap tahap untuk memahami 'blind spot' strategismu.
    4. **Blueprint & Check**: Di akhir sesi, AI akan memberikan Skor dan Action Items yang akan otomatis masuk ke Sidebar.
    """)
    st.markdown("---")

    if st.session_state.audit_stage == 'input':
        u_in = st.text_area("Apa tantangan strategis/teknis yang ingin kamu bedah hari ini?", height=150)
        if st.button("Mulai Sesi Audit"):
            if len(u_in) > 10:
                st.session_state.initial_tasks = u_in
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"Fase Interogasi {st.session_state.q_index}/4")
        hist_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        task_q = Task(
            description=f"Masalah: {st.session_state.initial_tasks}. History: {hist_str}. Tahap: {st.session_state.q_index}/4.",
            agent=consultant, expected_output="Analisa mendalam (2-3 paragraf) dan satu pertanyaan baru."
        )
        with st.spinner("Copilot sedang membedah sistem..."):
            current_q = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(current_q)
        u_ans = st.text_area("Input Jawaban:", key=f"ans_{st.session_state.q_index}")
        if st.button("Kirim Analisa"):
            st.session_state.chat_history.append({"q": current_q, "a": u_ans})
            if st.session_state.q_index < 4: st.session_state.q_index += 1
            else: st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_state.audit_stage == 'report':
        with st.spinner("Menyusun Strategi Final..."):
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

            pdf_bytes = generate_pdf(user_nickname, res, f_score)
            st.download_button("📥 Download PDF Laporan", data=pdf_bytes, file_name=f"Audit_{user_nickname}.pdf")
            if st.button("Selesai & Reset Sesi"):
                st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False; st.rerun()

with page[1]:
    st.title("📈 Dashboard Strategis")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data)
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
        st.dataframe(df.sort_values(by='created_at', ascending=False), use_container_width=True)