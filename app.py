import os
import warnings
import re
import io
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
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 2. CORE FUNCTIONS (PDF, DATA, CSV)
# ==========================================
def init_state():
    defaults = {
        'audit_stage': 'input', 'q_index': 0, 'chat_history': [], 
        'initial_tasks': "", 'data_saved': False, 'current_user': None
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()

def clean_txt(text):
    """Sanitasi teks agar FPDF tidak error."""
    return text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").replace("*", "-").encode('ascii', 'ignore').decode('ascii')

def process_images(files):
    if not files: return ""
    descriptions = []
    for f in files:
        img = Image.open(f)
        res = vision_model.generate_content(["Ekstrak data teknis dan fakta objektif.", img])
        descriptions.append(res.text)
    return " | ".join(descriptions)

def generate_pdf(nickname, report_text, score, tasks):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, txt=f"STRATEGIC AUDIT: {clean_txt(nickname)}", ln=True, align='C')
    pdf.set_font("Arial", size=10); pdf.cell(0, 10, txt=f"Date: {datetime.now().strftime('%d %b %Y')}", ln=True, align='C'); pdf.ln(10)
    pdf.set_fill_color(230, 230, 230); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt=f"SCORE: {score}/10", ln=True, fill=True); pdf.ln(5)
    pdf.set_font("Arial", size=10); pdf.multi_cell(0, 5, txt=clean_txt(report_text))
    if tasks:
        pdf.ln(10); pdf.set_font("Arial", 'B', 12); pdf.cell(0, 10, txt="ACTION ITEMS:", ln=True)
        for i, t in enumerate(tasks, 1):
            pdf.multi_cell(0, 5, txt=f"{i}. [ ] {clean_txt(t['task_name'].split('|')[0])}")
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 3. AGENT SETUP (STRICT FORMATTING)
# ==========================================
consultant = Agent(
    role='Lead Strategic Auditor',
    goal='Mendiagnosa hambatan operasional melalui interogasi progresif.',
    backstory="""Kamu auditor senior yang dingin. Gunakan 'saya' dan 'kamu'. Dilarang kata 'Anda'.
    Berikan analisa teknis mendalam (3-4 kalimat) sebelum bertanya. Jangan mengulang pertanyaan history.""",
    llm=llm_gemini, allow_delegation=False
)

architect = Agent(
    role='Solutions Architect',
    goal='Menyusun blueprint solusi kaku.',
    backstory="""Kamu arsitek sistem. WAJIB memberikan format: 
    SKOR_FINAL: [0-10]
    ### DIAGNOSA_AWAL: ...
    ### ACTION_ITEMS: 
    **Nama Tugas**: Deskripsi Tugas
    **Nama Tugas**: Deskripsi Tugas
    ### CONTINUITY_PROTOCOL: ...""",
    llm=llm_gemini, allow_delegation=False
)

# ==========================================
# 4. UI FLOW: LOGIN GATE
# ==========================================
st.set_page_config(page_title="Strategic Auditor V16", layout="wide")

if st.session_state.current_user is None:
    _, col_mid, _ = st.columns([1, 2, 1])
    with col_mid:
        st.markdown("<h1 style='text-align: center;'>🔐 Strategic Access</h1>", unsafe_allow_html=True)
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

# --- SIDEBAR: CLEAN CHECKLIST ---
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")

st.sidebar.markdown("---")
st.sidebar.subheader("📋 Pending Checklist")

# Fungsi Hapus Semua Tugas
if st.sidebar.button("🗑️ Hapus Semua Tugas", use_container_width=True):
    supabase.table("pending_tasks").delete().eq("user_id", user_nickname).eq("status", "Pending").execute(); st.rerun()

# Menampilkan Daftar Tugas dengan Rapi
res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
if res_tasks.data:
    for t in res_tasks.data:
        task_title = t['task_name'].split("|")[0].strip()
        with st.sidebar.expander(f"📌 {task_title}"):
            st.caption(t['task_name'])
            if st.button("Selesaikan ✅", key=f"t_{t['id']}", use_container_width=True):
                supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute(); st.rerun()
else:
    st.sidebar.info("Tidak ada tugas pending.")

st.sidebar.markdown("---")
if st.sidebar.button("Keluar Systems"): st.session_state.current_user = None; st.rerun()

# ==========================================
# 5. MAIN PAGE: TABS
# ==========================================
tabs = st.tabs(["🔍 Audit Sesi", "📊 Dashboard & Excel Data"])

with tabs[0]:
    st.subheader("Fase Audit Strategis")
    
    if st.session_state.audit_stage == 'input':
        u_in = st.text_area("Apa tantangan strategis/teknismu hari ini?", height=150)
        u_f = st.file_uploader("Lampirkan PDF Memori / Bukti Visual", accept_multiple_files=True)
        if st.button("Mulai Audit"):
            if len(u_in) > 5:
                img_data = process_images(u_f)
                st.session_state.initial_tasks = u_in + f" [Evidence: {img_data}]"
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"🔍 Interogasi {st.session_state.q_index}/4")
        hist_text = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        task_q = Task(
            description=f"History: {hist_text}. Masalah: {st.session_state.initial_tasks}. Tahap {st.session_state.q_index}/4. Berikan analisa teknis 3-4 kalimat dan tanya 1 hal baru. Gunakan 'kamu'.",
            agent=consultant, expected_output="Analisa dan satu pertanyaan baru."
        )
        with st.spinner("Auditor sedang membedah sistem..."):
            q_text = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(q_text)
        u_ans = st.text_area("Jawaban kamu:", key=f"ans_{st.session_state.q_index}")
        u_img = st.file_uploader("📷 Lampirkan Bukti Visual", accept_multiple_files=True, key=f"img_{st.session_state.q_index}")
        
        if st.button("Kirim Jawaban"):
            visual_info = process_images(u_img)
            st.session_state.chat_history.append({"q": q_text, "a": u_ans + f" [Visual: {visual_info}]"})
            if st.session_state.q_index < 4: st.session_state.q_index += 1
            else: st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_state.audit_stage == 'report':
        st.subheader("🏁 Final Solution Blueprint")
        with st.spinner("Membangun laporan akhir..."):
            full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
            task_f = Task(description=f"History: {full_hist}.", agent=architect, expected_output="SKOR_FINAL: [0-10], ### DIAGNOSA_AWAL, ### ACTION_ITEMS, ### CONTINUITY_PROTOCOL.")
            report_res = str(Crew(agents=[architect], tasks=[task_f]).kickoff().raw)
        
        st.markdown(report_res)
        
        score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", report_res, re.IGNORECASE)
        final_score = float(score_match.group(1)) if score_match else 0.0
        
        if not st.session_state.data_saved:
            supabase.table("audit_log").insert({"user_id": user_nickname, "score": final_score, "audit_report": report_res}).execute()
            # PERBAIKAN REGEX CHECKLIST (V16)
            matches = re.findall(r"\*\*(.+?)\*\*[:\-]\s*(.+)", report_res)
            for title, desc in matches:
                if len(title.strip()) > 3:
                    supabase.table("pending_tasks").insert({
                        "user_id": user_nickname, 
                        "task_name": f"{title.strip()} | {desc.strip()}", 
                        "status": "Pending"
                    }).execute()
            st.session_state.data_saved = True; st.rerun()

        pdf_data = generate_pdf(user_nickname, report_res, final_score, res_tasks.data)
        st.download_button("📥 UNDUH LAPORAN PDF", data=pdf_data, file_name=f"Audit_{user_nickname}.pdf", use_container_width=True)
        
        if st.button("Reset & Mulai Audit Baru"):
            st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False; st.rerun()

# ==========================================
# 6. TAB: DASHBOARD & EXCEL VIEW
# ==========================================
with tabs[1]:
    st.subheader("📑 Riwayat Audit & Data Excel")
    res_dash = supabase.table("audit_log").select("created_at, score, audit_report").eq("user_id", user_nickname).order("created_at").execute()
    
    if res_dash.data:
        df = pd.DataFrame(res_dash.data)
        df['created_at'] = pd.to_datetime(df['created_at']).dt.strftime('%Y-%m-%d %H:%M')
        
        fig = px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10], title="Trend Skor")
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("### 📊 Excel View")
        st.dataframe(df.sort_values(by='created_at', ascending=False), use_container_width=True)
        
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Ekspor Semua Data ke Excel (CSV)", data=csv, file_name=f"History_{user_nickname}.csv", mime='text/csv')
    else:
        st.info("Belum ada data audit.")