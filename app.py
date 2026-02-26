import os
import warnings
import logging
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
        rate = (completed / total_tasks * 100) if total_tasks > 0 else 0
        
        history_summary = "\n".join([f"- {d['created_at'][:10]} (Skor: {d['score']})" for d in res_audit.data])
        return f"""
        HISTORI AUDIT TERAKHIR:
        {history_summary if res_audit.data else "Belum ada histori."}
        
        STATISTIK EKSEKUSI NYATA:
        - Total Tugas: {total_tasks}, Selesai: {completed}, Mangkrak: {pending}
        - Rasio Keberhasilan: {rate:.1f}%
        """
    except:
        return "Gagal mengambil konteks histori."

def generate_pdf(nickname, report_text, score, tasks):
    pdf = FPDF()
    pdf.add_page()
    def clean_text(text):
        text = text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").replace("*", "-")
        return text.encode('ascii', 'ignore').decode('ascii')
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(0, 10, txt=f"STRATEGIC AUDIT REPORT: {clean_text(nickname)}", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(0, 10, txt=f"Generated: {datetime.now().strftime('%d %b %Y | %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    pdf.set_fill_color(230, 230, 230)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt=f"PERFORMANCE SCORE: {score}/10", ln=True, fill=True)
    pdf.ln(5)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, txt="STRATEGIC BLUEPRINT:", ln=True)
    pdf.set_font("Arial", size=10)
    pdf.multi_cell(0, 5, txt=clean_text(report_text))
    if tasks:
        pdf.ln(10)
        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="OPERATIONAL CHECKLIST:", ln=True)
        pdf.set_font("Arial", size=10)
        for i, t in enumerate(tasks, 1):
            name = t['task_name'].split("|")[0] if "|" in t['task_name'] else t['task_name']
            pdf.multi_cell(0, 5, txt=f"{i}. [ ] {clean_text(name)}")
    return pdf.output(dest='S').encode('latin-1')

def process_images(files):
    descriptions = []
    for f in files:
        img = Image.open(f)
        response = vision_model.generate_content(["Identifikasi fakta teknis dan data objektif.", img])
        descriptions.append(response.text)
    return " | ".join(descriptions)

def save_audit_to_db(user_input, audit_result, nickname):
    if st.session_state.data_saved: return 
    score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", audit_result, re.IGNORECASE)
    raw_score = float(score_match.group(1)) if score_match else 0.0
    score = raw_score / 10 if raw_score > 10 else raw_score
    supabase.table("audit_log").insert({
        "user_id": nickname, "score": score, "audit_report": audit_result, "input_preview": user_input[:100]
    }).execute()
    st.session_state.data_saved = True

def extract_and_save_tasks(audit_result, nickname):
    match = re.search(r"### ACTION_ITEMS\s*(.*?)(?:\n###|$)", audit_result, re.DOTALL | re.IGNORECASE)
    if not match: return
    content = match.group(1)
    valid_tasks = re.findall(r"\*\*(.+?)\*\*[:\-]\s*(.+)", content)
    for title, desc in valid_tasks[:6]:
        clean_title = title.strip()
        clean_desc = re.sub(r"\(?deadline.*", "", desc, flags=re.IGNORECASE).strip()
        if len(clean_title) > 5:
            full_data = f"{clean_title} | {clean_desc}"
            supabase.table("pending_tasks").insert({
                "user_id": nickname, "task_name": full_data, "status": "Pending"
            }).execute()

# ==========================================
# 3. AGENT SETUP (V9.7 - STRATEGIC AUDITOR)
# ==========================================
consultant = Agent(
    role='Lead Strategic Auditor',
    goal='Mendiagnosa akar masalah melalui interogasi teknis 4 tahap tanpa kebocoran fase.',
    backstory="""Kamu adalah auditor tingkat tinggi yang dingin, objektif, dan sangat benci asumsi buta. 
    Kamu dilarang memberikan apresiasi atau basa-basi. Gunakan 'saya' dan 'kamu'.

    PROTOKOL INTEROGASI KETAT (WAJIB):
    1. PERTANYAAN 1-3: Fokus 100% pada penggalian data teknis, variansi, dan bukti objektif. 
       DILARANG menyuruh user mengunduh PDF atau memberikan instruksi 'kembali lagi di masa depan' pada tahap ini.
    
    2. PERTANYAAN 4 (FINAL): Lakukan Micro-Audit berdasarkan semua jawaban sebelumnya. 
       HANYA pada tahap ini kamu wajib menjalankan 'Continuity Protocol':
       - Mengunduh PDF laporan hari ini.
       - Instruksi kembali dalam 7-14 hari dengan data baru.
       - Memberikan file PDF hari ini sebagai 'Memori Konteks' di sesi depan.

    Jangan melompati urutan ini. Fokus pada satu tugas per tahap.""",
    llm=llm_gemini,
    allow_delegation=False
)

architect = Agent(
    role='High-Leverage Solutions Architect',
    goal='Memberikan blueprint solusi kaku dengan instruksi keberlanjutan konteks.',
    backstory="""Kamu arsitek yang benci inefisiensi. Wajib memberikan output:
    1. SKOR_FINAL: [0.0 - 10.0]
    2. ### DIAGNOSA_AWAL: Analisa pola teknis.
    3. ### ACTION_ITEMS: Tugas dengan format **Nama Tugas**: Deskripsi teknis.
    4. ### CONTINUITY_PROTOCOL: Instruksi detail kapan harus kembali dan perintah mengunggah PDF hari ini di sesi mendatang.""",
    llm=llm_gemini,
    allow_delegation=False
)

# ==========================================
# 4. TAMPILAN WEB & OTENTIKASI SISTEM
# ==========================================
st.set_page_config(page_title="Strategic Auditor V9.7", layout="wide")

def manage_access(name, password):
    if not name or name.strip() == "": return False
    try:
        res = supabase.table("user_access").select("*").eq("username", name).execute()
        if not res.data:
            st.info(f"Nickname '{name}' baru. Daftarkan?")
            if st.button("Daftarkan Akun"):
                supabase.table("user_access").insert({"username": name, "password": password}).execute()
                st.success("Berhasil! Silakan login.")
                st.rerun()
            return False
        return res.data[0]['password'] == password
    except: return False

# LOGIN TENGAH
if st.session_state.current_user is None:
    l, m, r = st.columns([1, 2, 1])
    with m:
        st.markdown("<h1 style='text-align: center;'>🔐 Secure Access</h1>", unsafe_allow_html=True)
        u_name = st.text_input("Nickname:")
        u_pass = st.text_input("Password:", type="password")
        if st.button("Login", use_container_width=True):
            if manage_access(u_name, u_pass):
                st.session_state.current_user = u_name
                st.rerun()
            else: st.error("Akses Ditolak.")
    st.stop()

# NAVIGATION
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
page = st.sidebar.radio("Navigasi:", ["Audit & Konsultasi", "Dashboard"])
if st.sidebar.button("Log Out"):
    st.session_state.current_user = None
    st.rerun()

# SIDEBAR CHECKLIST
st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Operational Checklist")
if st.sidebar.button("🗑️ Bersihkan Checklist"):
    supabase.table("pending_tasks").delete().eq("user_id", user_nickname).eq("status", "Pending").execute()
    st.rerun()

res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
if res_tasks.data:
    for task in res_tasks.data:
        title, desc = task['task_name'].split("|") if "|" in task['task_name'] else (task['task_name'], "Eksekusi.")
        st.sidebar.markdown(f"**{title}**")
        st.sidebar.caption(desc)
        if st.sidebar.button("Selesaikan", key=f"t_{task['id']}", use_container_width=True):
            supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", task['id']).execute()
            st.rerun()

# HALAMAN UTAMA
if page == "Audit & Konsultasi":
    st.title(f"Lead Auditor AI: {user_nickname}")
    st.markdown("---")

    if st.session_state.audit_stage == 'input':
        st.warning("""
        ### **🛠️ Panduan Operasional**
        1. **Input**: Jelaskan masalah teknis atau rencana strategismu.
        2. **Memori**: Jika ini sesi lanjutan, **unggah PDF audit sebelumnya**.
        3. **Continuity**: Simpan PDF hasil hari ini untuk 'jembatan memori' sesi depan.
        """)
        u_in = st.text_area("Apa yang ingin kamu audit hari ini?", height=120)
        u_files = st.file_uploader("Upload Bukti/PDF Audit Sebelumnya", accept_multiple_files=True)
        if st.button("Mulai Sesi"):
            if len(u_in) > 10:
                st.session_state.initial_evidence = process_images(u_files) if u_files else ""
                st.session_state.initial_tasks = u_in
                st.session_state.audit_stage = 'interrogation'
                st.session_state.q_index = 1
                st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"🔍 Interogasi Auditor ({st.session_state.q_index}/4)")
        
        # LOGIKA PENGUNCIAN FASE (V9.7)
        today = datetime.now()
        ret_date = (today + timedelta(days=7)).strftime('%d %B %Y')

        if st.session_state.q_index < 4:
            exp_out = "Satu pertanyaan teknis spesifik. DILARANG memberikan instruksi penutup atau Continuity Protocol."
            desc_task = f"Gali data teknis terkait: {st.session_state.initial_tasks}. Sejarah: {st.session_state.chat_history}. Fokus pada satu variabel spesifik."
        else:
            exp_out = f"Micro-Audit pola dan Instruksi Continuity Protocol untuk kembali pada {ret_date}."
            desc_task = f"Lakukan kesimpulan (Micro-Audit) dari history: {st.session_state.chat_history}. Perintahkan user unduh PDF dan kembali pada {ret_date}."

        with st.spinner("Auditor sedang berpikir..."):
            task_q = Task(description=desc_task, agent=consultant, expected_output=exp_out)
            current_q = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(current_q)
        u_ans = st.text_area("Jawaban kamu:", key=f"ans_{st.session_state.q_index}")
        u_img = st.file_uploader("Lampirkan Bukti", accept_multiple_files=True, key=f"img_{st.session_state.q_index}")
        if st.button("Kirim"):
            img_data = f" [Bukti: {process_images(u_img)}]" if u_img else ""
            st.session_state.chat_history.append({"q": current_q, "a": u_ans + img_data})
            if st.session_state.q_index < 4:
                st.session_state.q_index += 1
            else: st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_state.audit_stage == 'report':
        with st.spinner("Menyusun Blueprint Final..."):
            hist_context = get_user_context(user_nickname)
            full_hist = "\n".join([f"Q{i+1}: {h['q']}\nA: {h['a']}" for i, h in enumerate(st.session_state.chat_history)])
            task_fin = Task(
                description=f"TANGGAL: {datetime.now().strftime('%Y-%m-%d')}. DB: {hist_context}. INTERAKSI: {full_hist}.",
                agent=architect,
                expected_output="Laporan kaku dengan SKOR_FINAL, DIAGNOSA_AWAL, ACTION_ITEMS, dan CONTINUITY_PROTOCOL."
            )
            res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw)
            st.markdown(res)
            save_audit_to_db(st.session_state.initial_tasks, res, user_nickname)
            extract_and_save_tasks(res, user_nickname)
            try:
                score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
                f_score = score_match.group(1) if score_match else "0.0"
                p_tasks = supabase.table("pending_tasks").select("task_name").eq("user_id", user_nickname).eq("status", "Pending").execute().data
                pdf_bytes = generate_pdf(user_nickname, res, f_score, p_tasks)
                st.download_button("📥 Unduh Laporan (Simpan untuk Sesi Depan)", data=pdf_bytes, file_name=f"Audit_{user_nickname}.pdf")
            except: st.error("PDF Error.")
            if st.button("Selesai & Reset"):
                st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False
                st.rerun()

elif page == "Dashboard":
    st.title("📊 Dashboard Performa")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data); df['created_at'] = pd.to_datetime(df['created_at'])
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
        st.subheader("Riwayat Audit")
        st.dataframe(df.sort_values(by='created_at', ascending=False), use_container_width=True)
    else: st.info("Belum ada data.")