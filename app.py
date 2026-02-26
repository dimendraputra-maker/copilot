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
        response = vision_model.generate_content(["Identifikasi fakta teknis dan data objektif.", img])
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
# 3. AGENT SETUP (V10.1 - RESTORED)
# ==========================================
consultant = Agent(
    role='Lead Strategic Auditor',
    goal='Mendiagnosa bottleneck operasional melalui tindakan fisik dan prosedur nyata.',
    backstory="""Kamu adalah auditor tingkat tinggi yang dingin dan tajam. 
    WAJIB: Gunakan kata 'saya' dan 'kamu'. JANGAN PERNAH gunakan kata 'Anda', 'Saudara', atau 'Beliau'.

    STANDAR INVESTIGASI (STRICT PROTOCOL):
    1. NO TEACHING: Berhenti bersikap seperti guru. Jangan tanya teori atau perasaan. 
    2. OPERATIONAL FOCUS: Fokus pada PROSEDUR. Tanya tentang checklist, jurnal, atau langkah fisik (misal: "Apa yang kamu lakukan saat harga masuk zona?").
    3. ADAPTIVE ANALOGY: Gunakan analogi maksimal 1 kalimat untuk istilah teknis bagi orang awam.
    4. NO APPRECIATION: Jangan memuji jawaban user. Langsung bedah polanya secara dingin.

    PROTOKOL FASE:
    - Q1-Q3: Investigasi Prosedural. Dilarang instruksi penutup/PDF.
    - Q4 (FINAL): Micro-Audit & Continuity Protocol (Unduh PDF & Jadwal Kembali).""",
    llm=llm_gemini,
    allow_delegation=False
)

architect = Agent(
    role='High-Leverage Solutions Architect',
    goal='Memberikan blueprint solusi operasional berdasarkan kegagalan sistem user.',
    backstory="""Kamu arsitek yang benci inefisiensi. Wajib memberikan output kaku dengan format:
    1. SKOR_FINAL: [0.0 - 10.0]
    2. ### DIAGNOSA_AWAL: Analisa kegagalan prosedur.
    3. ### ACTION_ITEMS: Tugas dengan format **Nama Tugas**: Deskripsi teknis.
    4. ### CONTINUITY_PROTOCOL: Instruksi detail kapan harus kembali dan perintah mengunggah PDF hari ini.""",
    llm=llm_gemini,
    allow_delegation=False
)

# ==========================================
# 4. TAMPILAN WEB & LOGIN
# ==========================================
st.set_page_config(page_title="Strategic Auditor V10.1", layout="wide")

def manage_access(name, password):
    if not name or name.strip() == "": return False
    try:
        res = supabase.table("user_access").select("*").eq("username", name).execute()
        if not res.data:
            st.info(f"Nickname '{name}' belum terdaftar.")
            if st.button("Daftarkan Akun Baru"):
                supabase.table("user_access").insert({"username": name, "password": password}).execute()
                st.success("Berhasil! Silakan login kembali."); st.rerun()
            return False
        return res.data[0]['password'] == password
    except: return False

# --- LOGIKA LOGIN ---
if st.session_state.current_user is None:
    l, m, r = st.columns([1, 2, 1])
    with m:
        st.markdown("<h1 style='text-align: center;'>🔐 Access Control</h1>", unsafe_allow_html=True)
        u_name = st.text_input("Nickname:")
        u_pass = st.text_input("Password:", type="password")
        if st.button("Login ke Sistem", use_container_width=True):
            if manage_access(u_name, u_pass): st.session_state.current_user = u_name; st.rerun()
            else: st.error("Akses Ditolak.")
    st.stop()

# --- NAVIGATION ---
user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
page = st.sidebar.radio("Navigasi:", ["Audit & Konsultasi", "Dashboard"])
if st.sidebar.button("Keluar Sistem"): st.session_state.current_user = None; st.rerun()

# --- SIDEBAR CHECKLIST (RESTORED) ---
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Checklist: {user_nickname}")
if st.sidebar.button("🗑️ Hapus Tugas Mangkrak"):
    supabase.table("pending_tasks").delete().eq("user_id", user_nickname).eq("status", "Pending").execute(); st.rerun()

res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
if res_tasks.data:
    for task in res_tasks.data:
        title, desc = task['task_name'].split("|") if "|" in task['task_name'] else (task['task_name'], "Eksekusi segera.")
        st.sidebar.markdown(f"**{title}**")
        st.sidebar.caption(desc)
        if st.sidebar.button("Selesaikan", key=f"btn_{task['id']}", use_container_width=True):
            supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", task['id']).execute(); st.rerun()
        st.sidebar.markdown("---")

# --- HALAMAN AUDIT ---
if page == "Audit & Konsultasi":
    st.title("Strategic Auditor AI")
    st.markdown("---")

    if st.session_state.audit_stage == 'input':
        st.warning("""
        ### **🛠️ Panduan Operasional Konsultasi**
        1. **Input Tantangan**: Jelaskan kendala teknis atau rencana strategismu secara detail.
        2. **Gunakan Memori**: Jika ini sesi lanjutan, **unggah PDF audit sebelumnya** untuk menjaga konteks AI.
        3. **Interaksi**: Jawab 4 tahap interogasi dari AI Auditor untuk hasil maksimal.
        4. **Continuity**: Simpan PDF hasil akhir sebagai 'jembatan memori' untuk audit berikutnya.
        """)
        u_in = st.text_area("Apa tantangan teknis atau rencana yang ingin kamu audit?", height=120)
        u_files = st.file_uploader("Upload Bukti Visual/PDF Audit Sebelumnya", accept_multiple_files=True)
        if st.button("Mulai Analisis"):
            if len(u_in) > 10:
                with st.spinner("Menginisialisasi auditor..."):
                    st.session_state.initial_evidence = process_images(u_files) if u_files else ""
                    st.session_state.initial_tasks = u_in
                    st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"🔍 Interogasi Auditor ({st.session_state.q_index}/4)")
        ret_date = (datetime.now() + timedelta(days=7)).strftime('%d %B %Y')

        if st.session_state.q_index < 4:
            exp_out = "Analisa pola operasional singkat dan satu pertanyaan tindakan fisik. GUNAKAN 'KAMU'."
            desc_task = f"Analisa: {st.session_state.initial_tasks}. History: {st.session_state.chat_history}. Fokus pada PROSEDUR, bukan perasaan. Pakai kata 'kamu'."
        else:
            exp_out = f"Micro-Audit Final dan instruksi kembali pada {ret_date}. Pakai kata 'kamu'."
            desc_task = f"Lakukan kesimpulan operasional final. Perintahkan unduh PDF dan kembali {ret_date}."

        with st.spinner("Auditor sedang membedah prosedur..."):
            task_q = Task(description=desc_task, agent=consultant, expected_output=exp_out)
            current_q = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
        
        st.info(current_q)
        u_ans = st.text_area("Jawaban kamu:", key=f"ans_{st.session_state.q_index}")
        u_img = st.file_uploader("Lampirkan Bukti Visual", accept_multiple_files=True, key=f"img_{st.session_state.q_index}")
        if st.button("Kirim Data"):
            img_data = f" [Bukti: {process_images(u_img)}]" if u_img else ""
            st.session_state.chat_history.append({"q": current_q, "a": u_ans + img_data})
            if st.session_state.q_index < 4: st.session_state.q_index += 1
            else: st.session_state.audit_stage = 'report'
            st.rerun()

    elif st.session_state.audit_stage == 'report':
        with st.spinner("Membangun Blueprint Solusi..."):
            hist_context = get_user_context(user_nickname)
            full_hist = "\n".join([f"Q{i+1}: {h['q']}\nA: {h['a']}" for i, h in enumerate(st.session_state.chat_history)])
            task_fin = Task(description=f"DB: {hist_context}. INTERAKSI: {full_hist}.", agent=architect, expected_output="SKOR_FINAL, ### DIAGNOSA_AWAL, ### ACTION_ITEMS, ### CONTINUITY_PROTOCOL.")
            res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw); st.markdown(res)
            save_audit_to_db(st.session_state.initial_tasks, res, user_nickname); extract_and_save_tasks(res, user_nickname)
            try:
                score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
                f_score = score_match.group(1) if score_match else "0.0"
                p_tasks = supabase.table("pending_tasks").select("task_name").eq("user_id", user_nickname).eq("status", "Pending").execute().data
                pdf_bytes = generate_pdf(user_nickname, res, f_score, p_tasks)
                st.download_button("📥 Unduh Laporan (Memori Sesi Depan)", data=pdf_bytes, file_name=f"Audit_{user_nickname}.pdf")
            except: st.error("Gagal mencetak PDF.")
            if st.button("Selesai & Reset"): st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved = 'input', [], False; st.rerun()

elif page == "Dashboard":
    st.title("📊 Dashboard Performa")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data); df['created_at'] = pd.to_datetime(df['created_at'])
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
        st.subheader("Riwayat Detail")
        st.dataframe(df.sort_values(by='created_at', ascending=False), use_container_width=True)
    else: st.info("Belum ada data audit tercatat.")