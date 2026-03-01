import os
import warnings
import re
import time # Tambahan untuk tracking durasi
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
# 2. CORE FUNCTIONS (VISION, PDF, & ANALYTICS)
# ==========================================
def init_state():
    defaults = {
        'audit_stage': 'input', 'q_index': 0, 'chat_history': [],
        'initial_tasks': "", 'initial_evidence': "", 'data_saved': False, 
        'current_user': None, 'start_time': None # Tambahan start_time
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()

# Fungsi baru untuk simpan analitik ke Supabase
def save_to_analytics(nickname, duration, accuracy, clarity, readiness, critique, word_count):
    try:
        data = {
            "user_nickname": nickname,
            "time_spent": float(duration),
            "ai_accuracy": int(accuracy),
            "clarity_score": int(clarity),
            "action_readiness": int(readiness),
            "critique": str(critique),
            "word_count_input": int(word_count)
        }
        supabase.table("audit_analytics").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"Gagal mencatat data: {e}")
        return False

def process_vision(files):
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
# 3. AGENT SETUP
# ==========================================
consultant = Agent(
    role='Lead Strategic Copilot',
    goal='Mendiagnosa hambatan sistemik dengan gaya percakapan konsultan senior.',
    backstory="""Kamu adalah konsultan bisnis papan atas. Gunakan 'Saya' dan 'Kamu'.
    
    ATURAN INTERAKSI (WAJIB):
    1. **Jangan Seperti Formulir**: Jangan pernah memberikan daftar pertanyaan 1, 2, 3. 
    2. **Gaya Percakapan**: Berikan tanggapan/validasi singkat atas jawaban user sebelumnya, lalu ajukan **maksimal 1-2 pertanyaan** yang paling krusial saja.
    3. **Empati & Tajam**: Bertanyalah seperti sedang coaching. Gali "kenapa" dan "bagaimana" secara mendalam, juga jangan lupa menanyakan data teknis yang diperlukan.
    4. **Memory Bridge**: Gunakan data dari PDF/Foto untuk menantang jawaban user jika tidak sinkron(jika ada).""",
    llm=llm_gemini
)

architect = Agent(
    role='Solutions Architect',
    goal='Menyusun Blueprint Strategis yang komprehensif dan mudah dibaca.',
    backstory="""Kamu adalah ahli strategi senior. Tugasmu adalah menyusun laporan yang rapi, profesional, dan enak dibaca (scannable).
    
    ATURAN FORMATTING (WAJIB):
    1. Gunakan Jeda 2 Baris Kosong antar sub-judul (Heading).
    2. Gunakan Jeda 1 Baris Kosong antar paragraf dalam satu bagian.
    3. JANGAN menuliskan poin-poin dalam satu paragraf panjang.
    
    STRUKTUR LAPORAN:
    # 📝 BLUEPRINT SOLUSI STRATEGIS
    
    ## 📊 SKOR_FINAL: [Skor]/10
    
    ### 🎯 ANALISA KONKLUSIF
    (Tuliskan rangkuman dalam 2-3 paragraf pendek yang dipisahkan baris baru).
    
    ### 🚀 STRATEGIC ROADMAP
    (Berikan penjelasan alur fase per fase, gunakan baris baru untuk setiap fase).
    
    ### 📋 ACTION_ITEMS
    (PENTING: Gunakan format ini dan WAJIB ganti baris untuk setiap poin agar Sidebar muncul):
    **Nama Tugas**: Deskripsi teknis lengkap.
    
    **Nama Tugas**: Deskripsi teknis lengkap.
    
    ### 🛡️ PROTOKOL KEBERLANJUTAN
    (Berikan tips dalam bentuk paragraf atau list yang rapi).""",
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

user_nickname = st.session_state.current_user
st.sidebar.title(f"👤 {user_nickname}")
st.sidebar.markdown("### 📋 Pending Tasks")

# Load Tasks
res_t = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()
res_tasks_data = res_t.data

if res_tasks_data:
    # 1. Menampilkan tugas sesuai desain awal kamu (menggunakan expander/panah)
    for t in res_tasks_data:
        parts = t['task_name'].split("|")
        title_with_date = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else "Eksekusi segera."
        
        with st.sidebar.expander(f"📌 {title_with_date}"):
            st.write(desc)
            if st.button("Selesaikan", key=f"done_{t['id']}", use_container_width=True):
                supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", t['id']).execute()
                st.rerun()
    
    # 2. FITUR BARU: Tombol Hapus Semua di bagian bawah daftar tugas
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Hapus Semua Tugas", type="primary", use_container_width=True):
        # Perintah ini menghapus SEMUA tugas yang berstatus Pending milik user tersebut
        supabase.table("pending_tasks").delete().eq("user_id", user_nickname).eq("status", "Pending").execute()
        st.rerun()

else:
    st.sidebar.info("Tidak ada tugas aktif.")

# ==========================================
# 5. AUDIT PAGE
# ==========================================
page = st.tabs(["🔍 Sesi Audit", "📊 Dashboard"])

with page[0]:
    st.warning("### ⚠️ PANDUAN OPERASIONAL STRATEGIC COPILOT")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. INPUT & MEMORY BRIDGE**
        * Masukkan tantangan teknis atau rencana strategismu secara mendalam.
        * **Penting:** Jika ini lanjutan, unggah **PDF Laporan Terakhir** agar AI ingat konteksnya.
        """)
    with col2:
        st.markdown("""
        **2. INTEROGASI & VALIDASI**
        * Jawab **4 tahap pertanyaan** investigasi untuk membongkar *blind spot*.
        * Gunakan **Upload Foto** di setiap tahap untuk melampirkan bukti lapangan.
        """)
    with col3:
        st.markdown("""
        **3. OUTPUT & EKSEKUSI**
        * Dapatkan **Skor Performa** dan **Blueprint Solusi**.
        * Cek **Sidebar (Action Items)**: Tugas berlabel tanggal akan otomatis muncul di sana.
        """)
    st.markdown("---")

    if st.session_state.audit_stage == 'input':
        u_in = st.text_area("Apa tantangan strategis/teknismu hari ini?", height=150)
        u_f = st.file_uploader("Upload Foto/PDF Laporan Lama (Memory Bridge)", accept_multiple_files=True)
        if st.button("Mulai Sesi"):
            if len(u_in) > 10:
                st.session_state.start_time = time.time() # Mulai catat waktu
                st.session_state.initial_evidence = process_vision(u_f)
                st.session_state.initial_tasks = u_in
                st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1; st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"Fase Interogasi {st.session_state.q_index}/4")
        hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
        
        task_q = Task(
            description=f"Masalah: {st.session_state.initial_tasks}. Memori Dokumen: {st.session_state.initial_evidence}. History: {hist}. Tahap: {st.session_state.q_index}/4.",
            agent=consultant, 
            expected_output="Respon percakapan satu arah yang berisi validasi jawaban sebelumnya dan maksimal 1-2 pertanyaan lanjutan yang tajam (Tanpa daftar angka/list)."
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
            
            # TAMBALAN: Instruksi Task diperketat agar laporan rapi (JEDA 2 BARIS)
            task_fin = Task(
                description=f"History: {full_hist}.", 
                agent=architect,
                expected_output="Laporan strategi utuh dengan JEDA 2 BARIS antar section. Isi: 1. Analisa Konklusi (per paragraf), 2. Roadmap Strategis, dan 3. Action Items (Wajib baris baru per tugas)."
            )
            res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw); st.markdown(res)
            
            score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
            f_score = float(score_match.group(1)) if score_match else 0.0
            
            if not st.session_state.data_saved:
                supabase.table("audit_log").insert({"user_id": user_nickname, "score": f_score, "audit_report": res, "input_preview": st.session_state.initial_tasks[:100]}).execute()
                
                action_section = re.search(r"### ACTION_ITEMS\s*(.*?)(?:\n###|$)", res, re.DOTALL | re.IGNORECASE)
                if action_section:
                    date_str = datetime.now().strftime("%d %b")
                    
                    # TAMBALAN: Regex baru agar mendukung deskripsi panjang/multi-baris
                    tasks = re.findall(r"\*\*(.+?)\*\*[:\-]\s*(.+?)(?=\n\n|\n\*\*|$)", action_section.group(1), re.DOTALL)
                    
                    for title, desc in tasks:
                        supabase.table("pending_tasks").insert({
                            "user_id": user_nickname, 
                            "task_name": f"[{date_str}] {title.strip()} | {desc.strip()}", 
                            "status": "Pending"
                        }).execute()
                st.session_state.data_saved = True; st.rerun()

            st.download_button("📥 Download PDF Laporan", data=generate_pdf(user_nickname, res, f_score), file_name=f"Audit_{user_nickname}.pdf")
            
            st.divider()
            st.subheader("📊 Evaluasi Sistem (Beta Test)")
            with st.form("evaluation_form"):
                col_a, col_b = st.columns(2)
                with col_a:
                    acc = st.select_slider("Akurasi Diagnosa AI (1-5):", options=[1, 2, 3, 4, 5], value=4)
                    clr = st.select_slider("Kejelasan Instruksi (1-5):", options=[1, 2, 3, 4, 5], value=4)
                with col_b:
                    readi = st.radio("Kesiapan Eksekusi:", [1, 2, 3, 4, 5], help="1: Bingung, 5: Sangat Paham")
                
                crit = st.text_area("Masukan/Kritik:", placeholder="Apa yang perlu diperbaiki?")
                submit_eval = st.form_submit_button("Kirim Evaluasi & Reset")

                if submit_eval:
                    dur = (time.time() - st.session_state.start_time) / 60 if st.session_state.start_time else 0
                    words = len(str(st.session_state.chat_history).split())
                    save_to_analytics(user_nickname, dur, acc, clr, readi, crit, words)
                    st.success("Analitik terkirim!")
                    st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved, st.session_state.start_time = 'input', [], False, None
                    st.rerun()

            if st.button("Reset Tanpa Kirim Evaluasi"):
                st.session_state.audit_stage, st.session_state.chat_history, st.session_state.data_saved, st.session_state.start_time = 'input', [], False, None
                st.rerun()
with page[1]:
    st.title("📈 Dashboard")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data)
        st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
        st.dataframe(df.sort_values(by='created_at', ascending=False), use_container_width=True)