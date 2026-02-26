import os
import warnings
import logging
import re
import io
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
        res_audit = supabase.table("audit_log").select("score, input_preview, created_at").eq("user_id", nickname).order("created_at", desc=True).limit(3).execute()
        res_tasks = supabase.table("pending_tasks").select("status").eq("user_id", nickname).execute()
        
        total_tasks = len(res_tasks.data)
        completed = len([t for t in res_tasks.data if t['status'] == 'Completed'])
        pending = total_tasks - completed
        rate = (completed / total_tasks * 100) if total_tasks > 0 else 0
        
        history_summary = "\n".join([f"- {d['created_at'][:10]} (Skor: {d['score']}): {d['input_preview']}..." for d in res_audit.data])
        return f"""
        HISTORI AUDIT TERAKHIR:
        {history_summary if res_audit.data else "Belum ada histori."}
        
        STATISTIK EKSEKUSI NYATA:
        - Total Tugas Diberikan: {total_tasks}
        - Tugas Selesai: {completed}
        - Tugas Mangkrak: {pending}
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
        pdf.cell(0, 10, txt="OPERATIONAL CHECKLIST (PENDING):", ln=True)
        pdf.set_font("Arial", size=10)
        for i, t in enumerate(tasks, 1):
            name = t['task_name'].split("|")[0] if "|" in t['task_name'] else t['task_name']
            task_txt = f"{i}. [ ] {clean_text(name)}"
            pdf.multi_cell(0, 5, txt=task_txt)
            
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
        "user_id": nickname,
        "score": score,
        "audit_report": audit_result,
        "input_preview": user_input[:100]
    }).execute()
    st.session_state.data_saved = True

def extract_and_save_tasks(audit_result, nickname):
    """Logika Ekstraksi Positif: Hanya mengambil baris berformat **Judul**: Deskripsi."""
    match = re.search(r"### ACTION_ITEMS\s*(.*?)(?:\n###|$)", audit_result, re.DOTALL | re.IGNORECASE)
    if not match: return
    
    content = match.group(1)
    valid_tasks = re.findall(r"\*\*(.+?)\*\*[:\-]\s*(.+)", content)
    
    saved_count = 0
    for title, desc in valid_tasks:
        if saved_count >= 6: break
        clean_title = title.strip()
        clean_desc = re.sub(r"\(?deadline.*", "", desc, flags=re.IGNORECASE).strip()
        clean_desc = clean_desc.rstrip("(").rstrip("[").strip()

        if len(clean_title) > 5 and len(clean_desc) > 10:
            full_data = f"{clean_title} | {clean_desc}"
            supabase.table("pending_tasks").insert({
                "user_id": nickname,
                "task_name": full_data,
                "status": "Pending"
            }).execute()
            saved_count += 1

# ==========================================
# 3. AGENT SETUP (V9.2 - STRATEGIC ARCHITECT)
# ==========================================
consultant = Agent(
    role='Expert Tactical Auditor',
    goal='Mendiagnosa bottleneck melalui data variansi dan bukti artefak.',
    backstory="""Kamu auditor teknis yang dingin. Dilarang memberikan apresiasi atau basa-basi. 
    Tugasmu adalah meminta data (foto/estimasi) dan memberikan 'Logika Teknis' di baliknya.
    Gunakan 'saya' dan 'kamu'. Fokus pada fakta bahwa variansi waktu adalah bukti inefisiensi kognitif.""",
    llm=llm_gemini,
    allow_delegation=False
)

architect = Agent(
    role='High-Leverage Solutions Architect',
    goal='Hanya memberikan tugas dengan daya ungkit tinggi dalam format yang kaku.',
    backstory="""Kamu arsitek yang benci inefisiensi. 
    Wajib memberikan output ### ACTION_ITEMS dengan format:
    - **Nama Tugas**: Deskripsi teknis singkat tentang kenapa/bagaimana melakukan tugas ini.
    
    ATURAN KETAT:
    1. Gunakan format **Judul**: Deskripsi untuk setiap poin tugas.
    2. Dilarang menggunakan simbol bullet pada header kategori (HABIT_SYSTEMS atau PROJECT_TASKS).
    3. Jangan berikan tugas sampah administratif. Fokus pada 20% tindakan untuk 80% hasil.
    4. HABIT_SYSTEMS: Tanpa deadline. 
    5. PROJECT_TASKS: Wajib deadline 2026 di akhir baris laporan (bukan di dalam baris checklist).""",
    llm=llm_gemini,
    allow_delegation=False
)

# ==========================================
# 4. TAMPILAN WEB & OTENTIKASI SISTEM
# ==========================================
st.set_page_config(page_title="Strategic Auditor V9.3", layout="wide")

def manage_access(name, password):
    # Mode Guest dihapus secara total dari logika akses
    if not name or name.strip() == "":
        st.sidebar.warning("Silakan masukkan Nickname.")
        return False
        
    if not password: 
        st.sidebar.warning("Masukkan Password.")
        return False
        
    try:
        res = supabase.table("user_access").select("*").eq("username", name).execute()
        
        if not res.data:
            st.sidebar.info(f"Nickname '{name}' baru. Daftarkan?")
            if st.sidebar.button("Daftarkan Akun & Kunci Data"):
                supabase.table("user_access").insert({"username": name, "password": password}).execute()
                st.sidebar.success("Akun Dibuat! Silakan klik login.")
                st.rerun()
            return False
        else:
            if res.data[0]['password'] == password:
                return True
            else:
                st.sidebar.error("Password Salah!")
                return False
    except Exception as e:
        st.sidebar.error("⚠️ Sistem Login Belum Siap")
        st.sidebar.info("Gunakan SQL Editor di Supabase untuk membuat tabel 'user_access'.")
        return False

# UI Login di Sidebar
st.sidebar.title("🔐 Secure Access")
# Placeholder menggantikan value default "Guest"
u_name = st.sidebar.text_input("Nickname:", placeholder="Masukkan Nickname kamu...").strip()
u_pass = st.sidebar.text_input("Password:", type="password")

# Proteksi: Jika nickname kosong, aplikasi berhenti di sini
if not u_name:
    st.sidebar.info("👋 Selamat datang. Silakan masukkan identitas untuk mengakses sistem audit.")
    st.stop()

# Reset Sesi jika User Berganti
if st.session_state.current_user != u_name:
    st.session_state.audit_stage = 'input'
    st.session_state.chat_history = []
    st.session_state.data_saved = False
    st.session_state.current_user = u_name

is_authenticated = manage_access(u_name, u_pass)

if is_authenticated:
    user_nickname = u_name
    page = st.sidebar.radio("Navigasi:", ["Audit & Konsultasi", "Dashboard"])

    # --- SIDEBAR CHECKLIST ---
    st.sidebar.markdown("---")
    st.sidebar.markdown(f"### 📋 Checklist: {user_nickname}")

    if st.sidebar.button("🗑️ Hapus Semua Tugas Mangkrak"):
        supabase.table("pending_tasks").delete().eq("user_id", user_nickname).eq("status", "Pending").execute()
        st.sidebar.success("Checklist dibersihkan!")
        st.rerun()

    # Menambahkan .order("created_at", desc=True) agar tugas terbaru muncul paling atas
res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("status", "Pending").order("created_at", desc=True).execute()

    if pending:
        for task in pending:
            title, desc = task['task_name'].split("|") if "|" in task['task_name'] else (task['task_name'], "Eksekusi.")
            st.sidebar.markdown(f"### {title}") 
            st.sidebar.write(desc) 
            if st.sidebar.button(f"Selesaikan", key=f"btn_{task['id']}", use_container_width=True):
                supabase.table("pending_tasks").update({"status": "Completed"}).eq("id", task['id']).execute()
                st.rerun()
            st.sidebar.markdown("---")
    else:
        st.sidebar.success("Tidak ada tugas pending. 🚀")

    # --- LOGIKA HALAMAN UTAMA ---
    if page == "Audit & Konsultasi":
        st.title(f"Lead Auditor AI: {user_nickname}")
        st.markdown("---")

        if st.session_state.audit_stage == 'input':
            st.warning("""
            ### **🛠️ Panduan Operasional Konsultasi**
            1. **Input Tantangan**: Jelaskan kendala teknis atau rencana strategismu.
            2. **Lampirkan Bukti**: Gunakan fitur upload untuk screenshot data.
            3. **Interaksi**: Jawab permintaan data teknis dari AI Auditor.
            """)
            u_in = st.text_area("Apa tantangan teknis atau rencana yang ingin kamu audit?", height=120)
            u_files = st.file_uploader("Upload Bukti Visual/Data (Opsional)", accept_multiple_files=True)
            
            if st.button("Mulai Analisis"):
                if len(u_in) > 15:
                    with st.spinner("Menganalisa data awal..."):
                        st.session_state.initial_evidence = process_images(u_files) if u_files else ""
                        st.session_state.initial_tasks = u_in
                        st.session_state.audit_stage = 'interrogation'
                        st.session_state.q_index = 1
                        st.rerun()

        elif st.session_state.audit_stage == 'interrogation':
            st.subheader(f"🔍 Analisis Auditor ({st.session_state.q_index}/3)")
            with st.spinner("Auditor sedang menyusun interogasi..."):
                hist_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
                task_q = Task(
                    description=f"HARI INI: {datetime.now().strftime('%Y-%m-%d')}. Masalah: {st.session_state.initial_tasks}. History: {hist_str}.",
                    agent=consultant,
                    expected_output="Satu permintaan data teknis spesifik disertai Logika Teknis."
                )
                current_q = str(Crew(agents=[consultant], tasks=[task_q]).kickoff().raw)
            
            st.info(current_q)
            u_ans = st.text_area("Input Jawaban:", key=f"ans_{st.session_state.q_index}")
            u_img = st.file_uploader("Lampirkan Bukti", accept_multiple_files=True, key=f"img_{st.session_state.q_index}")
            
            if st.button("Kirim Data"):
                img_data = f" [Foto: {process_images(u_img)}]" if u_img else ""
                st.session_state.chat_history.append({"q": current_q, "a": u_ans + img_data})
                if st.session_state.q_index < 3:
                    st.session_state.q_index += 1
                else:
                    st.session_state.audit_stage = 'report'
                st.rerun()

        elif st.session_state.audit_stage == 'report':
            with st.spinner("Membangun Blueprint Solusi..."):
                today_str = datetime.now().strftime('%Y-%m-%d')
                past_context = get_user_context(user_nickname)
                full_hist = f"Input: {st.session_state.initial_tasks}\n" + "\n".join([f"Q{i+1}: {h['q']}\nA: {h['a']}" for i, h in enumerate(st.session_state.chat_history)])
                
                task_fin = Task(
                    description=f"""
                    KONTEKS WAKTU: {today_str}. 
                    HISTORI: {past_context}
                    INTERAKSI: {full_hist}
                    
                    Wajib memberikan SKOR_FINAL: [0.0 - 10.0].
                    Format tugas di bawah header '### ACTION_ITEMS' WAJIB menggunakan format **Judul**: Deskripsi.
                    1. HABIT_SYSTEMS (Tanpa deadline).
                    2. PROJECT_TASKS (Wajib deadline 2026).
                    """,
                    agent=architect,
                    expected_output="Laporan blueprint solusi dengan format tugas yang divalidasi sistem."
                )
                res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw)
                st.markdown(res)
                
                save_audit_to_db(st.session_state.initial_tasks, res, user_nickname)
                extract_and_save_tasks(res, user_nickname)
                
                st.markdown("---")
                try:
                    p_tasks = supabase.table("pending_tasks").select("task_name").eq("user_id", user_nickname).eq("status", "Pending").execute().data
                    score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
                    final_score = score_match.group(1) if score_match else "0.0"
                    
                    pdf_bytes = generate_pdf(user_nickname, res, final_score, p_tasks)
                    st.download_button(label="📥 Download PDF Report", data=pdf_bytes, file_name=f"Audit_{user_nickname}.pdf", mime="application/pdf")
                except Exception as e:
                    st.error(f"Gagal menyiapkan PDF: {e}")
                
                if st.button("Selesai & Reset"):
                    st.session_state.audit_stage = 'input'
                    st.session_state.chat_history = []
                    st.session_state.data_saved = False
                    st.rerun()

    elif page == "Dashboard":
        st.title(f"📊 Tracking: {user_nickname}")
        res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).order("created_at", desc=False).execute()
        if res_log.data:
            df = pd.DataFrame(res_log.data)
            df['created_at'] = pd.to_datetime(df['created_at'])
            st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0, 10]), use_container_width=True)
            st.dataframe(df.sort_values(by='created_at', ascending=False))
        else:
            st.info("Belum ada data audit untuk identitas ini.")