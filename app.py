import os
import warnings
import re
import time 
from datetime import datetime
from PIL import Image
import streamlit as st
import pandas as pd
import plotly.express as px
from crewai import Agent, Task, Crew
from langchain_google_genai import ChatGoogleGenerativeAI
import google.generativeai as genai
from supabase import create_client, Client 
from fpdf import FPDF

# ==========================================
# 0. KONFIGURASI SISTEM
# ==========================================
os.environ["OTEL_SDK_DISABLED"] = "true"
os.environ["CREWAI_TELEMETRY_OUT_OUT"] = "true"
warnings.filterwarnings("ignore", category=FutureWarning)

API_KEY = st.secrets["GOOGLE_API_KEY"]
os.environ["GOOGLE_API_KEY"] = API_KEY.strip()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

SB_URL = st.secrets["SUPABASE_URL"]
SB_KEY = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(SB_URL, SB_KEY)

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3)
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 1. CORE FUNCTIONS & SELECTIVE MEMORY
# ==========================================
def init_state():
    defaults = {
        'audit_stage': 'input', 'q_index': 0, 'chat_history': [], 'ui_chat': [],
        'initial_tasks': "", 'initial_evidence': "", 'data_saved': False, 
        'current_user': None, 'start_time': None, 'active_workspace': 'General',
        'report_cache': "", 'score_cache': 0.0, 'memory_context': "", 'last_ai_q': ""
    }
    for k, v in defaults.items():
        if k not in st.session_state: st.session_state[k] = v

init_state()

def get_task_stats(user_id, category):
    res = supabase.table("pending_tasks").select("status").eq("user_id", user_id).eq("category", category).execute()
    if not res.data: return 0, 0
    total = len(res.data)
    completed = len([t for t in res.data if t['status'] == 'Completed'])
    return total, completed

def save_to_analytics(nickname, duration, accuracy, clarity, readiness, critique, word_count, category):
    try:
        data = {
            "user_nickname": str(nickname), "time_spent": float(duration),
            "ai_accuracy": int(accuracy), "clarity_score": int(clarity),
            "action_readiness": int(readiness), "critique": str(critique),
            "word_count_input": int(word_count), "category": category
        }
        supabase.table("audit_analytics").insert(data).execute()
        return True
    except Exception as e:
        st.error(f"❌ DATABASE REJECTED: {str(e)}")
        return False

def get_user_workspaces(uid):
    res = supabase.table("user_workspaces").select("workspace_name").eq("user_id", uid).execute()
    if not res.data: return ["General"]
    return [d['workspace_name'] for d in res.data]

def get_memory_context(user_id, category):
    try:
        logs = supabase.table("audit_log").select("audit_report").eq("user_id", user_id).eq("category", category).order("created_at", desc=True).limit(2).execute()
        tasks = supabase.table("pending_tasks").select("task_name").eq("user_id", user_id).eq("category", category).eq("status", "Pending").execute()
        mem = f"--- MEMORI WORKSPACE: {category} ---\n"
        if tasks.data:
            mem += "TUGAS PENDING: " + ", ".join([t['task_name'] for t in tasks.data]) + "\n"
        if logs.data:
            mem += "RINGKASAN SEBELUMNYA: " + logs.data[0]['audit_report'][:500] + "..."
        return mem
    except: return "Belum ada histori."

def process_vision(files):
    if not files: return ""
    descriptions = []
    for f in files:
        try:
            img = Image.open(f)
            res = vision_model.generate_content(["Ekstrak data strategis dan bukti fisik dari gambar ini.", img])
            descriptions.append(res.text)
        except: continue
    return " | ".join(descriptions)

def clean_txt(text):
    return text.replace("**", "").replace("###", "").replace("##", "").replace("#", "").replace("*", "-").encode('ascii', 'ignore').decode('ascii')

def generate_pdf(nickname, report_text, score):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 16); pdf.cell(0, 10, txt=f"STRATEGIC AUDIT: {clean_txt(nickname)}", ln=True, align='C')
    pdf.set_font("Arial", size=10); pdf.cell(0, 10, txt=f"Score: {score}/10", ln=True, align='C'); pdf.ln(10)
    pdf.multi_cell(0, 5, txt=clean_txt(report_text))
    return pdf.output(dest='S').encode('latin-1')

# ==========================================
# 2. LOGIN & AUTHENTICATION
# ==========================================
st.set_page_config(page_title="Strategic Copilot V2", layout="wide")

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
                st.success("Terdaftar! Silakan klik masuk."); st.rerun()
            else: st.error("Salah password.")
    st.stop()

user_nickname = st.session_state.current_user

# ==========================================
# 3. SIDEBAR (DYNAMIC WORKSPACE & TASKS)
# ==========================================
st.sidebar.title(f"👤 {user_nickname}")

st.sidebar.markdown("### 📁 Create New Workspace")
new_ws_input = st.sidebar.text_input("Nama Ruang Kerja Baru:")
if st.sidebar.button("Tambah & Pindah", use_container_width=True):
    if new_ws_input:
        try:
            supabase.table("user_workspaces").insert({"user_id": user_nickname, "workspace_name": new_ws_input}).execute()
            st.session_state.active_workspace = new_ws_input
            st.session_state.audit_stage = 'input' 
            st.rerun()
        except: st.sidebar.error("Nama sudah ada.")

available_ws = get_user_workspaces(user_nickname)
selected_ws = st.sidebar.selectbox("Pilih Ruang Kerja Aktif:", available_ws, index=available_ws.index(st.session_state.active_workspace) if st.session_state.active_workspace in available_ws else 0)

if selected_ws != st.session_state.active_workspace:
    st.session_state.active_workspace = selected_ws
    st.session_state.audit_stage = 'input'
    st.session_state.ui_chat = []
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Tasks: {selected_ws}")

res_t = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("category", selected_ws).eq("status", "Pending").order("created_at", desc=True).execute()

if res_t.data:
    for t in res_t.data:
        parts = t['task_name'].split("|")
        with st.sidebar.expander(f"📌 {parts[0].strip()}"):
            st.write(parts[1].strip() if len(parts) > 1 else "Segera kerjakan.")
            if st.button("✅ Selesai", key=f"done_{t['id']}", use_container_width=True):
                supabase.table("pending_tasks").update({"status": "Completed", "completed_at": datetime.now().isoformat()}).eq("id", t['id']).execute()
                st.rerun()
    
    st.sidebar.markdown("---")
    if st.sidebar.button("🗑️ Hapus Semua Tugas di Ruang Ini", type="primary", use_container_width=True):
        supabase.table("pending_tasks").delete().eq("user_id", user_nickname).eq("category", selected_ws).eq("status", "Pending").execute()
        st.rerun()
else:
    st.sidebar.info("Tidak ada tugas tertunda.")

# ==========================================
# 4. CHAT AGENTS SETUP (WISE BUSINESS MENTOR)
# ==========================================
consultant = Agent(
    role='Senior Business Mentor',
    goal='Menganalisis tantangan bisnis dengan teori strategis dan membimbing user secara elegan.',
    backstory=f"""Kamu adalah mentor bisnis senior yang bijaksana di ruang kerja {selected_ws}. 
    Dalam berkomunikasi, gunakan kata ganti 'aku' dan 'kamu' dengan nada yang profesional namun santai.
    
    ATURAN KOMUNIKASI:
    1. JANGAN gunakan basa-basi seperti 'Halo', 'Waduh', 'Wah', atau sapaan robotik lainnya.
    2. Langsung berikan respon yang menunjukkan kedalaman analisis. 
    3. Hubungkan setiap jawaban user dengan teori yang relevan sesuai dengan topik yang dibahas untuk memberikan perspektif baru bagi user.
    4. Setelah memberikan analisis singkat, ajukan 1 pertanyaan strategis yang membimbing. 
    5. Pertanyaanmu harus terasa seperti ajakan diskusi, bukan interogasi. Berikan petunjuk area mana 
       yang perlu digali (misal: 'Mungkin kita bisa bedah dari sisi standarisasi tim atau kualitas vendor?').
    6. WAJIB: Berikan bimbingan eksplisit tentang apa yang harus user balas. 
       Berikan 2-3 poin atau kategori yang kamu butuhkan agar user tidak bingung.
       Contoh: 'Kamu bisa ceritakan dari sisi (A) Detail Kejadian, (B) Dampak Finansial, atau (C) Langkah yang sudah diambil'.
    
    FITUR TAMBAHAN (PROGRESS TRACKING):
    7. Jika 'memory_context' menunjukkan adanya histori tugas sebelumnya, kamu harus menanyakan progres nyata dan bukti (foto/angka). 
       Jangan lanjut ke topik baru sebelum mengevaluasi apakah tugas lama sudah berdampak di dunia nyata.""",
    llm=llm_gemini
)

architect = Agent(
    role='Lead Solutions Architect',
    goal='Menyusun Blueprint Strategis yang komprehensif, mencakup analisis risiko dan langkah konkret.',
    backstory="""Kamu bertugas merangkum diskusi menjadi laporan strategis yang punya otoritas. 
    Struktur laporan wajib terdiri dari:
    1. SKOR_FINAL: (0-10)
    2. RINGKASAN_EKSEKUTIF: (Analisis mendalam terhadap situasi saat ini)
    3. EVALUASI_PROGRESS_LAPANGAN: (Bedah keberhasilan atau kegagalan user dalam mengeksekusi tugas sebelumnya berdasarkan bukti yang diberikan).
    4. POTENSI_RISIKO: (Hal-hal yang harus diwaspadai user ke depannya)
    5. ACTION_ITEMS: (Daftar tugas konkret dengan format: - **Nama Tugas**: Deskripsi strategis).
    
    Gunakan bahasa profesional yang jernih dan tajam.""",
    llm=llm_gemini
)

# ==========================================
# 5. MAIN UI (REAL CHAT MODE)
# ==========================================
page = st.tabs([f"💬 {selected_ws} Chat", "📊 Analytics"])

with page[0]:
    if st.session_state.audit_stage == 'input':
        st.markdown(f"## Ruang Kerja: {selected_ws}")
        
        # Hitung histori laporan
        rep_res = supabase.table("audit_log").select("id").eq("user_id", user_nickname).eq("category", selected_ws).execute()
        report_count = len(rep_res.data) if rep_res.data else 0
        
        u_in = st.text_area("Mulai percakapan... Ceritakan tantangan atau rencanamu hari ini:", height=100)
        
        # Fitur Progress Dunia Nyata (Hanya jika laporan > 0)
        u_prog = ""
        if report_count > 0:
            st.info("💡 Ada histori di ruang ini. Jika ini terkait tugas lama, ceritakan progres & buktinya.")
            u_prog = st.text_area("Update Progress Dunia Nyata (Bukti/Angka):", placeholder="Contoh: Penjualan naik 10% setelah SOP dijalankan...")

        col1, col2 = st.columns([1, 1])
        with col1:
            u_f = st.file_uploader("Lampirkan Bukti Foto/Data (Opsional)", accept_multiple_files=True)
        with col2:
            use_memory = st.checkbox("🧠 Hubungkan dengan memori & tugas", value=(report_count > 0))
        
        if st.button("Kirim Pesan Pertama 🚀", use_container_width=True):
            if len(u_in) > 5:
                with st.spinner("Menganalisis..."):
                    st.session_state.start_time = time.time()
                    st.session_state.initial_evidence = process_vision(u_f)
                    combined_input = f"Input: {u_in}\nProgres Dunia Nyata: {u_prog}"
                    
                    if use_memory:
                        st.session_state.memory_context = get_memory_context(user_nickname, selected_ws)
                    else:
                        st.session_state.memory_context = "Topik baru. Abaikan histori."
                    
                    task_start = Task(
                        description=f"Konteks: {st.session_state.memory_context}. Input: {combined_input}. Bukti: {st.session_state.initial_evidence}. Balas analisis dan tanya 1 hal pembuka.",
                        agent=consultant, expected_output="Respon mentor awal."
                    )
                    reply = str(Crew(agents=[consultant], tasks=[task_start]).kickoff().raw)
                    st.session_state.last_ai_q = reply
                    st.session_state.ui_chat = [{"role": "user", "content": u_in}, {"role": "assistant", "content": reply}]
                    st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1
                    st.rerun()

    elif st.session_state.audit_stage == 'interrogation':
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.ui_chat:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

        if prompt := st.chat_input("Balas di sini..."):
            st.session_state.ui_chat.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"q": st.session_state.last_ai_q, "a": prompt})
            
            if st.session_state.q_index >= 4:
                st.session_state.audit_stage = 'report'
                st.rerun()
            else:
                st.session_state.q_index += 1
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Mengetik..."):
                            hist_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
                            task_next = Task(
                                description=f"Histori: {hist_str}. Berikan analisis dan tanya 1 hal (Tahap {st.session_state.q_index}/4).",
                                agent=consultant, expected_output="Pesan chat pendek."
                            )
                            reply = str(Crew(agents=[consultant], tasks=[task_next]).kickoff().raw)
                            st.markdown(reply); st.session_state.last_ai_q = reply
                            st.session_state.ui_chat.append({"role": "assistant", "content": reply})
                st.rerun()

    elif st.session_state.audit_stage == 'report':
        if not st.session_state.data_saved:
            with st.spinner("Finalisasi Strategi..."):
                full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
                task_fin = Task(
                    description=f"Susun Blueprint dari chat: {full_hist}. Wajib ada SKOR_FINAL dan ACTION_ITEMS.", agent=architect, expected_output="Laporan Strategis."
                )
                res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw)
                score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*([\d.]+)", res, re.IGNORECASE)
                f_score = float(score_match.group(1)) if score_match else 5.0
                supabase.table("audit_log").insert({"user_id": user_nickname, "category": selected_ws, "score": f_score, "audit_report": res}).execute()
                
                tasks_found = re.findall(r"(?:^|\n)(?:\d+\.|\*|\-)\s*\*\*(.+?)\*\*\s*[:\-]?\s*(.+)", res)
                for title, desc in tasks_found:
                    supabase.table("pending_tasks").insert({"user_id": user_nickname, "category": selected_ws, "task_name": f"[{datetime.now().strftime('%d %b')}] {title} | {desc}", "status": "Pending"}).execute()
                
                st.session_state.report_cache, st.session_state.score_cache, st.session_state.data_saved = res, f_score, True
                st.rerun()

        st.markdown(st.session_state.report_cache)
        st.download_button("📥 Download PDF", data=generate_pdf(user_nickname, st.session_state.report_cache, st.session_state.score_cache), file_name=f"Audit_{selected_ws}.pdf", use_container_width=True)
        if st.button("Reset Sesi"):
            for key in ['audit_stage', 'chat_history', 'ui_chat', 'data_saved']: st.session_state[key] = 'input' if key == 'audit_stage' else ([] if 'chat' in key else False)
            st.rerun()

# ==========================================
# 6. DASHBOARD (GATED ANALYTICS)
# ==========================================
with page[1]:
    st.title(f"📊 Dashboard Strategis: {selected_ws}")
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).eq("category", selected_ws).execute()
    
    if res_log.data:
        df = pd.DataFrame(res_log.data)
        df['created_at'] = pd.to_datetime(df['created_at'], utc=True)
        df = df.sort_values('created_at')
        
        # Gated Logic
        data_span = (df['created_at'].max() - df['created_at'].min()).days
        
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1: st.metric("Rerata Skor", f"{df['score'].mean():.1f}/10")
        with m_col2:
            total, done = get_task_stats(user_nickname, selected_ws)
            st.metric("Tugas Selesai", f"{done}/{total}")
        with m_col3: st.metric("Total Sesi", len(df))

        st.divider()
        st.subheader("📈 Analisis Tren Waktu")
        t1, t2, t3 = st.tabs(["Per Sesi", "Mingguan (Locked)", "Bulanan (Locked)"])

        with t1: st.plotly_chart(px.line(df, x='created_at', y='score', markers=True, range_y=[0,10]), use_container_width=True)
        with t2:
            if data_span >= 7:
                df_w = df.set_index('created_at').resample('W').mean(numeric_only=True).reset_index()
                st.plotly_chart(px.bar(df_w, x='created_at', y='score', range_y=[0,10]), use_container_width=True)
                st.write(f"**Evaluasi:** Kamu menyelesaikan **{done}** tugas minggu ini.")
            else: st.warning(f"🔒 Terkunci. Butuh 7 hari data. (Progress: {data_span}/7)")
        with t3:
            if data_span >= 30:
                df_m = df.set_index('created_at').resample('MS').mean(numeric_only=True).reset_index()
                st.plotly_chart(px.bar(df_m, x='created_at', y='score', range_y=[0,10]), use_container_width=True)
            else: st.warning(f"🔒 Terkunci. Butuh 30 hari data. (Progress: {data_span}/30)")
    else:
        st.info("Belum ada data.")