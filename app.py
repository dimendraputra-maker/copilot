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

llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3) # Sedikit dinaikkan agar chat lebih natural
vision_model = genai.GenerativeModel('gemini-2.0-flash')

# ==========================================
# 1. CORE FUNCTIONS & MEMORY SYSTEM
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

def get_memory_context(user_id, category):
    """Menarik memori selektif untuk menghemat token (RAG System)"""
    try:
        # Ambil 2 laporan terakhir di workspace ini
        logs = supabase.table("audit_log").select("audit_report").eq("user_id", user_id).eq("category", category).order("created_at", desc=True).limit(2).execute()
        # Ambil tugas yang belum selesai
        tasks = supabase.table("pending_tasks").select("task_name").eq("user_id", user_id).eq("category", category).eq("status", "Pending").execute()
        
        mem = f"--- HISTORI WORKSPACE [{category.upper()}] ---\n"
        mem += "TUGAS YANG MASIH PENDING (Tanyakan alasannya jika relevan):\n" + "\n".join([f"- {t['task_name']}" for t in tasks.data]) + "\n\n"
        mem += "LAPORAN TERAKHIR (Singkat):\n" + "\n".join([l['audit_report'][:400] + "..." for l in logs.data])
        return mem
    except Exception:
        return "Belum ada histori di workspace ini."

def process_vision(files):
    if not files: return ""
    descriptions = []
    for f in files:
        try:
            img = Image.open(f)
            res = vision_model.generate_content(["Ekstrak poin penting dan data dari gambar ini.", img])
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
# 2. LOGIN & SIDEBAR (WORKSPACE SWITCHER)
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
                st.success("Terdaftar! Klik masuk."); st.rerun()
            else: st.error("Salah password.")
    st.stop()

user_nickname = st.session_state.current_user

# -- SIDEBAR NAVIGATOR --
st.sidebar.title(f"👤 {user_nickname}")
st.sidebar.markdown("### 📂 Ruang Kerja (Workspace)")
workspaces = ["General", "Marketing", "Operations", "Writing"]
selected_ws = st.sidebar.selectbox("Pilih Divisi:", workspaces, index=workspaces.index(st.session_state.active_workspace))

# Reset sesi jika pindah workspace
if selected_ws != st.session_state.active_workspace:
    st.session_state.active_workspace = selected_ws
    st.session_state.audit_stage = 'input'
    st.session_state.ui_chat = []
    st.session_state.chat_history = []
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Pending Tasks: {selected_ws}")

# Load Tasks by Workspace
res_t = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("category", selected_ws).eq("status", "Pending").order("created_at", desc=True).execute()

if res_t.data:
    for t in res_t.data:
        parts = t['task_name'].split("|")
        title_with_date = parts[0].strip()
        desc = parts[1].strip() if len(parts) > 1 else "Eksekusi segera."
        
        with st.sidebar.expander(f"📌 {title_with_date}"):
            st.write(desc)
            if st.button("✅ Selesai", key=f"done_{t['id']}", use_container_width=True):
                # Update status dan waktu selesai untuk perhitungan efisiensi nanti
                supabase.table("pending_tasks").update({"status": "Completed", "completed_at": datetime.now().isoformat()}).eq("id", t['id']).execute()
                st.rerun()
else:
    st.sidebar.info("Bersih! Tidak ada tugas pending.")

# ==========================================
# 3. AI AGENTS (DYNAMIC PERSONA)
# ==========================================
# Menentukan gaya bahasa AI berdasarkan Workspace
personas = {
    "Marketing": "Growth Hacker agresif yang fokus pada konversi, CPA, dan psikologi audiens.",
    "Operations": "Six Sigma Expert yang sangat teliti mengaudit pemborosan waktu dan bottleneck.",
    "Writing": "Creative Editor yang fokus pada daya tarik emosional, struktur narasi, dan hook.",
    "General": "Konsultan Bisnis Senior yang karismatik dan analitis."
}
current_persona = personas.get(selected_ws, "Konsultan Bisnis Senior.")

consultant = Agent(
    role=f'Lead {selected_ws} Copilot',
    goal='Mendiagnosa hambatan dengan gaya percakapan chat interaktif.',
    backstory=f"""Kamu adalah {current_persona}. Gunakan bahasa yang natural, santai tapi profesional. Gunakan 'Saya' dan 'Kamu'.
    
    ATURAN INTERAKSI (WAJIB):
    1. **Format Chat Singkat**: JANGAN berikan laporan panjang. Respons layaknya pesan WhatsApp/Slack.
    2. **Memory Aware**: Gunakan data 'Histori' untuk menantang/menegur user jika mereka mengulangi masalah lama atau menunda tugas.
    3. **Satu Pertanyaan Tajam**: Berikan empati/validasi, lalu ajukan HANYA 1 pertanyaan lanjutan yang membongkar akar masalah.""",
    llm=llm_gemini
)

architect = Agent(
    role='Solutions Architect',
    goal='Menyusun Blueprint Strategis.',
    backstory="""Kamu ahli strategi. Susun laporan rapi.
    WAJIB menyertakan:
    ## 📊 SKOR_FINAL: [Skor 1-10]/10
    ### 📋 ACTION_ITEMS
    Gunakan format ini untuk tiap tugas (Wajib baris baru):
    **Nama Tugas**: Deskripsi teknis.""",
    llm=llm_gemini
)

# ==========================================
# 4. MAIN APP (CHAT INTERFACE)
# ==========================================
page = st.tabs([f"💬 Asisten {selected_ws}", "📊 Dashboard & Refleksi"])

with page[0]:
    # --- STAGE: INPUT ---
    if st.session_state.audit_stage == 'input':
        st.markdown(f"### Selamat datang di Ruang Kerja **{selected_ws}**")
        st.write("Apa tantangan utama yang ingin kita bedah hari ini?")
        
        u_in = st.text_area("Ketik masalahmu di sini...", height=100)
        u_f = st.file_uploader("Lampirkan Data/Laporan (Opsional)", accept_multiple_files=True)
        
        if st.button("Mulai Diskusi 🚀", use_container_width=True):
            if len(u_in) > 10:
                with st.spinner("Memanggil memori dan asisten..."):
                    st.session_state.start_time = time.time()
                    st.session_state.initial_evidence = process_vision(u_f)
                    st.session_state.initial_tasks = u_in
                    
                    # RAG: Tarik Ingatan
                    st.session_state.memory_context = get_memory_context(user_nickname, selected_ws)
                    
                    # Eksekusi Pertanyaan Pertama AI
                    task_q1 = Task(
                        description=f"Konteks Memori: {st.session_state.memory_context}. Masalah User Hari Ini: {st.session_state.initial_tasks}. Berikan respons pembuka yang hangat dan 1 pertanyaan tajam untuk menggali masalah ini.",
                        agent=consultant, expected_output="Pesan chat pembuka."
                    )
                    first_reply = str(Crew(agents=[consultant], tasks=[task_q1]).kickoff().raw)
                    
                    st.session_state.last_ai_q = first_reply
                    st.session_state.ui_chat = [{"role": "assistant", "content": first_reply}]
                    st.session_state.audit_stage, st.session_state.q_index = 'interrogation', 1
                    st.rerun()

    # --- STAGE: CHAT INTERROGATION ---
    elif st.session_state.audit_stage == 'interrogation':
        st.subheader(f"Sesi Interogasi ({st.session_state.q_index}/4)")
        
        # Render riwayat chat di UI
        for msg in st.session_state.ui_chat:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Input Box ala ChatGPT
        if prompt := st.chat_input("Ketik balasanmu..."):
            # 1. Tampilkan di UI
            st.session_state.ui_chat.append({"role": "user", "content": prompt})
            with st.chat_message("user"): st.markdown(prompt)
            
            # 2. Simpan ke sistem memori AI
            st.session_state.chat_history.append({"q": st.session_state.last_ai_q, "a": prompt})
            
            # 3. Cek apakah sudah 4 tahap
            if st.session_state.q_index >= 4:
                st.session_state.audit_stage = 'report'
                st.rerun()
            else:
                st.session_state.q_index += 1
                with st.chat_message("assistant"):
                    with st.spinner("Mengetik..."):
                        hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
                        task_qn = Task(
                            description=f"Memori Masa Lalu: {st.session_state.memory_context}. Percakapan Saat Ini: {hist}. Tanggapi jawaban terakhir '{prompt}' dan berikan 1 pertanyaan lanjutan. Tahap {st.session_state.q_index}/4.",
                            agent=consultant, expected_output="Pesan chat interaktif."
                        )
                        reply = str(Crew(agents=[consultant], tasks=[task_qn]).kickoff().raw)
                        st.markdown(reply)
                        
                        st.session_state.last_ai_q = reply
                        st.session_state.ui_chat.append({"role": "assistant", "content": reply})

    # --- STAGE: REPORT & EXTRACTION ---
    elif st.session_state.audit_stage == 'report':
        if not st.session_state.data_saved:
            with st.spinner("Menganalisa percakapan dan menyusun Blueprint..."):
                full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
                task_fin = Task(
                    description=f"Susun Blueprint dari chat: {full_hist}.", 
                    agent=architect, expected_output="Laporan akhir sesuai format."
                )
                res = str(Crew(agents=[architect], tasks=[task_fin]).kickoff().raw)
                
                score_match = re.search(r"SKOR_FINAL\s*[:=-]?\s*(?:\[)?([\d.]+)(?:\])?", res, re.IGNORECASE)
                f_score = float(score_match.group(1)) if score_match else 0.0
                
                # Simpan Log ke Workspace saat ini
                supabase.table("audit_log").insert({"user_id": user_nickname, "category": selected_ws, "score": f_score, "audit_report": res, "input_preview": st.session_state.initial_tasks[:100]}).execute()
                
                # Regex Task Extraction
                action_search = re.search(r"(?:ACTION_ITEMS|Action Items|📋).*?\n(.*?)($|###|🛡️|Protokol)", res, re.DOTALL | re.IGNORECASE)
                if action_search:
                    task_content = action_search.group(1).strip()
                    task_list = re.findall(r"(?:\n|^)(?:\*|- )?(?:\*\*)?(.+?)(?:\*\*)?[:\-]\s*(.+?)(?=\n(?:\*|- )?(?:\*\*)?[A-Z]|\n\n|$)", task_content, re.DOTALL)
                    date_str = datetime.now().strftime("%d %b")
                    for title, desc in task_list:
                        clean_title = title.replace("**", "").replace("-", "").strip()
                        clean_desc = desc.replace("**", "").strip()
                        if len(clean_title) > 3:
                            supabase.table("pending_tasks").insert({
                                "user_id": user_nickname, "category": selected_ws,
                                "task_name": f"[{date_str}] {clean_title} | {clean_desc}", 
                                "status": "Pending"
                            }).execute()
                
                st.session_state.report_cache = res
                st.session_state.score_cache = f_score
                st.session_state.data_saved = True
                st.rerun()

        st.success("✅ Audit Selesai! Ini Blueprint untuk eksekusi:")
        st.markdown(st.session_state.report_cache)
        
        st.divider()
        st.subheader("📊 Evaluasi Sesi")
        with st.form("evaluation_form"):
            col_a, col_b = st.columns(2)
            with col_a:
                acc = st.select_slider("Akurasi Diagnosa (1-5):", options=[1, 2, 3, 4, 5], value=4)
                clr = st.select_slider("Kejelasan Instruksi (1-5):", options=[1, 2, 3, 4, 5], value=4)
            with col_b:
                readi = st.radio("Kesiapan Eksekusi:", [1, 2, 3, 4, 5])
            crit = st.text_area("Kritik/Masukan:")
            
            if st.form_submit_button("Selesaikan & Reset"):
                dur = (time.time() - st.session_state.start_time) / 60 if st.session_state.start_time else 0
                words = len(str(st.session_state.chat_history).split())
                save_to_analytics(user_nickname, dur, acc, clr, readi, crit, words, selected_ws)
                
                init_state() # Reset semua state
                st.rerun()

# --- TAB: DASHBOARD ---
with page[1]:
    st.title(f"📈 Performa Divisi: {selected_ws}")
    st.write("Catatan: Data ini khusus untuk ruang kerja yang sedang aktif.")
    
    res_log = supabase.table("audit_log").select("*").eq("user_id", user_nickname).eq("category", selected_ws).order("created_at", desc=False).execute()
    if res_log.data:
        df = pd.DataFrame(res_log.data)
        st.plotly_chart(px.line(df, x='created_at', y='score', title=f"Trend Strategi {selected_ws}", markers=True, range_y=[0, 10]), use_container_width=True)
        st.dataframe(df.sort_values(by='created_at', ascending=False), use_container_width=True)
    else:
        st.info("Belum ada data audit di ruang kerja ini.")