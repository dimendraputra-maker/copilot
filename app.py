import json
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
        # Menarik Master Summary terbaru dari ruang kerja
        res_ws = supabase.table("user_workspaces").select("master_summary").eq("user_id", user_id).eq("workspace_name", category).execute()
        master_summary = res_ws.data[0].get('master_summary', 'Belum ada memori.') if res_ws.data else 'Belum ada memori.'
        
        # Tetap menarik tugas pending agar tidak lupa dievaluasi
        tasks = supabase.table("pending_tasks").select("task_name").eq("user_id", user_id).eq("category", category).eq("status", "Pending").execute()
        
        mem = f"--- MASTER SUMMARY WORKSPACE: {category} ---\n{master_summary}\n\n"
        if tasks.data:
            mem += "TUGAS PENDING: " + ", ".join([t['task_name'] for t in tasks.data]) + "\n"
        return mem
    except Exception as e: 
        return "Belum ada memori."

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

# Jika user berhasil lewat dari st.stop() di atas, namanya disimpan:
user_nickname = st.session_state.current_user

# ==========================================
# 2.5 FITUR TRIAL 7 HARI & ADMIN BYPASS
# ==========================================
from datetime import datetime, timedelta, timezone

# 👑 1. TENTUKAN SIAPA BOSNYA (Ganti dengan Username yang kamu pakai buat login!)
ADMIN_USERNAME = "Anabionn" 

if user_nickname == ADMIN_USERNAME:
    # --- JALUR KHUSUS ADMIN (VIP) ---
    st.sidebar.success(f"👑 VIP Access: Welcome back, {user_nickname}!")
    
else:
    # --- JALUR TESTER BIASA (DENGAN BATAS WAKTU 7 HARI) ---
    user_check = supabase.table("beta_users").select("*").eq("user_nickname", user_nickname).execute()
    
    if not user_check.data:
        # JIKA TESTER BARU: Catat waktu kedatangannya hari ini
        supabase.table("beta_users").insert({"user_nickname": user_nickname}).execute()
        st.sidebar.success("🎉 Selamat datang! Masa Trial 7 Hari Anda dimulai hari ini.")
        
    else:
        # JIKA TESTER LAMA: Cek apakah argonya sudah habis?
        first_login_str = user_check.data[0]['first_login']
        first_login_date = datetime.fromisoformat(first_login_str.replace('Z', '+00:00'))
        waktu_sekarang = datetime.now(timezone.utc)
        
        batas_trial = first_login_date + timedelta(days=7) 

        if waktu_sekarang > batas_trial:
            # 🛑 GEMBOK APLIKASI JIKA WAKTU HABIS
            st.error("⏳ Masa Trial Beta Anda sudah habis!")
            st.warning("Terima kasih telah berpartisipasi! Silakan hubungi Admin untuk mendapatkan akses penuh.")
            
            # Berikan tombol untuk keluar/logout agar mereka tidak stuck
            if st.button("Keluar (Logout)"):
                st.session_state.current_user = None
                st.rerun()
                
            st.stop() # <--- Ini yang memblokir mereka masuk ke menu Workspace
            
        else:
            # Jika waktu masih ada, tampilkan sisa hari di sidebar
            sisa_hari = (batas_trial - waktu_sekarang).days
            st.sidebar.info(f"⏱️ Versi Beta. Sisa masa trial: {sisa_hari} hari.")

# ==========================================
# SIDEBAR: WORKSPACE & ACTION ITEMS
# ==========================================

# 1. PILIH WORKSPACE DULU (Ini harus di atas agar tidak NameError)
available_ws = get_user_workspaces(user_nickname)
selected_ws = st.sidebar.selectbox("Pilih Ruang Kerja Aktif:", available_ws, index=available_ws.index(st.session_state.active_workspace) if st.session_state.active_workspace in available_ws else 0)

# FITUR TAMBAH WORKSPACE BARU
with st.sidebar.expander("➕ Buat Workspace Baru"):
    new_ws_name = st.text_input("Nama Proyek/Bisnis Baru:")
    if st.button("Buat Ruang Kerja", use_container_width=True):
        if new_ws_name:
            if new_ws_name not in available_ws:
                try:
                    # Menyimpan workspace baru ke database
                    supabase.table("user_workspaces").insert({
                        "user_id": user_nickname,
                        "workspace_name": new_ws_name,
                        "master_summary": "" # Memori awal dikosongkan
                    }).execute()
                    st.success(f"Berhasil dibuat!")
                    
                    # Pindah otomatis ke workspace baru & refresh layar
                    st.session_state.active_workspace = new_ws_name
                    st.session_state.audit_stage = 'input'
                    st.session_state.ui_chat = []
                    st.session_state.chat_history = []
                    st.rerun()
                except Exception as e:
                    st.error(f"Gagal: {e}")
            else:
                st.warning("Nama workspace sudah ada!")
        else:
            st.warning("Nama tidak boleh kosong.")

# 2. HANDLE PERUBAHAN WORKSPACE
if selected_ws != st.session_state.active_workspace:
    st.session_state.active_workspace = selected_ws
    st.session_state.audit_stage = 'input'
    st.session_state.ui_chat = []
    st.session_state.chat_history = []
    st.rerun()

# 3. TAMPILKAN TUGAS TERTUNDA
st.sidebar.markdown("---")
st.sidebar.markdown(f"### 📋 Action Items: {selected_ws}")

try:
    # Tarik SEMUA data tugas untuk user dan workspace ini
    res_tasks = supabase.table("pending_tasks").select("*").eq("user_id", user_nickname).eq("category", selected_ws).execute()
    
    # Filter Cerdas (Tangkap 'Pending', 'NULL', atau 'False')
    pending_tasks = []
    if res_tasks.data:
        for t in res_tasks.data:
            if t.get('status') == 'Pending' or t.get('completed') is None or t.get('completed') is False:
                pending_tasks.append(t)
    
    # Render UI jika ada tugas
    if pending_tasks:
        for t in pending_tasks:
            # Format UI lamamu yang memisahkan Judul dan Deskripsi
            parts = t.get('task_name', 'Tugas Tanpa Nama').split("|")
            judul = parts[0].strip()
            deskripsi = parts[1].strip() if len(parts) > 1 else "Segera kerjakan."
            
            with st.sidebar.expander(f"📌 {judul}"):
                st.write(deskripsi)
                
                # Tombol Selesai
                if st.button("✅ Selesai", key=f"done_{t['id']}", use_container_width=True):
                    # Update database agar super rapi (status dan completed diubah barengan)
                    supabase.table("pending_tasks").update({
                        "status": "Completed", 
                        "completed": True,
                        "completed_at": datetime.now().isoformat()
                    }).eq("id", t['id']).execute()
                    st.rerun()
        
        # Tombol Hapus Semua Tugas
        st.sidebar.markdown("---")
        if st.sidebar.button("🗑️ Hapus Semua Tugas di Ruang Ini", type="primary", use_container_width=True):
            # Hapus berdasarkan ID tugas yang tampil saja agar aman
            for task_to_delete in pending_tasks:
                supabase.table("pending_tasks").delete().eq("id", task_to_delete['id']).execute()
            st.rerun()
            
    else:
        st.sidebar.info("🎉 Bersih! Tidak ada tugas tertunda.")

except Exception as e:
    st.sidebar.error(f"Gagal memuat tugas: {e}")

# ==========================================
# 4. CHAT AGENTS SETUP (WISE BUSINESS MENTOR & AUDITOR)
# ==========================================
consultant = Agent(
    role='Senior Business Mentor',
    goal='Menganalisis tantangan yang dimiliki user dengan teori strategis dan membimbing user secara elegan.',
    backstory=f"""Kamu adalah mentor senior yang bijaksana sebagai teman diskusi {selected_ws}. 
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
       Jangan lanjut ke topik baru sebelum mengevaluasi apakah tugas lama sudah berdampak di dunia nyata.
    8. PENANGANAN TANPA BUKTI: Jika user mengklaim tugas selesai tapi TIDAK ADA BUKTI, jangan hukum/marahi, tapi berikan PERINGATAN RISIKO secara natural bahwa asumsi tanpa data bisa membahayakan strategi berikutnya.""",
    llm=llm_gemini
)

# AGEN BARU: The Devil's Advocate (Skeptical Auditor)
auditor = Agent(
    role='Strategic Risk Auditor',
    goal='Mencari titik buta (blind spots) dan asumsi tanpa bukti dari percakapan.',
    backstory="""Kamu adalah 'Devil's Advocate'. Tugasmu bukan memberi solusi, melainkan mencari celah.
    Baca riwayat diskusi dan temukan klaim user yang TIDAK didukung oleh bukti nyata atau angka.
    Buat 'Nota Peringatan Risiko' yang objektif mengenai bahaya dari asumsi tersebut. Tandai tugas yang tak berbukti dengan label [Unverified].""",
    llm=llm_gemini
)

architect = Agent(
    role='Lead Solutions Architect & Strict Scorer',
    goal='Menyusun Blueprint Strategis dalam format JSON yang valid berdasarkan temuan Auditor.',
    backstory="""Kamu adalah arsitek bisnis senior yang merangkum temuan Auditor menjadi laporan final. Kamu SANGAT KRITIS dan tidak mudah terkesan oleh ide tanpa data.
    
    [RUMUS SKORING MUTLAK - WAJIB DIIKUTI]: 
    1. Nilai awal bisnis adalah 10.0.
    2. Kurangi 1.0 poin untuk SETIAP tanda peringatan (⚠️) yang ditemukan oleh Auditor dalam evaluasi.
    3. Kurangi 2.0 poin ekstra JIKA user secara eksplisit menolak riset data, hanya mengandalkan 'feeling', atau tidak memiliki bukti konkret.
    4. Jika total peringatan ⚠️ berjumlah 3 atau lebih, skor MAKSIMAL yang boleh diberikan adalah 5.0.
    5. Jangan pernah memberikan skor di atas 8.0 kecuali semua asumsi didukung data valid.
    
    [FORMAT LAPORAN JSON - WAJIB DIIKUTI]:
    Output HANYA blok JSON yang valid dengan struktur persis seperti ini (gunakan bahasa Indonesia):
    {
      "skor_final": (angka float, hasil dari rumus skoring),
      "ringkasan_eksekutif": "Ringkasan tajam 2-3 kalimat mengenai status proyek dan tingkat risikonya.",
      "evaluasi_progress_lapangan": "Rangkuman evaluasi yang WAJIB menyertakan tanda ⚠️ untuk setiap klaim/asumsi yang tidak ada buktinya.",
      "potensi_risiko": [
        "Risiko fatal 1",
        "Risiko fatal 2"
      ],
      "action_items": [
        {
          "judul": "Nama Tugas|Deskripsi singkat untuk Sidebar",
          "deskripsi": "Penjelasan mendetail mengenai langkah yang harus dilakukan (Minimal 3-4 kalimat instruksi langkah demi langkah)"
        }
      ]
    }
    
    JANGAN ADA teks apa pun sebelum karakter '{' atau sesudah karakter '}'.
    """,
    llm=llm_gemini
)

# AGEN BARU: The Knowledge Archivist (Pengarsip Memori Jangka Panjang)
archivist = Agent(
    role='Chief Knowledge Officer',
    goal='Memperbarui Master Summary (Buku Induk) ruang kerja agar AI tidak melupakan keputusan krusial masa lalu.',
    backstory="""Kamu adalah pengarsip jenius. Tugasmu adalah menggabungkan 'Master Summary Lama' dengan 'Laporan Sesi Hari Ini'.
    Buatlah ringkasan baru yang padat (maksimal 300 kata). 
    Pertahankan keputusan strategis masa lalu, tambahkan perkembangan terbaru, dan buang obrolan basi yang sudah selesai.
    Fokus pada: 1. Core Problem bisnis ini, 2. Strategi jangka panjang yang sudah disepakati, 3. Progres terbaru.""",
    llm=llm_gemini
)

# ==========================================
# 5. MAIN UI (REAL CHAT MODE)
# ==========================================
page = st.tabs([f"💬 {selected_ws} Chat", "📊 Analytics"])

with page[0]:
    # --- TAHAP 1: INPUT AWAL ---
    if st.session_state.audit_stage == 'input':
        st.markdown(f"## Ruang Kerja: {selected_ws}")
        
        rep_res = supabase.table("audit_log").select("id").eq("user_id", user_nickname).eq("category", selected_ws).execute()
        report_count = len(rep_res.data) if rep_res.data else 0
        
        u_in = st.text_area("Mulai percakapan... Ceritakan tantangan atau rencanamu hari ini:", height=100)
        
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

    # --- TAHAP 2: INTERAKTIF CHAT (DENGAN VISION) ---
    elif st.session_state.audit_stage == 'interrogation':
        chat_container = st.container()
        with chat_container:
            for msg in st.session_state.ui_chat:
                with st.chat_message(msg["role"]): st.markdown(msg["content"])

        u_f_interrogation = st.file_uploader("📎 Lampirkan Bukti Gambar (Opsional)", accept_multiple_files=True, key=f"uploader_{st.session_state.q_index}")

        if prompt := st.chat_input("Balas di sini..."):
            
            evidence_text = ""
            if u_f_interrogation:
                with st.spinner("Mengekstrak data dari gambar..."):
                    evidence_text = process_vision(u_f_interrogation)
            
            combined_prompt = prompt
            if evidence_text:
                combined_prompt += f"\n\n[BUKTI VISUAL TERLAMPIR]: {evidence_text}"

            st.session_state.ui_chat.append({"role": "user", "content": prompt})
            st.session_state.chat_history.append({"q": st.session_state.last_ai_q, "a": combined_prompt})
            
            if st.session_state.q_index >= 4:
                st.session_state.audit_stage = 'report'
                st.rerun()
            else:
                st.session_state.q_index += 1
                with chat_container:
                    with st.chat_message("assistant"):
                        with st.spinner("Menganalisis balasan & bukti..."):
                            hist_str = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
                            
                            # 1. Buat instruksi visual yang dinamis (Cerdas mengecek ada foto atau tidak)
                            if evidence_text and evidence_text.strip() != "":
                                vision_instruction = f"Terdapat [BUKTI VISUAL TERLAMPIR]: {evidence_text}. Tolong evaluasi gambar tersebut karena sangat penting."
                            else:
                                vision_instruction = "PERINGATAN KERAS: User TIDAK melampirkan gambar/visual apa pun. JANGAN PERNAH menyinggung, membahas, atau menagih soal foto/desain/gambar/visual di balasanmu."

                            # 2. Masukkan instruksi dinamis itu ke dalam Task
                            task_next = Task(
                                description=f"Histori percakapan: {hist_str}.\n\n{vision_instruction}\n\nBerikan analisis tajam, serang kelemahan asumsi user, dan akhiri dengan bertanya 1 hal spesifik (Tahap {st.session_state.q_index}/4).",
                                agent=consultant,
                                expected_output="Balasan chat singkat, tajam, dan langsung pada intinya."
                            )
                            
                            reply = str(Crew(agents=[consultant], tasks=[task_next]).kickoff().raw)
                            st.markdown(reply); st.session_state.last_ai_q = reply
                            st.session_state.ui_chat.append({"role": "assistant", "content": reply})
                st.rerun()

    # --- TAHAP 3: LAPORAN, JSON PARSER & ROLLING MEMORY ---
    elif st.session_state.audit_stage == 'report':
        if not st.session_state.data_saved:
            with st.spinner("Melakukan Audit & Menyusun Blueprint Strategis..."):
                full_hist = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in st.session_state.chat_history])
                
                task_audit = Task(
                    description=f"Analisa riwayat chat: {full_hist}. Buat nota peringatan untuk klaim yang tidak punya bukti.",
                    agent=auditor, expected_output="Nota peringatan risiko."
                )
                
                task_fin = Task(
                    description=f"Gunakan riwayat chat dan nota Auditor. Susun Blueprint WAJIB format JSON.", 
                    agent=architect, expected_output="Strict JSON format."
                )
                
                res = str(Crew(agents=[auditor, architect], tasks=[task_audit, task_fin]).kickoff().raw)
                
                try:
                    json_match = re.search(r'\{.*\}', res, re.DOTALL)
                    json_str = json_match.group() if json_match else res
                    data = json.loads(json_str)
                    
                    f_score = float(data.get("skor_final", 5.0))
                    
                    md_report = f"### SKOR_FINAL: {f_score}\n\n"
                    md_report += f"**RINGKASAN_EKSEKUTIF:**\n{data.get('ringkasan_eksekutif', '')}\n\n"
                    md_report += f"**EVALUASI_PROGRESS_LAPANGAN:**\n{data.get('evaluasi_progress_lapangan', '')}\n\n"
                    md_report += "**POTENSI_RISIKO:**\n" + "\n".join([f"- {r}" for r in data.get('potensi_risiko', [])]) + "\n\n"
                    md_report += "**ACTION_ITEMS:**\n"
                    
                    # --- SIMPAN TUGAS KE SIDEBAR (PENDING TASKS) ---
                    for task in data.get("action_items", []):
                        t_title = task.get("judul", "Tugas Baru")
                        t_desc = task.get("deskripsi", "Segera lakukan tindakan ini.")
                        
                        # Kita gabungkan judul dan deskripsi agar muncul rapi di Sidebar
                        full_task_text = f"{t_title} | {t_desc}"
                        
                        supabase.table("pending_tasks").insert({
                            "user_id": user_nickname, 
                            "category": selected_ws, # Ini harus sama dengan nama workspace di sidebar
                            "task_name": full_task_text,
                            "status": "Pending",
                            "created_at": datetime.now().isoformat()
                        }).execute()
                    
                    # --- ROLLING MASTER SUMMARY ---
                    old_mem_res = supabase.table("user_workspaces").select("master_summary").eq("user_id", user_nickname).eq("workspace_name", selected_ws).execute()
                    old_summary = old_mem_res.data[0].get('master_summary', '') if old_mem_res.data else ""
                    
                    task_archive = Task(
                        description=f"Memori Lama: {old_summary}\n\nLaporan Baru: {json_str}\n\nGabungkan menjadi 1 Master Summary baru (maks 300 kata).",
                        agent=archivist, expected_output="Teks paragraf Master Summary yang baru."
                    )
                    new_summary = str(Crew(agents=[archivist], tasks=[task_archive]).kickoff().raw)
                    
                    supabase.table("user_workspaces").update({"master_summary": new_summary}).eq("user_id", user_nickname).eq("workspace_name", selected_ws).execute()
                    
                except Exception as e:
                    f_score = 5.0
                    md_report = f"**Sistem Peringatan: Format JSON Gagal. Error: {str(e)}**\n\n{res}"
                    data = {} # Mencegah aplikasi crash jika AI gagal membuat JSON
                
                # --- KODE BARU: MENYIMPAN KE KOLOM TERPISAH UNTUK DASHBOARD CARD VIEW ---
                supabase.table("audit_log").insert({
                    "user_id": user_nickname, 
                    "category": selected_ws, 
                    "score": f_score, 
                    "audit_report": md_report,
                    "ringkasan_eksekutif": data.get('ringkasan_eksekutif', 'Tidak ada ringkasan.'),
                    "evaluasi_progress_lapangan": data.get('evaluasi_progress_lapangan', 'Tidak ada evaluasi.'),
                    "potensi_risiko": json.dumps(data.get('potensi_risiko', []))
                }).execute()
                
                st.session_state.report_cache, st.session_state.score_cache, st.session_state.data_saved = md_report, f_score, True
                st.rerun()

        st.markdown(st.session_state.report_cache)
        st.download_button("📥 Download PDF", data=generate_pdf(user_nickname, st.session_state.report_cache, st.session_state.score_cache), file_name=f"Audit_{selected_ws}.pdf", use_container_width=True)
        
        # ==========================================
        # FITUR KEPUASAN USER (FEEDBACK FORM)
        # ==========================================
        st.markdown("---")
        st.markdown("### ⭐ Seberapa berguna audit ini untukmu?")
        
        with st.form("feedback_form"):
            rating = st.slider("Berikan Nilai (1 = Sangat Buruk, 5 = Sangat Membantu)", min_value=1, max_value=5, value=5)
            komentar = st.text_area("Ada masukan atau keluhan tentang AI Consultant ini? (Opsional)")
            submit_feedback = st.form_submit_button("Kirim Penilaian", type="primary")
            
            if submit_feedback:
                try:
                    supabase.table("user_feedback").insert({
                        "user_id": user_nickname,
                        "workspace": selected_ws,
                        "rating": rating,
                        "comment": komentar,
                        "created_at": datetime.now().isoformat()
                    }).execute()
                    st.success("Terima kasih atas penilaianmu! 🙏 Feedback ini membantu AI menjadi lebih pintar.")
                except Exception as e:
                    st.error(f"Gagal menyimpan feedback: {e}")
        
        # SATU TOMBOL RESET YANG RAPI DI PALING BAWAH
        if st.button("🔄 Mulai Sesi Audit Baru", type="secondary", use_container_width=True):
            for key in ['audit_stage', 'chat_history', 'ui_chat', 'data_saved', 'q_index']: 
                if key == 'audit_stage': st.session_state[key] = 'input'
                elif key == 'q_index': st.session_state[key] = 0
                elif key == 'data_saved': st.session_state[key] = False
                else: st.session_state[key] = []
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
        
        # Urutkan untuk Grafik (Lama ke Baru)
        df_sorted_asc = df.sort_values('created_at') 
        # Urutkan untuk Card UI (Baru ke Lama)
        df_sorted_desc = df.sort_values('created_at', ascending=False)
        
        # Gated Logic
        data_span = (df['created_at'].max() - df['created_at'].min()).days
        
        # --- BAGIAN METRIK (TIDAK DIUBAH) ---
        m_col1, m_col2, m_col3 = st.columns(3)
        with m_col1: st.metric("Rerata Skor", f"{df['score'].mean():.1f}/10")
        with m_col2:
            total, done = get_task_stats(user_nickname, selected_ws)
            st.metric("Tugas Selesai", f"{done}/{total}")
        with m_col3: st.metric("Total Sesi", len(df))

        # --- BAGIAN GRAFIK (TIDAK DIUBAH) ---
        st.divider()
        st.subheader("📈 Analisis Tren Waktu")
        t1, t2, t3 = st.tabs(["Per Sesi", "Mingguan (Locked)", "Bulanan (Locked)"])

        with t1: st.plotly_chart(px.line(df_sorted_asc, x='created_at', y='score', markers=True, range_y=[0,10]), use_container_width=True)
        with t2:
            if data_span >= 7:
                df_w = df_sorted_asc.set_index('created_at').resample('W').mean(numeric_only=True).reset_index()
                st.plotly_chart(px.bar(df_w, x='created_at', y='score', range_y=[0,10]), use_container_width=True)
                st.write(f"**Evaluasi:** Kamu menyelesaikan **{done}** tugas minggu ini.")
            else: st.warning(f"🔒 Terkunci. Butuh 7 hari data. (Progress: {data_span}/7)")
        with t3:
            if data_span >= 30:
                df_m = df_sorted_asc.set_index('created_at').resample('MS').mean(numeric_only=True).reset_index()
                st.plotly_chart(px.bar(df_m, x='created_at', y='score', range_y=[0,10]), use_container_width=True)
            else: st.warning(f"🔒 Terkunci. Butuh 30 hari data. (Progress: {data_span}/30)")
            
        # ==========================================
        # TAMBAHAN: RIWAYAT EVALUASI (CARD UI)
        # ==========================================
        st.divider()
        st.subheader("📝 Riwayat Evaluasi (Card View)")
        
        # Ubah dataframe menjadi bentuk dictionary agar mudah dilooping
        records = df_sorted_desc.to_dict('records')
        
        for rec in records:
            with st.container(border=True):
                col_teks, col_skor = st.columns([3, 1])
                
                with col_teks:
                    tanggal = rec['created_at'].strftime("%d %b %Y, %H:%M")
                    st.markdown(f"#### 🏢 {rec.get('category', selected_ws)} | 📅 {tanggal}")
                    
                    # Membaca kolom ringkasan (ditangani agar aman dari data kosong)
                    ringkasan = rec.get('ringkasan_eksekutif', 'Tidak ada ringkasan yang tersimpan.')
                    if pd.isna(ringkasan): ringkasan = 'Tidak ada ringkasan yang tersimpan.'
                    st.write(f"*{ringkasan}*")
                
                with col_skor:
                    # Menggunakan 'score' sesuai nama kolom database-mu
                    skor = rec.get('score', 0)
                    st.metric(label="Skor Audit", value=f"{skor}/10")
                
                with st.expander("🔍 Lihat Detail Risiko & Temuan"):
                    st.markdown("**⚠️ Potensi Risiko:**")
                    risiko = rec.get('potensi_risiko', [])
                    
                    # Logika cerdas: Jika risiko berbentuk list, tampilkan berjejer
                    if isinstance(risiko, list):
                        for r in risiko:
                            st.markdown(f"- {r}")
                    elif isinstance(risiko, str) and risiko.startswith('['):
                        try:
                            import json
                            for r in json.loads(risiko): st.markdown(f"- {r}")
                        except:
                            st.write(risiko)
                    else:
                        val = "Tidak ada catatan risiko." if pd.isna(risiko) else risiko
                        st.write(val)
                        
                    st.markdown("**📋 Evaluasi Lapangan:**")
                    evaluasi = rec.get('evaluasi_progress_lapangan', 'Tidak ada catatan.')
                    val_eval = "Tidak ada catatan detail lapangan." if pd.isna(evaluasi) else evaluasi
                    st.info(val_eval)

    else:
        st.info("Belum ada data.")