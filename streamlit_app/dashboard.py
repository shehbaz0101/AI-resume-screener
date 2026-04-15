"""
NEXUS — AI Resume Screener Dashboard
=====================================
Entry point: streamlit run streamlit_app/dashboard.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import tempfile
import logging

import streamlit as st

from src.logger import setup_logging
from src.parsers.resume_parser import ResumeParser
from src.extractor.contact_extractor import ContactExtractor
from src.extractor.entity_extractor import EntityExtractor
from src.extractor.skill_extractor import SkillExtractor
from src.preprocessing.text_cleaner import TextCleaner
from src.preprocessing.skill_normalizer import SkillNormalizer
from src.embeddings.embedding_model import EmbeddingModel
from src.embeddings.embedding_generator import EmbeddingGenerator
from src.vector_store.chroma_client import ChromaClient
from src.vector_store.resume_index import ResumeIndex
from src.rag.rag_pipeline import RagPipeline
from src.llm.llm_client import LLMClient
from src.exceptions import (
    ParseError,
    EmbeddingError,
    VectorStoreError,
    LLMError,
    EmptyCollectionError,
)

setup_logging()
logger = logging.getLogger(__name__)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NEXUS — AI Resume Screener",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600&display=swap');

:root {
    --neon-cyan:   #00f5ff;
    --neon-pink:   #ff006e;
    --neon-green:  #39ff14;
    --neon-purple: #bf00ff;
    --bg-deep:     #020408;
    --bg-panel:    rgba(0,20,30,0.75);
    --border-glow: rgba(0,245,255,0.25);
    --text-main:   #c8e6f0;
    --text-dim:    #4a7a8a;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: var(--bg-deep) !important;
    font-family: 'Rajdhani', sans-serif !important;
    color: var(--text-main) !important;
}
[data-testid="stAppViewContainer"]::before {
    content: "";
    position: fixed; inset: 0;
    background-image:
        linear-gradient(rgba(0,245,255,0.04) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,245,255,0.04) 1px, transparent 1px);
    background-size: 50px 50px;
    pointer-events: none; z-index: 0;
}
[data-testid="stVerticalBlock"], [data-testid="stHorizontalBlock"], section.main > div {
    position: relative; z-index: 1;
}
#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"], .stDeployButton { display: none !important; }

.top-accent {
    position: fixed; top: 0; left: 0; right: 0; height: 2px;
    background: linear-gradient(90deg, var(--neon-purple), var(--neon-cyan), var(--neon-pink), var(--neon-cyan), var(--neon-purple));
    background-size: 200% 100%;
    animation: slide 4s linear infinite; z-index: 1000;
}
@keyframes slide { 0% { background-position: 0% 50%; } 100% { background-position: 200% 50%; } }

.crt-overlay {
    position: fixed; inset: 0;
    background: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.03) 2px, rgba(0,0,0,0.03) 4px);
    pointer-events: none; z-index: 9999;
}

.nexus-hero { text-align: center; padding: 3rem 1rem 1.5rem; }
.nexus-hero .tagline-top {
    font-family: 'Share Tech Mono', monospace; font-size: 0.75rem;
    letter-spacing: 0.4em; color: var(--neon-cyan); text-transform: uppercase; opacity: 0.8;
}
.nexus-hero h1 {
    font-family: 'Orbitron', monospace !important; font-size: clamp(2.4rem,6vw,4.5rem) !important;
    font-weight: 900 !important; letter-spacing: 0.12em !important; color: #fff !important;
    text-shadow: 0 0 20px var(--neon-cyan), 0 0 60px rgba(0,245,255,0.4);
    margin: 0 !important; line-height: 1.1 !important;
}
.nexus-hero h1 span { color: var(--neon-cyan); }
.nexus-hero .sub {
    font-family: 'Rajdhani', sans-serif; font-size: 1.1rem;
    color: var(--text-dim); letter-spacing: 0.2em; margin-top: 0.6rem; text-transform: uppercase;
}
.nexus-hero .scan-line {
    height: 2px;
    background: linear-gradient(90deg, transparent, var(--neon-cyan), var(--neon-pink), transparent);
    margin: 1.4rem auto 0; width: 60%; box-shadow: 0 0 12px var(--neon-cyan);
}

.cyber-card {
    background: var(--bg-panel); border: 1px solid var(--border-glow);
    border-radius: 4px; padding: 1.6rem; backdrop-filter: blur(12px);
    position: relative; margin-bottom: 1.2rem;
    box-shadow: inset 0 0 30px rgba(0,245,255,0.03), 0 0 40px rgba(0,245,255,0.05);
}
.cyber-card::before, .cyber-card::after {
    content: ""; position: absolute; width: 18px; height: 18px;
    border-color: var(--neon-cyan); border-style: solid;
}
.cyber-card::before { top: -1px; left: -1px; border-width: 2px 0 0 2px; }
.cyber-card::after  { bottom: -1px; right: -1px; border-width: 0 2px 2px 0; }

.cyber-card-label {
    font-family: 'Share Tech Mono', monospace; font-size: 0.65rem;
    letter-spacing: 0.35em; color: var(--neon-cyan); text-transform: uppercase;
    margin-bottom: 0.9rem; opacity: 0.7;
}
.section-header {
    font-family: 'Orbitron', monospace; font-size: 0.95rem; font-weight: 700;
    letter-spacing: 0.2em; color: var(--neon-cyan); text-transform: uppercase;
    margin: 0 0 1rem; display: flex; align-items: center; gap: 0.6rem;
}
.section-header::after { content: ""; flex: 1; height: 1px; background: linear-gradient(90deg, var(--border-glow), transparent); }

[data-testid="stFileUploader"] {
    background: rgba(0,245,255,0.03) !important;
    border: 1px dashed rgba(0,245,255,0.3) !important;
    border-radius: 4px !important; padding: 0.8rem !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--neon-cyan) !important;
    box-shadow: 0 0 20px rgba(0,245,255,0.12) !important;
}
[data-testid="stFileUploader"] label,
[data-testid="stFileUploaderDropzoneInstructions"] {
    color: var(--text-dim) !important; font-family: 'Share Tech Mono', monospace !important; font-size: 0.8rem !important;
}

textarea {
    background: rgba(0,20,30,0.8) !important; border: 1px solid rgba(0,245,255,0.2) !important;
    border-radius: 4px !important; color: var(--text-main) !important;
    font-family: 'Share Tech Mono', monospace !important; font-size: 0.82rem !important;
    caret-color: var(--neon-cyan) !important;
}
textarea:focus { border-color: var(--neon-cyan) !important; box-shadow: 0 0 0 2px rgba(0,245,255,0.1) !important; }

label, .stTextInput label, .stTextArea label {
    font-family: 'Orbitron', monospace !important; font-size: 0.7rem !important;
    letter-spacing: 0.18em !important; color: var(--text-dim) !important; text-transform: uppercase !important;
}

[data-testid="stButton"] button {
    background: transparent !important; border: 1px solid var(--neon-cyan) !important;
    color: var(--neon-cyan) !important; font-family: 'Orbitron', monospace !important;
    font-size: 0.78rem !important; font-weight: 700 !important; letter-spacing: 0.25em !important;
    text-transform: uppercase !important; padding: 0.7rem 2rem !important; border-radius: 2px !important;
    box-shadow: 0 0 15px rgba(0,245,255,0.15) !important; transition: all 0.3s ease !important;
}
[data-testid="stButton"] button:hover {
    background: rgba(0,245,255,0.08) !important;
    box-shadow: 0 0 30px rgba(0,245,255,0.35) !important;
    transform: translateY(-1px) !important;
}

.skill-grid { display: flex; flex-wrap: wrap; gap: 0.5rem; margin-top: 0.5rem; }
.skill-tag {
    font-family: 'Share Tech Mono', monospace; font-size: 0.72rem;
    padding: 0.25rem 0.7rem; border: 1px solid rgba(0,245,255,0.35);
    border-radius: 2px; color: var(--neon-cyan); background: rgba(0,245,255,0.06);
}

.result-box {
    background: rgba(0,245,255,0.03); border: 1px solid rgba(0,245,255,0.18);
    border-radius: 4px; padding: 1.2rem; font-family: 'Share Tech Mono', monospace;
    font-size: 0.82rem; line-height: 1.8; color: var(--text-main); white-space: pre-wrap;
    position: relative;
}
.result-box::before {
    content: "OUTPUT"; font-size: 0.55rem; letter-spacing: 0.4em; color: var(--neon-green);
    position: absolute; top: -0.55rem; left: 1rem; background: var(--bg-deep); padding: 0 0.4rem;
}

.status-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    font-family: 'Share Tech Mono', monospace; font-size: 0.68rem; letter-spacing: 0.12em;
    color: var(--neon-green); border: 1px solid rgba(57,255,20,0.3);
    background: rgba(57,255,20,0.06); padding: 0.2rem 0.6rem; border-radius: 2px;
}
.status-dot {
    width: 6px; height: 6px; background: var(--neon-green); border-radius: 50%;
    box-shadow: 0 0 6px var(--neon-green); animation: pulse 2s infinite;
}
@keyframes pulse { 0%,100% { opacity:1; transform:scale(1); } 50% { opacity:0.5; transform:scale(0.8); } }

.stat-row { display: flex; gap: 1rem; margin-top: 1rem; }
.stat-box {
    flex: 1; border: 1px solid rgba(0,245,255,0.15);
    background: rgba(0,245,255,0.03); padding: 0.8rem 1rem;
    border-radius: 3px; text-align: center;
}
.stat-box .val {
    font-family: 'Orbitron', monospace; font-size: 1.3rem; font-weight: 700;
    color: var(--neon-cyan); text-shadow: 0 0 10px rgba(0,245,255,0.5);
}
.stat-box .lbl {
    font-family: 'Share Tech Mono', monospace; font-size: 0.6rem;
    letter-spacing: 0.25em; color: var(--text-dim); margin-top: 0.2rem;
}

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-deep); }
::-webkit-scrollbar-thumb { background: rgba(0,245,255,0.25); border-radius: 2px; }
hr, [data-testid="stDivider"] {
    border: none !important; height: 1px !important;
    background: linear-gradient(90deg, transparent, rgba(0,245,255,0.2), transparent) !important;
}
</style>

<div class="top-accent"></div>
<div class="crt-overlay"></div>
<div class="nexus-hero">
    <div class="tagline-top">[ neural screening system v3.0.0 ]</div>
    <h1>NEX<span>US</span></h1>
    <div class="sub">AI-Powered Talent Intelligence Engine</div>
    <div class="scan-line"></div>
</div>
""", unsafe_allow_html=True)


# ── Pipeline singleton ────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Initialising pipeline...")
def load_pipeline():
    parser        = ResumeParser()
    contact       = ContactExtractor()
    entity        = EntityExtractor()
    skill_ext     = SkillExtractor()
    cleaner       = TextCleaner()
    normalizer    = SkillNormalizer()
    model         = EmbeddingModel().get_model()
    embedder      = EmbeddingGenerator(model)
    chroma_client = ChromaClient()
    resume_index  = ResumeIndex(chroma_client.get_client())
    llm_client    = LLMClient()
    rag           = RagPipeline(embedder, resume_index.collection, llm_client)
    logger.info("Pipeline loaded successfully.")
    return parser, contact, entity, skill_ext, cleaner, normalizer, embedder, resume_index, rag


try:
    parser, contact, entity, skill_ext, cleaner, normalizer, embedder, resume_index, rag = load_pipeline()
except Exception as e:
    st.error(f"⚠ Pipeline initialisation failed: {e}")
    st.stop()


# ── Stats bar ─────────────────────────────────────────────────────────────────
indexed_count = resume_index.count()
st.markdown(f"""
<div class="stat-row">
    <div class="stat-box"><div class="val">RAG</div><div class="lbl">retrieval mode</div></div>
    <div class="stat-box"><div class="val">ChromaDB</div><div class="lbl">vector store</div></div>
    <div class="stat-box"><div class="val">{indexed_count}</div><div class="lbl">resumes indexed</div></div>
    <div class="stat-box"><div class="val">LightGBM</div><div class="lbl">ranking engine</div></div>
</div>
<br>
""", unsafe_allow_html=True)


# ── Two-column layout ─────────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1], gap="large")


# ── LEFT: Upload ──────────────────────────────────────────────────────────────
with col_left:
    st.markdown("""
    <div class="cyber-card">
        <div class="cyber-card-label">// module_01</div>
        <div class="section-header">⬆ Upload Resume</div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "DROP PDF OR DOCX HERE",
        type=["pdf", "docx"],
        label_visibility="visible",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        with st.spinner("Parsing neural signature..."):
            suffix = ".pdf" if uploaded_file.name.endswith(".pdf") else ".docx"
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(uploaded_file.read())
                    file_path = tmp.name

                raw_text         = parser.parse(file_path)
                clean_text       = cleaner.clean(raw_text)
                name             = entity.extract_name(raw_text)
                email            = contact.extract_email(raw_text)
                skills           = skill_ext.extract_skills(clean_text)
                skills           = normalizer.normalize(skills)
                embedding        = embedder.generate(clean_text)
                resume_id        = name.replace(" ", "_").lower() if name else uploaded_file.name

                resume_index.add_resume(
                    resume_id=resume_id,
                    embedding=embedding.tolist(),
                    metadata={"name": name, "skills": skills, "email": email},
                )

                logger.info("Indexed resume: %s | skills: %s", name, skills)

            except ParseError as e:
                st.error(f"⚠ Parse error: {e}")
                st.stop()
            except EmbeddingError as e:
                st.error(f"⚠ Embedding error: {e}")
                st.stop()
            except VectorStoreError as e:
                st.error(f"⚠ Vector store error: {e}")
                st.stop()
            except Exception as e:
                st.error(f"⚠ Unexpected error: {e}")
                logger.exception("Unexpected error during resume upload")
                st.stop()

        st.markdown(f"""
        <div style="margin-top:1rem;">
            <div class="status-badge">
                <div class="status-dot"></div>
                CANDIDATE INDEXED — {(name or "UNKNOWN").upper()}
            </div>
        </div><br>
        """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="cyber-card" style="margin-top:0.5rem;">
            <div class="cyber-card-label">// extracted_profile</div>
            <div class="section-header">👤 Candidate Profile</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem; color:#4a7a8a; margin-bottom:0.8rem;">
                NAME &nbsp;&nbsp; <span style="color:#c8e6f0;">{name or "—"}</span><br>
                EMAIL &nbsp; <span style="color:#c8e6f0;">{email or "—"}</span>
            </div>
            <div style="font-family:'Orbitron',monospace; font-size:0.65rem; letter-spacing:0.2em; color:#00f5ff; margin-bottom:0.5rem; opacity:0.7;">SKILL STACK</div>
        """, unsafe_allow_html=True)

        if skills:
            tags = '<div class="skill-grid">' + \
                "".join(f'<div class="skill-tag">{s}</div>' for s in skills) + \
                "</div>"
            st.markdown(tags, unsafe_allow_html=True)
        else:
            st.markdown(
                '<p style="font-family:\'Share Tech Mono\',monospace; font-size:0.75rem; color:#4a7a8a;">NO SKILLS DETECTED</p>',
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)


# ── RIGHT: Query ──────────────────────────────────────────────────────────────
with col_right:
    st.markdown("""
    <div class="cyber-card">
        <div class="cyber-card-label">// module_02</div>
        <div class="section-header">🎯 Match Candidates</div>
    """, unsafe_allow_html=True)

    job_description = st.text_area(
        "PASTE JOB DESCRIPTION",
        height=180,
        placeholder="e.g. Senior ML Engineer, 5+ years, PyTorch, LLM fine-tuning, distributed training...",
    )
    st.markdown("</div>", unsafe_allow_html=True)

    btn_col, _ = st.columns([1, 2])
    with btn_col:
        run = st.button("⚡  EXECUTE SCAN")

    if run:
        if not job_description.strip():
            st.warning("⚠ No job description detected. Feed the system.")
        else:
            with st.spinner("Querying vector manifold..."):
                try:
                    rag_result = rag.run(job_description)
                except LLMError as e:
                    st.error(f"⚠ LLM error: {e}")
                    st.stop()
                except Exception as e:
                    st.error(f"⚠ Unexpected error: {e}")
                    logger.exception("Unexpected error during RAG query")
                    st.stop()

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="cyber-card">
                <div class="cyber-card-label">// rag_output · {rag_result.candidates_retrieved} candidates retrieved</div>
                <div class="section-header">🏆 Top Candidates</div>
                <div class="result-box">{rag_result.llm_response}</div>
            </div>
            """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""<br><br>
<div style="text-align:center; padding:1rem 0 2rem;">
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.6rem; letter-spacing:0.35em; color:#1a3a4a; text-transform:uppercase;">
        NEXUS v3.0.0 &nbsp;·&nbsp; RAG-POWERED &nbsp;·&nbsp; VECTOR-NATIVE &nbsp;·&nbsp; LIGHTGBM RANKING
    </div>
</div>
""", unsafe_allow_html=True)
