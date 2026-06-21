# app.py

import os
import datetime
import streamlit as st
from pdf_reader import extract_text_from_pdf, get_pdf_info
from rag_pipeline import process_pdf, create_vectorstore, get_qa_chain, ask

st.set_page_config(
    page_title="DocBot AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ── All CSS ───────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [data-testid="stAppViewContainer"] {
    background: #EAF4FF !important;
    font-family: 'Inter', sans-serif !important;
    color: #1A2B4A !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* Fix ALL text to dark blue */
p, span, div, label, li, h1, h2, h3, h4, h5, h6 {
    color: #1A2B4A;
    font-family: 'Inter', sans-serif;
}

/* PDF text specifically */
[data-testid="stText"] {
    color: #1A2B4A !important;
    font-size: 14px !important;
    line-height: 1.9 !important;
    white-space: pre-wrap !important;
}

/* Chat message text */
[data-testid="stMarkdownContainer"] p {
    color: #1A2B4A !important;
}

[data-testid="stChatMessage"] {
    background: white !important;
    border: 1px solid #DDE8F8 !important;
    border-radius: 16px !important;
    margin-bottom: 10px !important;
    color: #1A2B4A !important;
}

/* Buttons */
[data-testid="stButton"] button[kind="primary"] {
    background: #2B7FFF !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stButton"] button[kind="primary"]:hover {
    background: #1A6FEF !important;
    transform: translateY(-1px) !important;
}

[data-testid="stButton"] button[kind="secondary"] {
    background: white !important;
    color: #2B7FFF !important;
    border: 1.5px solid #2B7FFF !important;
    border-radius: 12px !important;
    font-family: 'Inter', sans-serif !important;
}

/* Input fields */
[data-testid="stTextInput"] input {
    background: white !important;
    border: 1.5px solid #C5D8F5 !important;
    border-radius: 12px !important;
    color: #1A2B4A !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stTextInput"] input:focus {
    border-color: #2B7FFF !important;
    box-shadow: 0 0 0 3px rgba(43,127,255,0.15) !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: white !important;
    border: 2px dashed #C5D8F5 !important;
    border-radius: 16px !important;
}

/* Chat input box */
[data-testid="stChatInput"] {
    background: white !important;
    border: 1.5px solid #C5D8F5 !important;
    border-radius: 16px !important;
}

/* FIX: Chat input typed text — make it WHITE since box renders dark */
[data-testid="stChatInput"] textarea {
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
}

[data-testid="stChatInput"] textarea::placeholder {
    color: rgba(255,255,255,0.5) !important;
}

/* Expander */
[data-testid="stExpander"] {
    background: white !important;
    border: 1px solid #DDE8F8 !important;
    border-radius: 12px !important;
}

/* Info/success boxes */
[data-testid="stAlert"] {
    border-radius: 12px !important;
    color: #1A2B4A !important;
}

/* Divider */
hr {
    border-color: #DDE8F8 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #EAF4FF; }
::-webkit-scrollbar-thumb { background: #2B7FFF; border-radius: 4px; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────
defaults = {
    "step": 1,
    "messages": [],
    "qa_chain": None,
    "pdf_name": "",
    "pdf_pages": 0,
    "pdf_text": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def go(n):
    st.session_state.step = n
    st.rerun()


# ══════════════════════════════════════════════════════════════
# STEP 1 — LANDING
# ══════════════════════════════════════════════════════════════
if st.session_state.step == 1:

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div style='height:80px'></div>", unsafe_allow_html=True)

        st.markdown("""
            <div style='text-align:center; margin-bottom:28px;'>
                <div style='
                    width:96px; height:96px;
                    background:#2B7FFF;
                    border-radius:24px;
                    display:inline-flex;
                    align-items:center;
                    justify-content:center;
                    font-size:46px;
                    box-shadow:0 8px 32px rgba(43,127,255,0.3);
                '>📄</div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <h1 style='
                text-align:center;
                font-size:72px;
                font-weight:700;
                letter-spacing:-3px;
                line-height:1;
                margin:0 0 12px 0;
                color:#1A2B4A;
                font-family:Inter,sans-serif;
            '>
                <span style='color:#2B7FFF;'>Doc</span>Bot
            </h1>
            <p style='
                text-align:center;
                font-size:17px;
                color:#6B8CAD;
                font-weight:400;
                margin:0 0 48px 0;
                font-family:Inter,sans-serif;
            '>
                Your AI-powered document assistant
            </p>
        """, unsafe_allow_html=True)

        if st.button("Enter →", type="primary", use_container_width=True):
            go(2)


# ══════════════════════════════════════════════════════════════
# STEP 2 — API KEY
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 2:

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

        st.markdown("""
            <div style='display:flex; gap:6px; margin-bottom:28px;'>
                <div style='height:4px; width:48px;
                            background:#2B7FFF; border-radius:2px;'></div>
                <div style='height:4px; width:48px;
                            background:#DDE8F8; border-radius:2px;'></div>
            </div>
            <p style='font-size:12px; color:#6B8CAD; letter-spacing:1.5px;
                       font-weight:600; margin:0 0 8px 0;'>STEP 1 OF 2</p>
            <h2 style='font-size:32px; font-weight:700; color:#1A2B4A;
                        margin:0 0 8px 0; font-family:Inter,sans-serif;'>
                Paste your API key
            </h2>
            <p style='color:#6B8CAD; font-size:15px; margin:0 0 24px 0;'>
                Get a free key from Google AI Studio
            </p>
        """, unsafe_allow_html=True)

        env_key = os.getenv("GOOGLE_API_KEY")

        if env_key:
            st.success("✓ API key already loaded from .env file")
            if st.button("Continue →", type="primary", use_container_width=True):
                go(3)
        else:
            api_input = st.text_input(
                "API Key",
                type="password",
                placeholder="AIza••••••••••••••••••••••••••••••",
                label_visibility="collapsed"
            )
            st.caption("Get your free key → https://aistudio.google.com/apikey")
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

            if st.button("Continue →", type="primary", use_container_width=True):
                if api_input and len(api_input) > 10:
                    os.environ["GOOGLE_API_KEY"] = api_input
                    go(3)
                else:
                    st.error("Please enter a valid API key.")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            go(1)


# ══════════════════════════════════════════════════════════════
# STEP 3 — UPLOAD PDF
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 3:

    col1, col2, col3 = st.columns([1, 1.2, 1])
    with col2:
        st.markdown("<div style='height:60px'></div>", unsafe_allow_html=True)

        st.markdown("""
            <div style='display:flex; gap:6px; margin-bottom:28px;'>
                <div style='height:4px; width:48px;
                            background:#2B7FFF; border-radius:2px;'></div>
                <div style='height:4px; width:48px;
                            background:#2B7FFF; border-radius:2px;'></div>
            </div>
            <p style='font-size:12px; color:#6B8CAD; letter-spacing:1.5px;
                       font-weight:600; margin:0 0 8px 0;'>STEP 2 OF 2</p>
            <h2 style='font-size:32px; font-weight:700; color:#1A2B4A;
                        margin:0 0 8px 0; font-family:Inter,sans-serif;'>
                Upload your PDF
            </h2>
            <p style='color:#6B8CAD; font-size:15px; margin:0 0 24px 0;'>
                Notes, books, research papers — any text PDF
            </p>
        """, unsafe_allow_html=True)

        uploaded_file = st.file_uploader(
            "Upload PDF",
            type="pdf",
            label_visibility="collapsed"
        )

        if uploaded_file:
            info = get_pdf_info(uploaded_file)

            st.markdown(f"""
                <div style='
                    background:#EAF4FF;
                    border:1.5px solid #C5D8F5;
                    border-radius:14px;
                    padding:14px 18px;
                    margin:12px 0;
                '>
                    <p style='color:#2B7FFF; margin:0 0 4px 0;
                               font-size:14px; font-weight:600;'>
                        📄 {uploaded_file.name}
                    </p>
                    <p style='color:#6B8CAD; margin:0; font-size:12px;'>
                        {info['pages']} pages &nbsp;·&nbsp;
                        {round(uploaded_file.size/1024, 1)} KB
                    </p>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("👁 Preview extracted text"):
                preview = extract_text_from_pdf(uploaded_file)
                if preview and len(preview.strip()) > 50:
                    st.markdown(f"""
                        <p style='color:#1A2B4A; font-size:13px;
                                   line-height:1.8; white-space:pre-wrap;'>
                            {preview[:800]}...
                        </p>
                    """, unsafe_allow_html=True)
                else:
                    st.warning("No text found — may be a scanned/image PDF.")

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)

            if st.button("⚡ Process & Open Chat", type="primary", use_container_width=True):
                raw_text = extract_text_from_pdf(uploaded_file)
                if not raw_text or len(raw_text.strip()) < 50:
                    st.error("❌ Cannot read this PDF. Use a text-based PDF.")
                else:
                    with st.spinner("Building knowledge base..."):
                        try:
                            chunks = process_pdf(raw_text)
                            vectorstore = create_vectorstore(chunks)
                            st.session_state.qa_chain = get_qa_chain(vectorstore)
                            st.session_state.pdf_name = uploaded_file.name
                            st.session_state.pdf_pages = info['pages']
                            st.session_state.pdf_text = raw_text
                            st.session_state.messages = []
                            go(4)
                        except Exception as e:
                            if "429" in str(e) or "EXHAUSTED" in str(e):
                                st.error("⚠️ Quota exhausted. Go back and enter a new API key.")
                            else:
                                st.error(f"Error: {str(e)}")
        else:
            st.caption("Make sure text is selectable in your PDF before uploading.")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        if st.button("← Back", use_container_width=True):
            go(2)


# ══════════════════════════════════════════════════════════════
# STEP 4 — SPLIT LAYOUT (PDF left | Chat right)
# ══════════════════════════════════════════════════════════════
elif st.session_state.step == 4:

    # ── Top bar ───────────────────────────────────────────────
    t1, t2, t3 = st.columns([1, 3, 1])
    with t1:
        if st.button("← New PDF"):
            go(3)
    with t2:
        q = len([m for m in st.session_state.messages if m["role"] == "user"])
        st.markdown(f"""
            <div style='text-align:center; padding:6px 0;'>
                <span style='
                    background:#EAF4FF;
                    border:1.5px solid #C5D8F5;
                    border-radius:20px;
                    padding:6px 18px;
                    font-size:13px;
                    color:#2B7FFF;
                    font-weight:600;
                '>
                    📄 {st.session_state.pdf_name}
                    &nbsp;·&nbsp;
                    <span style='color:#6B8CAD; font-weight:400;'>
                        {st.session_state.pdf_pages} pages
                        &nbsp;·&nbsp;
                        {q} question{"s" if q != 1 else ""} asked
                    </span>
                </span>
            </div>
        """, unsafe_allow_html=True)
    with t3:
        if st.button("Change Key 🔑"):
            go(2)

    st.divider()

    # ── Split columns ─────────────────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    # ── LEFT: PDF viewer ──────────────────────────────────────
    with left:
        st.markdown(f"""
            <div style='
                background:white;
                border:1.5px solid #DDE8F8;
                border-radius:16px 16px 0 0;
                padding:14px 20px;
                display:flex;
                align-items:center;
                gap:10px;
            '>
                <span style='font-size:15px;'>📄</span>
                <span style='font-size:14px; font-weight:600; color:#1A2B4A;'>
                    {st.session_state.pdf_name}
                </span>
                <span style='
                    background:#EAF4FF;
                    border:1px solid #C5D8F5;
                    color:#2B7FFF;
                    font-size:11px;
                    font-weight:600;
                    padding:2px 10px;
                    border-radius:20px;
                '>{st.session_state.pdf_pages} pages</span>
            </div>
        """, unsafe_allow_html=True)

        with st.container(height=620):
            pdf_text = st.session_state.pdf_text

            if pdf_text:
                pages = pdf_text.split("--- Page ")
                for i, chunk in enumerate(pages):
                    if not chunk.strip():
                        continue
                    if i == 0:
                        st.markdown(f"""
                            <p style='color:#1A2B4A; font-size:13.5px;
                                       line-height:1.85; white-space:pre-wrap;
                                       margin:0 0 16px 0;'>
                                {chunk.strip()}
                            </p>
                        """, unsafe_allow_html=True)
                    else:
                        lines = chunk.split(" ---\n", 1)
                        page_num = lines[0].strip()
                        content = lines[1].strip() if len(lines) > 1 else chunk.strip()

                        st.markdown(f"""
                            <div style='
                                font-size:11px;
                                font-weight:700;
                                color:#2B7FFF;
                                letter-spacing:1.5px;
                                margin:20px 0 8px 0;
                                text-transform:uppercase;
                            '>— Page {page_num} —</div>
                            <p style='
                                color:#1A2B4A;
                                font-size:13.5px;
                                line-height:1.85;
                                white-space:pre-wrap;
                                margin:0 0 12px 0;
                            '>{content}</p>
                            <hr style='border:none; border-top:1px solid #EAF4FF;
                                       margin:16px 0;'>
                        """, unsafe_allow_html=True)
            else:
                st.info("No text content found.")

    # ── RIGHT: Chat ───────────────────────────────────────────
    with right:
        st.markdown("""
            <div style='
                background:white;
                border:1.5px solid #DDE8F8;
                border-radius:16px 16px 0 0;
                padding:14px 20px;
            '>
                <span style='font-size:14px; font-weight:600; color:#2B7FFF;'>
                    💬 Ask anything about your document
                </span>
            </div>
        """, unsafe_allow_html=True)

        if not st.session_state.messages:
            st.markdown("""
                <div style='text-align:center; padding:48px 0 24px 0;'>
                    <div style='font-size:40px; margin-bottom:14px;'>🤖</div>
                    <p style='color:#6B8CAD; font-size:15px;
                               font-weight:500; margin:0 0 6px 0;'>
                        Ready to answer your questions
                    </p>
                    <p style='color:#A0B8D0; font-size:13px; margin:0;'>
                        Try: "Summarize this" · "What are the main topics?"
                        · "Explain page 1"
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Messages
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                color = "white" if msg["role"] == "user" else "#1A2B4A"
                st.markdown(f"""
                    <p style='color:{color}; font-size:14px;
                               line-height:1.7; margin:0;'>
                        {msg["content"]}
                    </p>
                """, unsafe_allow_html=True)

        # Chat input
        question = st.chat_input("Ask about your document...")

        if question:
            st.session_state.messages.append({
                "role": "user",
                "content": question
            })
            with st.chat_message("user"):
                st.markdown(f"""
                    <p style='color:white; font-size:14px;
                               line-height:1.7; margin:0;'>
                        {question}
                    </p>
                """, unsafe_allow_html=True)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        answer, sources = ask(
                            st.session_state.qa_chain, question
                        )
                        st.markdown(f"""
                            <p style='color:#1A2B4A; font-size:14px;
                                       line-height:1.7; margin:0;'>
                                {answer}
                            </p>
                        """, unsafe_allow_html=True)

                        with st.expander("📎 Sources used"):
                            for i, doc in enumerate(sources):
                                st.markdown(f"""
                                    <p style='color:#6B8CAD; font-size:12px;
                                               margin:0 0 8px 0;'>
                                        <b style='color:#2B7FFF;'>
                                            Chunk {i+1}:
                                        </b>
                                        {doc.page_content[:200]}...
                                    </p>
                                """, unsafe_allow_html=True)

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": answer
                        })

                    except Exception as e:
                        if "429" in str(e) or "EXHAUSTED" in str(e):
                            st.error("⚠️ Quota exhausted — enter a new API key")
                            if st.button("← Change API Key"):
                                go(2)
                        else:
                            st.error(f"Error: {str(e)}")