# frontend/app.py
import streamlit as st
import requests
import os
import json
import time
import pandas as pd
import plotly.express as px
from io import BytesIO
from datetime import datetime

# Configure the Streamlit page
st.set_page_config(
    page_title="Advanced Document QA System",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
BACKEND_URL = "http://localhost:8001"
DEFAULT_AVATAR = "https://api.dicebear.com/7.x/avataaars/svg?seed=Felix"

# Initialize session state variables
if "active_document" not in st.session_state:
    st.session_state.active_document = None
if "chat_histories" not in st.session_state:
    st.session_state.chat_histories = {}
if "documents" not in st.session_state:
    st.session_state.documents = []
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True
if "llm_settings" not in st.session_state:
    st.session_state.llm_settings = {
        "temperature": 0.1,
        "max_tokens": 300
    }

# Load custom CSS for styling
def load_css():
    css_file = "assets/style.css"
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
    # Dynamic theme styles
    theme_css = """
    <style>
    :root {
        --background-color: #121212;
        --text-color: #E0E0E0;
        --card-bg-color: #1E2A38;
        --card-border-color: #3A4A5E;
        --accent-color: #A9CCE3;
        --hover-color: #D6EAF8;
    }
    
    .light-theme {
        --background-color: #FFFFFF;
        --text-color: #333333;
        --card-bg-color: #F8F9FA;
        --card-border-color: #DEE2E6;
        --accent-color: #007BFF;
        --hover-color: #0056b3;
    }
    </style>
    """
    st.markdown(theme_css, unsafe_allow_html=True)

load_css()

# Toggle theme function
def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode
    
# Backend API calls
def api_health_check():
    """Check if backend is available"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/health", timeout=3)
        return response.status_code == 200
    except:
        return False

def get_document_list():
    """Get list of processed documents from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/documents")
        if response.status_code == 200:
            return response.json().get("documents", [])
        return []
    except Exception as e:
        st.error(f"Failed to get document list: {e}")
        return []

def get_document_details(doc_id):
    """Get details of a specific document"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/documents/{doc_id}")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Failed to get document details: {e}")
        return None

def process_document(uploaded_file=None, text_input=None):
    """Process document through backend API"""
    try:
        backend_url = f"{BACKEND_URL}/api/upload"
        
        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            files = {"file": (uploaded_file.name, BytesIO(file_bytes), uploaded_file.type)}
            with st.spinner("Processing document... (This may take a while)"):
                res = requests.post(backend_url, files=files)
        elif text_input:
            payload = {"text": text_input}
            with st.spinner("Processing text... (This may take a while)"):
                res = requests.post(backend_url, json=payload)
        else:
            return None
            
        if res.status_code == 200:
            return res.json()
        else:
            error_msg = res.json().get("error", "Unknown error")
            st.error(f"Error: {error_msg}")
            return None
    except Exception as e:
        st.error(f"Failed to process document: {e}")
        return None

def ask_question(question, doc_id):
    """Send question to backend API"""
    try:
        payload = {
            "question": question,
            "document_id": doc_id
        }
        
        with st.spinner("Generating answer..."):
            res = requests.post(f"{BACKEND_URL}/api/answer", json=payload)
            
        if res.status_code == 200:
            return res.json()
        else:
            error_msg = res.json().get("error", "Unknown error")
            st.error(f"Error: {error_msg}")
            return None
    except Exception as e:
        st.error(f"Failed to get answer: {e}")
        return None

def get_document_summary(doc_id):
    """Get downloadable summary from backend"""
    try:
        response = requests.get(f"{BACKEND_URL}/api/document/{doc_id}/download")
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        st.error(f"Failed to get summary: {e}")
        return None

# Helper to get/set chat history for current document
def get_chat_history(doc_id):
    return st.session_state.chat_histories.get(doc_id, [])

def set_chat_history(doc_id, history):
    st.session_state.chat_histories[doc_id] = history

# UI Components
def render_header():
    """Render the application header"""
    col1, col2, col3 = st.columns([6, 3, 1])
    
    with col1:
        st.title(" Scientific Document Q&A System")
        st.write(
            "Upload documents or paste text for AI-powered summarization and question answering, "
            "all running locally on your machine."
        )
    
    with col2:
        backend_status = api_health_check()
        status_color = "green" if backend_status else "red"
        status_text = "Backend Online" if backend_status else "Backend Offline"
        st.markdown(f"""
        <div style="text-align: right; padding-top: 10px;">
            <div style="display: inline-block; color: white; background-color: {status_color}; 
                    padding: 5px 15px; border-radius: 15px; font-size: 0.8em;">
                ● {status_text}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        # Theme toggle button
        theme_icon = "🌙" if st.session_state.dark_mode else "☀️"
        if st.button(theme_icon, key="theme_toggle"):
            toggle_theme()

def render_sidebar():
    """Render sidebar content"""
    with st.sidebar:
        st.header("Navigation")
        
        # Document history
        st.subheader("Your Documents")
        if st.button("🔄 Refresh Documents"):
            st.session_state.documents = get_document_list()
        
        if not st.session_state.documents:
            st.info("No documents processed yet.")
        else:
            for doc in st.session_state.documents:
                doc_name = doc.get("filename", "Unnamed Document")
                if st.button(f"📄 {doc_name}", key=f"doc_{doc['id']}"):
                    st.session_state.active_document = doc['id']
                    doc_details = get_document_details(doc['id'])
                    if doc_details:
                        st.session_state.active_document_details = doc_details
                    # Load chat history for this document
                    st.session_state.chat_history = get_chat_history(doc['id'])
                    st.rerun()
        
        # Divider
        st.markdown("---")
        
        # LLM Settings
        st.subheader("⚙️ Model Settings")
        temperature = st.slider(
            "Temperature",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.llm_settings["temperature"],
            step=0.1,
            help="Higher values make output more random, lower values more deterministic"
        )
        
        max_tokens = st.slider(
            "Max Tokens",
            min_value=50,
            max_value=1000,
            value=st.session_state.llm_settings["max_tokens"],
            step=50,
            help="Maximum length of generated text"
        )
        
        # Update settings in session state
        st.session_state.llm_settings.update({
            "temperature": temperature,
            "max_tokens": max_tokens
        })
        
        # About section
        st.markdown("---")
        st.subheader("About")
        st.info(
            "This application uses a Flask backend with an open-source Large Language Model to process documents. "
            "All processing happens locally - your data never leaves your machine."
        )
        
        # System info
        st.caption("System Info")
        if api_health_check():
            try:
                health_info = requests.get(f"{BACKEND_URL}/api/health").json()
                st.code(f"Model: {health_info.get('model', 'Unknown')}")
            except:
                st.code("Model: Connection error")

def render_document_input():
    """Render document input UI"""
    st.header("1️⃣ Document Input")
    
    # Create tabs for different input methods
    tab_upload, tab_paste = st.tabs(["📁 Upload File", "📝 Paste Text"])
    
    uploaded_file = None
    pasted_text = ""
    
    with tab_upload:
        uploaded_file = st.file_uploader(
            "Choose a document file:",
            type=["pdf", "docx", "txt", "html", "csv", "json"],
            help="Supported formats: PDF, Word, Text, HTML, CSV, JSON"
        )
        
        if uploaded_file:
            file_info = f"File: {uploaded_file.name} ({uploaded_file.type}) - {uploaded_file.size/1024:.1f} KB"
            st.info(file_info)
    
    with tab_paste:
        pasted_text = st.text_area(
            "Or paste document text here:",
            height=200,
            placeholder="Paste your text content here for processing..."
        )
    
    # Process button
    col1, col2 = st.columns([1, 1])
    
    with col1:
        process_btn = st.button(
            "🔄 Process Document",
            type="primary",
            disabled=(not uploaded_file and not pasted_text),
            use_container_width=True
        )
    
    with col2:
        if st.button("🗑️ Reset", use_container_width=True):
            st.session_state.clear()
            st.rerun()
    
    # Real-time processing indicator
    if process_btn:
        if uploaded_file or pasted_text:
            with st.spinner("Processing document... (This may take a while)"):
                result = process_document(uploaded_file, pasted_text)
            if result:
                st.session_state.active_document = result.get("document_id")
                st.session_state.active_document_details = result
                st.session_state.documents = get_document_list()
                # Initialize chat history for this document
                set_chat_history(st.session_state.active_document, [])
                st.session_state.chat_history = []
                st.success(f"Document processing started with ID: {result.get('document_id')}. Check the 'Your Documents' list or refresh to see status updates.")
                st.rerun()

def render_document_summaries():
    """Render document summaries UI"""
    if "active_document_details" not in st.session_state:
        return
    
    doc_details = st.session_state.active_document_details
    chunks = doc_details.get("chunks", [])
    file_info = doc_details.get("file_info", {})
    
    st.header("2️⃣ Document Analysis")
    
    # Document info panel
    st.subheader(f"Document: {file_info.get('filename', 'Document')}")
    
    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
    with metrics_col1:
        st.metric("Total Chunks", len(chunks))
    with metrics_col2:
        total_words = sum(chunk.get("word_count", 0) for chunk in chunks)
        st.metric("Total Words", total_words)
    with metrics_col3:
        avg_words = total_words / len(chunks) if chunks else 0
        st.metric("Avg. Words per Chunk", f"{avg_words:.0f}")
    with metrics_col4:
        # Get document download link
        summary_data = get_document_summary(st.session_state.active_document)
        if summary_data:
            markdown_content = summary_data.get("markdown", "")
            filename = summary_data.get("filename", "document_summary.md")
            # Create download button
            st.download_button(
                label="📥 Download Summary",
                data=markdown_content,
                file_name=filename,
                mime="text/markdown"
            )
    
    # Data visualization tab
    viz_tab, summaries_tab = st.tabs(["📊 Visualizations", "📋 Chunk Summaries"])
    
    with viz_tab:
        # Create dataframe for chunks
        chunk_data = []
        for i, chunk in enumerate(chunks):
            chunk_data.append({
                "Chunk": i+1,
                "Words": chunk.get("word_count", 0),
                "Characters": chunk.get("char_count", 0),
                "Summary Length": len(chunk.get("summary", ""))
            })
        
        if chunk_data:
            df = pd.DataFrame(chunk_data)
            
            # Create charts
            col1, col2 = st.columns(2)
            
            with col1:
                # Word count by chunk
                fig1 = px.bar(
                    df, 
                    x="Chunk", 
                    y="Words",
                    title="Word Count by Chunk",
                    color="Words",
                    color_continuous_scale="Viridis"
                )
                fig1.update_layout(height=300)
                st.plotly_chart(fig1, use_container_width=True)
            
            with col2:
                # Summary length vs content length
                fig2 = px.scatter(
                    df,
                    x="Words",
                    y="Summary Length",
                    title="Content vs Summary Length",
                    size="Characters",
                    color="Chunk",
                    hover_data=["Chunk", "Words", "Characters"]
                )
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)
    
    with summaries_tab:
        # Expandable section for chunk summaries
        for i, chunk in enumerate(chunks):
            with st.expander(f"Chunk {i+1}", expanded=(i==0)):
                st.markdown("**Summary:**")
                st.markdown(f"{chunk.get('summary', 'No summary available')}")
                
                st.markdown("**Content Excerpt:**")
                st.text(chunk.get("excerpt", "No excerpt available"))

def render_qa_interface():
    """Render Q&A interface"""
    if "active_document" not in st.session_state or not st.session_state.active_document:
        return
        
    st.header("3️⃣ Question Answering")
    
    doc_id = st.session_state.active_document
    # Use per-document chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = get_chat_history(doc_id)
    chat_history = st.session_state.chat_history
    chat_container = st.container()
    
    # Display chat history with improved style
    with chat_container:
        for entry in chat_history:
            timestamp = datetime.fromtimestamp(entry.get("timestamp", time.time())).strftime("%Y-%m-%d %H:%M:%S")
            if entry["role"] == "user":
                st.markdown(f'''
                <div style="display: flex; align-items: flex-end; margin-bottom: 10px;">
                    <div style="background: #007BFF; color: white; padding: 12px 18px; border-radius: 18px 18px 4px 18px; max-width: 70%; margin-left: auto;">
                        <b>You</b> <span style="font-size:0.8em; color:#e0e0e0; float:right;">{timestamp}</span><br>{entry["content"]}
                    </div>
                    <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=User" width="36" style="margin-left:8px; border-radius:50%;" />
                </div>
                ''', unsafe_allow_html=True)
            else:
                st.markdown(f'''
                <div style="display: flex; align-items: flex-end; margin-bottom: 10px;">
                    <img src="https://api.dicebear.com/7.x/avataaars/svg?seed=Felix" width="36" style="margin-right:8px; border-radius:50%;" />
                    <div style="background: #1E2A38; color: #A9CCE3; padding: 12px 18px; border-radius: 18px 18px 18px 4px; max-width: 70%;">
                        <b>Assistant</b> <span style="font-size:0.8em; color:#b0b0b0; float:right;">{timestamp}</span><br>{entry["content"]}
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                # Display sources if available
                if "sources" in entry and entry["sources"]:
                    with st.expander("View sources", expanded=False):
                        for i, source in enumerate(entry["sources"]):
                            st.markdown(f"**Source {i+1} (Chunk {source['chunk_index']+1}):**")
                            st.markdown(f"**Relevance Score:** {source['similarity']:.2f}")
                            st.text(source["excerpt"])
                            st.markdown("---")
    
    # Input for new question
    st.markdown("---")
    col1, col2, col3 = st.columns([5, 1, 1])
    
    with col1:
        def on_ask():
            question = st.session_state.get("question_input", "")
            if question:
                chat_history.append({
                    "role": "user",
                    "content": question,
                    "timestamp": time.time()
                })
                set_chat_history(doc_id, chat_history)
                st.session_state.chat_history = chat_history
                with st.spinner("Generating answer..."):
                    answer_data = ask_question(question, doc_id)
                if answer_data:
                    answer = answer_data.get("answer", "Sorry, I couldn't generate an answer.")
                    sources = answer_data.get("sources", [])
                    chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "timestamp": time.time()
                    })
                    set_chat_history(doc_id, chat_history)
                    st.session_state.chat_history = chat_history
                st.session_state.question_input = ""  # Clear input
                st.rerun()

        question = st.text_input(
            "Ask a question about the document:",
            key="question_input",
            placeholder="What would you like to know about this document? (Press Enter to send)",
            on_change=on_ask
        )
    
    with col2:
        send_btn = st.button("🔍", use_container_width=True, help="Send question")
    
    with col3:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Clear chat history for this document"):
            set_chat_history(doc_id, [])
            st.session_state.chat_history = []
            st.rerun()
    
    if send_btn and question:
        chat_history.append({
            "role": "user",
            "content": question,
            "timestamp": time.time()
        })
        set_chat_history(doc_id, chat_history)
        st.session_state.chat_history = chat_history
        with st.spinner("Generating answer..."):
            answer_data = ask_question(question, doc_id)
        if answer_data:
            answer = answer_data.get("answer", "Sorry, I couldn't generate an answer.")
            sources = answer_data.get("sources", [])
            chat_history.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
                "timestamp": time.time()
            })
            set_chat_history(doc_id, chat_history)
            st.session_state.chat_history = chat_history
        st.session_state.question_input = ""
        st.rerun()

# Main application
def main():
    # Check if backend is available
    backend_available = api_health_check()
    
    # Apply theme class
    theme_class = "" if st.session_state.dark_mode else "light-theme"
    st.markdown(f"""<div class="{theme_class}">""", unsafe_allow_html=True)
    
    # Render header
    render_header()
    
    # Show warning if backend not available
    if not backend_available:
        st.error("""
        🚨 Backend server is not available. 
        
        Make sure the Flask backend is running on port 5001. You can start it with:
        ```
        python backend/app.py
        ```
        """)
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    if "active_document" in st.session_state and st.session_state.active_document:
        # Document is loaded, show analysis and QA
        doc_id = st.session_state.active_document
        if "chat_history" not in st.session_state or st.session_state.chat_history != get_chat_history(doc_id):
            st.session_state.chat_history = get_chat_history(doc_id)
        render_document_summaries()
        render_qa_interface()
    else:
        # No document loaded, show input form
        render_document_input()
    
    # Close theme div
    st.markdown("""</div>""", unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.caption("Advanced Document Q&A System | Powered by Open-Source LLMs")

if __name__ == "__main__":
    main()