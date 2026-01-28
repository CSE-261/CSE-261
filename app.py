"""Streamlit frontend for RAG application with source navigation."""

import streamlit as st
import sys
from pathlib import Path
from typing import Dict, Any, List
import base64
import tempfile
import os
import hashlib

# Add parent directory to path to import src as a package
sys.path.insert(0, str(Path(__file__).parent))

from src.config import RAGConfig
from src.retrieval import RAGPipeline

# Page config
st.set_page_config(
    page_title="RAG Application",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Citation footnote style */
    .citation {
        display: inline-block;
        background-color: #e3f2fd;
        color: #1565c0;
        padding: 0.1rem 0.4rem;
        border-radius: 0.3rem;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 0 0.2rem;
        cursor: pointer;
        text-decoration: none;
        border: 1px solid #90caf9;
    }
    .citation:hover {
        background-color: #bbdefb;
        border-color: #64b5f6;
    }

    /* Compact source reference */
    .source-ref {
        padding: 0.5rem;
        margin: 0.3rem 0;
        border-left: 3px solid #1976d2;
        background-color: #f5f5f5;
        font-size: 0.9rem;
    }
    .source-ref-header {
        font-weight: 600;
        color: #1565c0;
        margin-bottom: 0.2rem;
    }
    .source-ref-meta {
        color: #666;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_rag_pipeline():
    """Initialize RAG pipeline (cached)."""
    config = RAGConfig()
    return RAGPipeline(config)


def display_source_file(file_path: str, page_num: int = None, highlight_text: str = None):
    """Display source file (PDF or TXT) with optional highlighting."""
    try:
        # Debug info
        st.caption(f"📁 Attempting to open: {file_path}")

        file_path_obj = Path(file_path)

        # Check if file exists
        if not file_path_obj.exists():
            st.error(f"❌ File not found: {file_path}")
            st.info("💡 Tip: Make sure the file path in metadata matches the actual file location")
            return

        file_extension = file_path_obj.suffix.lower()

        if file_extension == '.pdf':
            # Display PDF file
            with open(file_path, "rb") as f:
                pdf_bytes = f.read()

            # Encode PDF to base64
            base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')

            # Create PDF viewer with page navigation
            pdf_display = f'''
            <iframe
                src="data:application/pdf;base64,{base64_pdf}#page={page_num if page_num else 1}"
                width="100%"
                height="800"
                type="application/pdf"
                style="border: 1px solid #ccc; border-radius: 4px;">
            </iframe>
            '''
            st.markdown(pdf_display, unsafe_allow_html=True)
            st.success(f"✅ PDF loaded successfully ({len(pdf_bytes)} bytes)")

        elif file_extension == '.txt':
            # Display TXT file
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()

            # Display in a scrollable text area
            st.text_area(
                "File Content",
                value=content,
                height=800,
                disabled=True,
                key=f"txt_viewer_{file_path}"
            )

            # If highlight text is provided, show it
            if highlight_text:
                st.info(f"💡 Matching content: {highlight_text[:200]}...")

            st.success(f"✅ TXT file loaded successfully ({len(content)} characters)")

        else:
            st.warning(f"Unsupported file type: {file_extension}. Only PDF and TXT are supported.")

    except FileNotFoundError as e:
        st.error(f"❌ File not found: {file_path}")
        st.code(str(e))
    except Exception as e:
        st.error(f"❌ Error loading file: {type(e).__name__}")
        st.code(str(e))
        import traceback
        st.code(traceback.format_exc())


def format_bbox(bbox: Dict[str, float]) -> str:
    """Format bounding box coordinates."""
    return f"({bbox['x0']:.1f}, {bbox['y0']:.1f}) → ({bbox['x1']:.1f}, {bbox['y1']:.1f})"


def create_html_viewer(file_path: str, highlight_text: str, file_name: str) -> str:
    """Create HTML page with highlighted text for viewing in new tab."""
    try:
        file_ext = Path(file_path).suffix.lower()

        if file_ext == '.txt':
            # Read text file
            with open(file_path, "r", encoding='utf-8') as f:
                content = f.read()

            # Escape HTML characters
            import html
            content_escaped = html.escape(content)
            highlight_escaped = html.escape(highlight_text)

            # Highlight the matching chunk
            if highlight_text and highlight_text in content:
                content_highlighted = content_escaped.replace(
                    highlight_escaped,
                    f'<mark style="background-color: #ffeb3b; padding: 2px 4px; border-radius: 2px;">{highlight_escaped}</mark>'
                )
            else:
                content_highlighted = content_escaped

            # Create HTML page
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>{file_name}</title>
                <style>
                    body {{
                        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                        max-width: 900px;
                        margin: 0 auto;
                        padding: 20px;
                        line-height: 1.6;
                        background-color: #f5f5f5;
                    }}
                    .header {{
                        background-color: #0068c9;
                        color: white;
                        padding: 20px;
                        border-radius: 8px;
                        margin-bottom: 20px;
                    }}
                    .content {{
                        background-color: white;
                        padding: 30px;
                        border-radius: 8px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        white-space: pre-wrap;
                        font-family: 'Courier New', monospace;
                        font-size: 14px;
                    }}
                    mark {{
                        background-color: #ffeb3b;
                        padding: 2px 4px;
                        border-radius: 2px;
                        font-weight: bold;
                    }}
                    .info {{
                        color: #666;
                        font-size: 0.9em;
                        margin-top: 10px;
                    }}
                </style>
            </head>
            <body>
                <div class="header">
                    <h1>📄 {file_name}</h1>
                    <div class="info">Highlighted text shows the retrieved chunk relevant to your query</div>
                </div>
                <div class="content">{content_highlighted}</div>
            </body>
            </html>
            """
            return html_content

        elif file_ext == '.pdf':
            # For PDF, return None (we'll handle differently)
            return None
        else:
            return None

    except Exception as e:
        return f"<html><body><h1>Error loading file</h1><p>{str(e)}</p></body></html>"


def create_citations(response: str, docs: List[Dict[str, Any]]) -> str:
    """Add citation footnotes to the response."""
    # Add citations at the end of the response
    citations_text = "\n\n---\n**Sources:**\n"
    for idx, doc in enumerate(docs, 1):
        source = Path(doc['metadata']['source']).name
        page = doc['metadata'].get('page', 'N/A')
        citations_text += f"\n[{idx}] {source} (Page {page})"

    return response + citations_text


def display_compact_source(doc: Dict[str, Any], idx: int):
    """Display a compact source reference with link to open in new tab."""
    metadata = doc['metadata']
    score = doc['score']

    # Extract metadata
    source = Path(metadata['source']).name
    page = metadata.get('page', 'N/A')
    para_index = metadata.get('para_index', 'N/A')
    bbox = metadata.get('bbox', None)

    # Create expandable section for each source
    with st.expander(f"[{idx+1}] {source} • Page {page} • Score: {score:.2f}", expanded=False):
        # Show preview of content
        st.markdown(f"**Excerpt:**")
        st.markdown(f"> {doc['content'][:200]}...")

        # Metadata
        col1, col2 = st.columns(2)
        with col1:
            st.caption(f"📄 Paragraph: {para_index}")
        with col2:
            if bbox:
                st.caption(f"📍 Position: {format_bbox(bbox)}")

        # View in new tab with highlighting
        file_path = metadata['source']
        file_ext = Path(file_path).suffix.lower()

        # Normalize path (remove double slashes)
        file_path = file_path.replace('//', '/')

        try:
            if file_ext == '.txt':
                # Generate HTML with highlighted chunk
                html_content = create_html_viewer(file_path, doc['content'], source)

                if html_content:
                    import streamlit.components.v1 as components

                    # Escape the HTML content for JavaScript
                    html_escaped = (html_content
                                  .replace('\\', '\\\\')
                                  .replace('`', '\\`')
                                  .replace('$', '\\$'))

                    # Use Streamlit components to create a button that opens a new window
                    components.html(f"""
                        <button
                            onclick="openWindow()"
                            style="display: block;
                                   padding: 0.5rem 1rem;
                                   background-color: #0068c9;
                                   color: white;
                                   border: none;
                                   border-radius: 0.25rem;
                                   font-weight: 500;
                                   cursor: pointer;
                                   width: 100%;
                                   font-family: sans-serif;
                                   font-size: 14px;">
                            🔗 View File with Highlights
                        </button>
                        <script>
                        function openWindow() {{
                            const htmlContent = `{html_escaped}`;
                            const newWindow = window.open('', '_blank');
                            if (newWindow) {{
                                newWindow.document.write(htmlContent);
                                newWindow.document.close();
                            }} else {{
                                alert('Please allow pop-ups for this site to view the file.');
                            }}
                        }}
                        </script>
                    """, height=50)
                else:
                    st.error("Failed to generate viewer")

            elif file_ext == '.pdf':
                # For PDF files - open directly
                with open(file_path, "rb") as f:
                    pdf_bytes = f.read()

                base64_pdf = base64.b64encode(pdf_bytes).decode('utf-8')
                pdf_link = f'<a href="data:application/pdf;base64,{base64_pdf}" target="_blank" style="display: inline-block; padding: 0.5rem 1rem; background-color: #0068c9; color: white; text-decoration: none; border-radius: 0.25rem; font-weight: 500; text-align: center; width: 100%;">🔗 View PDF</a>'
                st.markdown(pdf_link, unsafe_allow_html=True)
                st.caption("⚠️ PDF highlighting not yet supported")

            else:
                st.caption("⚠️ Unsupported file type")

        except FileNotFoundError:
            st.error(f"❌ File not found: {file_path}")
            st.caption(f"Looking for: {file_path}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.caption(f"Path: {file_path}")


def main():
    """Main Streamlit application."""

    # Initialize session state
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'retrieved_docs' not in st.session_state:
        st.session_state.retrieved_docs = []

    # Initialize RAG pipeline
    with st.spinner("Loading RAG pipeline..."):
        rag = init_rag_pipeline()

    # Title
    st.title("📚 RAG Application")
    st.markdown("Ask questions about your documents and navigate to the source!")

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        st.caption("Settings from .env file")
        st.info(f"**Max Sources:** {rag.config.top_k}")
        st.info(f"**Score Threshold:** {rag.config.score_threshold}")

        st.markdown("---")
        st.header("📊 Statistics")
        st.metric("Questions Asked", len(st.session_state.messages) // 2)
        st.metric("Sources Retrieved", len(st.session_state.retrieved_docs))

        if st.button("🗑️ Clear History"):
            st.session_state.messages = []
            st.session_state.retrieved_docs = []
            st.rerun()

        # Source Details in sidebar
        if st.session_state.retrieved_docs:
            st.markdown("---")
            st.header("📚 Source Details")
            st.caption("Click 🔗 to open in new tab")

            for idx, doc in enumerate(st.session_state.retrieved_docs):
                display_compact_source(doc, idx)

    # BEST PRACTICE: Display all chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # BEST PRACTICE: Chat input at the main level (auto-pinned to bottom)
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Use config values from .env
                result = rag.query(prompt)
                response = result['answer']
                retrieved_docs = result['retrieved_docs']

                # Debug: Show retrieval info
                st.caption(f"🔍 Retrieved {len(retrieved_docs)} documents (threshold: {rag.config.score_threshold})")
                if len(retrieved_docs) == 0:
                    st.warning("⚠️ No documents met the score threshold. Try lowering RAG_SCORE_THRESHOLD in .env")

                # Add citations to response
                response_with_citations = create_citations(response, retrieved_docs)
                st.markdown(response_with_citations)

        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response_with_citations})
        # Store retrieved docs
        st.session_state.retrieved_docs = retrieved_docs
        # Force rerun to update sidebar with new sources
        st.rerun()


if __name__ == "__main__":
    main()
