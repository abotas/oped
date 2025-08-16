"""Op-Ed Analyzer UI - Simple Streamlit interface for analyzing op-ed documents"""

import streamlit as st
import sys
import os
from pathlib import Path
from dotenv import load_dotenv
load_dotenv()

# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import InputConfig
from ui.cache_loader import load_cached_analysis
from ui.analysis_pipeline import run_analysis_pipeline


# Page config
st.set_page_config(
    page_title="Op-Ed Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def main():
    # Header with Example button
    col1, col2 = st.columns([4, 1])
    with col1:
        st.title("📝 Op-Ed Analyzer")
    with col2:
        # Add spacing to align with title
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        if st.button("📚 Example", help="Machines of loving grace + Sama blog", use_container_width=True):
            # Load the specific cached analysis
            cache_hash = "db6ed1d90088"
            if load_cached_analysis(cache_hash):
                config = InputConfig.load_from_cache(cache_hash)
                st.session_state.config = config
                st.session_state.cache_hash = cache_hash
                st.success(f"✅ Loaded example analysis")
                st.rerun()
    
    # Check if we're in analysis mode (session state indicates analysis has started)
    analysis_started = st.session_state.get('analysis_started', False)
    
    if not analysis_started:
        # Debug hash loading option
        with st.expander("🔍 Debug: Load from Cache Hash", expanded=False):
            # Get available cache hashes
            cache_dir = Path("data/cache")
            available_hashes = []
            if cache_dir.exists():
                available_hashes = [d.name for d in cache_dir.iterdir() if d.is_dir()]
            
            if available_hashes:
                cache_hash = st.selectbox(
                    "Select cache hash to load existing analysis",
                    options=[""] + available_hashes,
                    format_func=lambda x: "Choose a hash..." if x == "" else x
                )
                load_button_disabled = not cache_hash
            else:
                st.info("No cached analyses found")
                cache_hash = ""
                load_button_disabled = True
            
            if st.button("Load from Hash", disabled=load_button_disabled):
                if load_cached_analysis(cache_hash.strip()):
                    config = InputConfig.load_from_cache(cache_hash.strip())
                    st.session_state.config = config
                    st.session_state.cache_hash = cache_hash.strip()
                    st.success(f"✅ Loaded analysis from cache hash {cache_hash.strip()}")
                    st.rerun()
        
        st.markdown("---")
        
        # Configuration
        col1, col2, col3 = st.columns(3)
        with col1:
            num_docs = st.number_input("Number of documents", min_value=1, value=2)
        with col2:
            claims_per_doc = st.number_input("Claims per document", min_value=1, value=5)
        with col3:
            # Model selector
            selected_model = st.selectbox(
                "Model",
                options=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
                index=1,  # Default to gpt-5-mini
                help="Select the model to use for analysis"
            )
        
        # Topic input (optional)
        topic = st.text_input(
            "Topic (optional)",
            placeholder="e.g., artificial intelligence, climate change, healthcare...",
            help="Focus claim extraction on a specific topic"
        )
        
        # Text input areas
        documents = []
        all_docs_filled = True
        
        for i in range(num_docs):
            doc_text = st.text_area(
                f"Document {i+1}",
                height=200,
                placeholder=f"Paste op-ed text here...",
                key=f"doc_{i}"
            )
            if doc_text.strip():
                documents.append({
                    "id": f"doc_{i+1}",
                    "text": doc_text.strip()
                })
            else:
                all_docs_filled = False
        
        # Analyze button
        analyze_enabled = all_docs_filled and len(documents) == num_docs
        analyze_clicked = st.button("🔍 Analyze Documents", disabled=not analyze_enabled, type="primary")
        
        if not analyze_enabled:
            if not documents:
                st.info("👆 Paste op-ed text in all boxes above, then click Analyze")
            else:
                st.info(f"Please fill in all {num_docs} document boxes to enable analysis")
            return
        
        if not analyze_clicked:
            return
        
        # Create config object with all parameters
        config = InputConfig(
            documents=documents,
            claims_per_doc=claims_per_doc,
            model=selected_model,
            topic=topic.strip() if topic.strip() else None
        )
        
        # Save config to cache and store in session state
        cache_hash = config.save_to_cache()
        st.session_state.config = config
        st.session_state.cache_hash = cache_hash
        st.session_state.num_docs = num_docs
        st.session_state.claims_per_doc = claims_per_doc
        st.session_state.selected_model = selected_model
        st.session_state.topic = config.topic
        st.session_state.raw_documents = documents
        st.session_state.analysis_started = True
        
        # Initialize session state for analysis steps
        st.session_state.titled_documents = None
        st.session_state.all_claims = None
        st.session_state.coherence_results = None
        st.session_state.fact_checks = None
        
        st.rerun()
    
    else:
        # We're in analysis mode - retrieve stored configuration
        config = st.session_state.config
        num_docs = st.session_state.num_docs
        claims_per_doc = config.claims_per_doc
        selected_model = config.model
        topic = config.topic
        documents = config.documents
        
        # Show analysis header with document info and reset button
        col1, col2 = st.columns([3, 1])
        with col1:
            # st.markdown(f"### 📊 Analysis Results")
            topic_text = f" • Topic: {topic}" if topic else ""
            st.markdown(f"**{num_docs} documents • {claims_per_doc} claims each{topic_text}**")
            # Show cache hash for debugging
            current_hash = config.generate_cache_hash()
            st.code(f"Cache Hash: {current_hash}", language=None)
        with col2:
            if st.button("🔄 New Analysis", type="secondary", use_container_width=True):
                # Clear session state to start fresh
                for key in ['config', 'all_claims', 'coherence_results', 'fact_checks', 'num_docs', 'claims_per_doc', 'titled_documents', 'raw_documents', 'selected_model', 'topic', 'analysis_started']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Run the complete 4-step analysis pipeline with config
        run_analysis_pipeline(config)


if __name__ == "__main__":
    main()