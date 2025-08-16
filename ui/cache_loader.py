"""Cache loading utilities for the Op-Ed Analyzer UI."""

import json
import streamlit as st
from pathlib import Path
from models import ExtractedClaim, TitledDocument


def load_cached_analysis(cache_hash: str) -> bool:
    """Load cached analysis data into session state.
    
    Args:
        cache_hash: The cache hash to load from
        
    Returns:
        True if loading was successful, False otherwise
    """
    cache_dir = Path("data/cache") / cache_hash.strip()
    if not cache_dir.exists():
        st.error("Cache hash not found")
        return False
    
    # Try to load claims first to get document info
    claims_dir = cache_dir / "claims"
    if not claims_dir.exists():
        st.error("Invalid cache hash - no claims directory found")
        return False
    
    # Load all claims from cache
    all_claims = []
    for claim_file in claims_dir.glob("*.json"):
        claims_data = json.loads(claim_file.read_text())
        all_claims.extend([ExtractedClaim(**claim) for claim in claims_data])
    
    if not all_claims:
        st.error("No claims found in cache directory")
        return False
    
    # Try to load cached TitledDocuments first
    titled_docs_dir = cache_dir / "titled_documents"
    titled_docs_file = titled_docs_dir / "documents.json"
    
    if titled_docs_file.exists():
        # Load from cached TitledDocuments
        titled_docs_data = json.loads(titled_docs_file.read_text())
        titled_docs = [TitledDocument(**doc) for doc in titled_docs_data]
    else:
        # Fallback: create from claims (for old cached analyses)
        docs_by_id = {}
        doc_titles = {}
        for claim in all_claims:
            if claim.doc_id not in docs_by_id:
                docs_by_id[claim.doc_id] = ""  # No text available in old cache
                doc_titles[claim.doc_id] = getattr(claim, 'doc_title', claim.doc_id)
        
        titled_docs = [
            TitledDocument(id=doc_id, text=text, title=doc_titles[doc_id])
            for doc_id, text in docs_by_id.items()
        ]
    
    # Store in session state
    st.session_state.all_claims = all_claims
    st.session_state.titled_documents = titled_docs
    st.session_state.num_docs = len(titled_docs)
    st.session_state.claims_per_doc = len([c for c in all_claims if c.doc_id == titled_docs[0].id])
    
    return True