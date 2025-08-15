"""
Op-Ed Analyzer UI - Simple Streamlit interface for analyzing op-ed documents
"""

import streamlit as st
import sys
import os
import json
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import textwrap


# Add parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from claim_extractor import extract_claims_for_docs
from claim_coherence import analyze_coherence, coherence_to_matrix
from external_fact_checking import check_facts, get_fact_check_summary
from models import ExtractedClaim
from utils import get_conflict_metrics, get_top_load_bearing_claims, get_top_load_bearing_claims_filtered


# Page config
st.set_page_config(
    page_title="Op-Ed Analyzer",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed"
)


def wrap_text_for_hover(text, max_width=100):
    """Wrap text for hover tooltips to avoid excessively wide tooltips."""
    if not text:
        return ""
    
    # Convert to string if not already
    text = str(text)
    
    # Split by existing newlines first to preserve intentional breaks
    lines = text.split('\n')
    wrapped_lines = []
    
    for line in lines:
        # Wrap each line if it's too long
        if len(line) > max_width:
            wrapped = textwrap.wrap(line, width=max_width, break_long_words=False, break_on_hyphens=False)
            if wrapped:  # Only extend if wrap returned something
                wrapped_lines.extend(wrapped)
            else:
                wrapped_lines.append(line)  # Fallback to original line
        else:
            wrapped_lines.append(line)
    
    result = '<br>'.join(wrapped_lines)
    
    # Escape HTML characters but preserve line breaks
    result = result.replace('&', '&amp;')
    # Don't escape < and > as they might be in HTML tags we want to preserve
    
    return result


def create_veracity_buckets(fact_checks, selected_docs):
    """Create an interactive validation score bucket visualization using Plotly."""
    if not fact_checks or not selected_docs:
        return None
    
    # Filter fact checks based on selected documents
    filtered_checks = [fc for fc in fact_checks if fc.doc_id in selected_docs]
    if not filtered_checks:
        return None
    
    # Define buckets
    bucket_ranges = [(i, i+10) for i in range(0, 100, 10)]
    bucket_labels = [f"{i}-{i+10}" for i in range(0, 100, 10)]
    
    # Group claims into buckets
    buckets = {i: [] for i in range(10)}
    for fc in filtered_checks:
        bucket_idx = min(fc.veracity // 10, 9)  # Handle veracity=100 case
        buckets[bucket_idx].append(fc)
    
    # Prepare data for visualization
    x_positions = []
    y_positions = []
    colors = []
    hover_texts = []
    custom_data = []  # Store raw data for custom hover
    
    # Define color scale from red to green
    color_scale = [
        '#8B0000',  # 0-10: Dark red
        '#CD5C5C',  # 10-20: Indian red
        '#F08080',  # 20-30: Light coral
        '#FFA07A',  # 30-40: Light salmon
        '#FFD700',  # 40-50: Gold
        '#ADFF2F',  # 50-60: Green yellow
        '#7FFF00',  # 60-70: Chartreuse
        '#32CD32',  # 70-80: Lime green
        '#228B22',  # 80-90: Forest green
        '#006400'   # 90-100: Dark green
    ]
    
    # Create positions for each claim block
    for bucket_idx in range(10):
        bucket_claims = buckets[bucket_idx]
        for claim_idx, fc in enumerate(bucket_claims):
            x_positions.append(bucket_idx)
            y_positions.append(claim_idx)
            colors.append(color_scale[bucket_idx])
            
            # Wrap text for better display in hover (full text, no truncation)
            wrapped_claim = wrap_text_for_hover(fc.claim)
            wrapped_explanation = wrap_text_for_hover(fc.explanation)
            
            # Store full wrapped data for custom hover
            custom_data.append([fc.veracity, fc.doc_id, wrapped_claim, wrapped_explanation])
            
            # Keep simple hover text for fallback
            hover_text = f"Score: {fc.veracity}/100 | Doc: {fc.doc_id}"
            hover_texts.append(hover_text)
    
    # Debug output
    print(f"DEBUG: Creating validation plot with {len(x_positions)} points")
    if hover_texts:
        print(f"DEBUG: First hover text: {hover_texts[0]}")
    
    # Create the scatter plot with square markers - try customdata approach
    fig = go.Figure()
    
    # Add all claim blocks as a single trace - using customdata for hover
    if x_positions:  # Only add trace if we have data
        fig.add_trace(go.Scatter(
            x=x_positions,
            y=y_positions,
            mode='markers',
            marker=dict(
                size=30,
                color=colors,
                symbol='square',
                line=dict(color='white', width=1)
            ),
            customdata=custom_data,
            hovertemplate=(
                'Validation Score: %{customdata[0]}/100<br>' +
                'Document: %{customdata[1]}<br>' +
                'Claim: %{customdata[2]}<br>' +
                'Explanation: %{customdata[3]}' +
                '<extra></extra>'
            ),
            hoverlabel=dict(
                align='left',
                namelength=-1
            ),
            showlegend=False
        ))
    
    # Add bucket labels and counts
    for bucket_idx in range(10):
        count = len(buckets[bucket_idx])
        if count > 0:
            fig.add_annotation(
                x=bucket_idx,
                y=-1.5,
                text=f"{bucket_labels[bucket_idx]}<br>({count} claims)",
                showarrow=False,
                font=dict(size=10, color='white'),
                align='center'
            )
    
    # Calculate max height for y-axis
    max_height = max([len(bucket) for bucket in buckets.values()]) if buckets else 1
    
    fig.update_layout(
        title=dict(
            text="Claim Validation Distribution",
            font=dict(size=18, family="Arial, sans-serif", color="white"),
            x=0.5,
            xanchor='center'
        ),
        xaxis=dict(
            title="Validation Score Buckets",
            tickmode='array',
            tickvals=list(range(10)),
            ticktext=bucket_labels,
            tickangle=45,
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.5)',
            tickfont=dict(color='white')
        ),
        yaxis=dict(
            title="Claims per Bucket",
            range=[-2, max_height],
            showgrid=True,
            gridcolor='rgba(255,255,255,0.1)',
            linecolor='rgba(255,255,255,0.5)',
            tickfont=dict(color='white'),
            showticklabels=False
        ),
        plot_bgcolor='rgba(255,255,255,0.05)',
        paper_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color="white"),
        height=600,
        margin=dict(l=60, r=250, t=60, b=100),
        hovermode='closest'
    )
    
    return fig

def create_coherence_matrix(coherence_results, claims, all_claims, full_coherence_results=None):
    """Create an interactive coherence matrix visualization using Plotly."""
    if not coherence_results or not claims:
        return None
    
    # Create mapping from original claim indices to filtered claim indices
    original_to_filtered = {}
    for filtered_idx, claim in enumerate(claims):
        # Find the original index of this claim in all_claims
        for original_idx, original_claim in enumerate(all_claims):
            if (claim.doc_id == original_claim.doc_id and 
                claim.claim_idx == original_claim.claim_idx and
                claim.claim == original_claim.claim):
                original_to_filtered[original_idx] = filtered_idx
                break
    
    # Remap coherence results to use filtered indices
    remapped_coherence = []
    for c in coherence_results:
        if c.claim_i_idx in original_to_filtered and c.claim_j_idx in original_to_filtered:
            # Create a new coherence object with remapped indices
            from claim_coherence import ClaimCoherence
            remapped_c = ClaimCoherence(
                claim_i_idx=original_to_filtered[c.claim_i_idx],
                claim_j_idx=original_to_filtered[c.claim_j_idx],
                delta_prob=c.delta_prob,
                reasoning=c.reasoning
            )
            remapped_coherence.append(remapped_c)
    
    # Generate the matrix with remapped coherence
    matrix = coherence_to_matrix(remapped_coherence, len(claims))
    
    # Create claim labels with doc grouping
    claim_labels = []
    doc_groups = {}
    
    # Group claims by document
    for i, claim in enumerate(claims):
        if claim.doc_id not in doc_groups:
            doc_groups[claim.doc_id] = []
        doc_groups[claim.doc_id].append(i)
        
        # Create short label for axes (just claim number within doc)
        claim_num = len([c for c in claims[:i+1] if c.doc_id == claim.doc_id])
        claim_labels.append(f"{claim.doc_id}[{claim_num-1}]")
    
    # Create detailed hover text with full claims
    hover_text = []
    for i in range(len(claims)):
        hover_row = []
        for j in range(len(claims)):
            if i == j:
                wrapped_claim = wrap_text_for_hover(claims[i].claim)
                hover_text_cell = f"<b>Self-relationship:</b> 1.0<br><br><b>Claim:</b><br>{wrapped_claim}"
            else:
                delta = matrix[i][j]
                sign = "+" if delta >= 0 else ""
                
                # Find the reasoning for this relationship
                reasoning = ""
                search_results = full_coherence_results if full_coherence_results else coherence_results
                for c in search_results:
                    # Map filtered indices back to original indices to find the relationship
                    orig_i = None
                    orig_j = None
                    for orig_idx, orig_claim in enumerate(all_claims):
                        if (orig_claim.doc_id == claims[i].doc_id and 
                            orig_claim.claim_idx == claims[i].claim_idx and
                            orig_claim.claim == claims[i].claim):
                            orig_i = orig_idx
                        if (orig_claim.doc_id == claims[j].doc_id and 
                            orig_claim.claim_idx == claims[j].claim_idx and
                            orig_claim.claim == claims[j].claim):
                            orig_j = orig_idx
                    
                    if orig_i is not None and orig_j is not None and c.claim_i_idx == orig_i and c.claim_j_idx == orig_j:
                        reasoning = c.reasoning
                        break
                
                # Wrap claim text for better display
                claim_i_wrapped = wrap_text_for_hover(claims[i].claim)
                claim_j_wrapped = wrap_text_for_hover(claims[j].claim)
                reasoning_wrapped = wrap_text_for_hover(reasoning) if reasoning else "No reasoning available"
                
                hover_text_cell = (
                    f"<b>Effect:</b> {sign}{delta:.2f}<br><br>"
                    f"<b>Source Claim A ({claim_labels[i]}):</b><br>"
                    f"{claim_i_wrapped}<br><br>"
                    f"<b>→ Target Claim B ({claim_labels[j]}):</b><br>"
                    f"{claim_j_wrapped}<br><br>"
                    f"<b>Reasoning:</b><br>"
                    f"{reasoning_wrapped}"
                )
            hover_row.append(hover_text_cell)
        hover_text.append(hover_row)
    
    # Create the heatmap
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
        x=claim_labels,
        y=claim_labels,
        colorscale=[
            [0.0, 'darkred'],      # -1.0 (strong negative)
            [0.25, 'red'],         # -0.5
            [0.5, 'white'],        # 0.0 (neutral)
            [0.75, 'lightgreen'],  # +0.5
            [1.0, 'darkgreen']     # +1.0 (strong positive)
        ],
        zmid=0,  # Center the colorscale at 0
        zmin=-1,
        zmax=1,
        hovertemplate='%{hovertext}<extra></extra>',
        hovertext=hover_text,
        showscale=True,
        colorbar=dict(
            title=dict(text="Probability delta"),
            tickmode="linear",
            tick0=-1,
            dtick=0.5
        )
    ))
    
    # Add document group separators
    doc_boundaries = []
    current_pos = 0
    for doc_id in sorted(doc_groups.keys()):
        current_pos += len(doc_groups[doc_id])
        if current_pos < len(claims):  # Don't add line after last group
            doc_boundaries.append(current_pos - 0.5)
    
    # Add lines to separate document groups
    for boundary in doc_boundaries:
        # Vertical line
        fig.add_shape(
            type="line",
            x0=boundary, x1=boundary,
            y0=-0.5, y1=len(claims)-0.5,
            line=dict(color="black", width=2)
        )
        # Horizontal line  
        fig.add_shape(
            type="line",
            x0=-0.5, x1=len(claims)-0.5,
            y0=boundary, y1=boundary,
            line=dict(color="black", width=2)
        )
    
    fig.update_layout(
        title=dict(
            text="Claim Coherence Matrix: How Claims Affect Each Other's Likelihood",
            font=dict(size=20, family="Arial, sans-serif", color="white"),
            x=0.5,
            xanchor='center'
        ),
        xaxis_title=dict(
            text="Target Claims (affected)",
            font=dict(size=14, color="white")
        ),
        yaxis_title=dict(
            text="Source Claims (affecting)", 
            font=dict(size=14, color="white")
        ),
        width=None,  # Let it be responsive
        height=600,
        xaxis=dict(
            tickangle=45,
            tickfont=dict(size=10, color="white"),
            gridcolor="rgba(255,255,255,0.3)",
            linecolor="rgba(255,255,255,0.5)"
        ),
        yaxis=dict(
            autorange='reversed',  # Reverse y-axis to match matrix convention
            tickfont=dict(size=10, color="white"),
            gridcolor="rgba(255,255,255,0.3)", 
            linecolor="rgba(255,255,255,0.5)"
        ),
        plot_bgcolor='rgba(255,255,255,0.05)',
        paper_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color="white"),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig

def main():
    # Header
    st.title("📝 Op-Ed Analyzer")
    
    # Check if we're in demo mode (using preloaded cache data)
    demo_mode = True
    cache_hash = "5a1ccfd92f5c"
    
    if demo_mode:
        # Load demo data from preloaded cache
        cache_dir = Path("data/cache") / cache_hash
        if cache_dir.exists():
            # Load all claims from cache
            claims_dir = cache_dir / "claims"
            if claims_dir.exists():
                all_claims = []
                for claim_file in claims_dir.glob("*.json"):
                    claims_data = json.loads(claim_file.read_text())
                    all_claims.extend([ExtractedClaim(**claim) for claim in claims_data])
                
                if all_claims:
                    # Store in session state
                    st.session_state.all_claims = all_claims
                    # Create mock documents structure for compatibility
                    docs_by_id = {}
                    for claim in all_claims:
                        if claim.doc_id not in docs_by_id:
                            docs_by_id[claim.doc_id] = claim.document_text
                    
                    documents = [{"id": doc_id, "text": text} for doc_id, text in docs_by_id.items()]
                    st.session_state.documents = documents
                    st.session_state.num_docs = len(documents)
                    st.session_state.claims_per_doc = len([c for c in all_claims if c.doc_id == documents[0]["id"]])
                    
                    # Load coherence results
                    coherence_dir = cache_dir / "coherence"
                    if coherence_dir.exists():
                        coherence_results = []
                        for coherence_file in coherence_dir.glob("*.json"):
                            coherence_data = json.loads(coherence_file.read_text())
                            from claim_coherence import ClaimCoherence
                            coherence_results.append(ClaimCoherence(**coherence_data))
                        st.session_state.coherence_results = coherence_results
                    
                    # Load fact checks
                    fact_checks_dir = cache_dir / "fact_checks"
                    if fact_checks_dir.exists():
                        fact_checks = []
                        for fact_check_file in fact_checks_dir.glob("*.json"):
                            fact_check_data = json.loads(fact_check_file.read_text())
                            from external_fact_checking import FactCheck
                            fact_checks.append(FactCheck(**fact_check_data))
                        st.session_state.fact_checks = fact_checks
                    
                    st.info("🎬 Demo Mode: Showing preloaded analysis results")
        
        # Skip the input interface and go straight to analysis display
        analysis_started = True
    else:
        # Check if we're in analysis mode (session state indicates analysis has started)
        analysis_started = any(key in st.session_state and st.session_state[key] is not None 
                              for key in ['all_claims', 'coherence_results', 'fact_checks', 'documents'])
    
    if not analysis_started:
        # Debug hash loading option
        with st.expander("🔍 Debug: Load from Cache Hash", expanded=False):
            cache_hash = st.text_input("Paste cache hash to load existing analysis", placeholder="e.g., a1b2c3d4e5f6")
            if st.button("Load from Hash", disabled=not cache_hash.strip()):
                # Try to load data from the provided hash
                cache_dir = Path("data/cache") / cache_hash.strip()
                if cache_dir.exists():
                    # Try to load claims first to get document info
                    claims_dir = cache_dir / "claims"
                    if claims_dir.exists():
                        # Load all claims from cache
                        all_claims = []
                        for claim_file in claims_dir.glob("*.json"):
                            claims_data = json.loads(claim_file.read_text())
                            all_claims.extend([ExtractedClaim(**claim) for claim in claims_data])
                        
                        if all_claims:
                            # Store in session state
                            st.session_state.all_claims = all_claims
                            # Create mock documents structure for compatibility
                            docs_by_id = {}
                            for claim in all_claims:
                                if claim.doc_id not in docs_by_id:
                                    docs_by_id[claim.doc_id] = claim.document_text
                            
                            documents = [{"id": doc_id, "text": text} for doc_id, text in docs_by_id.items()]
                            st.session_state.documents = documents
                            st.session_state.num_docs = len(documents)
                            st.session_state.claims_per_doc = len([c for c in all_claims if c.doc_id == documents[0]["id"]])
                            
                            st.success(f"✅ Loaded {len(all_claims)} claims from cache hash {cache_hash.strip()}")
                            st.rerun()
                        else:
                            st.error("No claims found in cache directory")
                    else:
                        st.error("Invalid cache hash - no claims directory found")
                else:
                    st.error("Cache hash not found")
        
        st.markdown("---")
        
        # Configuration
        col1, col2 = st.columns(2)
        with col1:
            num_docs = st.number_input("Number of documents", min_value=1, value=2)
        with col2:
            claims_per_doc = st.number_input("Claims per document", min_value=1, value=5)
        
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
        
        # Store configuration in session state when analysis starts
        st.session_state.num_docs = num_docs
        st.session_state.claims_per_doc = claims_per_doc
        st.session_state.documents = documents
    
    else:
        # We're in analysis mode - retrieve stored configuration
        num_docs = st.session_state.num_docs
        claims_per_doc = st.session_state.claims_per_doc
        documents = st.session_state.documents
        
        # Show analysis header with document info and reset button
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown(f"### 📊 Analysis Results")
            st.markdown(f"**{num_docs} documents • {claims_per_doc} claims each**")
            # Show cache hash for debugging
            from cache_utils import generate_unified_hash_from_config
            current_hash = generate_unified_hash_from_config(documents, claims_per_doc)
            st.code(f"Cache Hash: {current_hash}", language=None)
        with col2:
            if st.button("🔄 New Analysis", type="secondary", use_container_width=True):
                # Clear session state to start fresh
                for key in ['all_claims', 'coherence_results', 'fact_checks', 'num_docs', 'claims_per_doc', 'documents']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Show compact document preview
        with st.expander("📄 Document Summary", expanded=False):
            for i, doc in enumerate(documents):
                preview = doc["text"][:200] + "..." if len(doc["text"]) > 200 else doc["text"]
                st.markdown(f"**{doc['id'].upper()}:** {preview}")
                if i < len(documents) - 1:  # Add separator except for last item
                    st.markdown("---")
        
        # Initialize session state for incremental results
        if 'all_claims' not in st.session_state:
            st.session_state.all_claims = None
        if 'coherence_results' not in st.session_state:
            st.session_state.coherence_results = None
        if 'fact_checks' not in st.session_state:
            st.session_state.fact_checks = None
    
        # Step 1: Extract claims
        if st.session_state.all_claims is None:
            with st.spinner("🔍 Extracting claims from documents..."):
                # Extract claims from all documents using multithreading
                all_claims = extract_claims_for_docs(documents, claims_per_doc)
                
                if not all_claims:
                    st.error("No claims could be extracted from the documents")
                    return
                
                st.session_state.all_claims = all_claims
        
        all_claims = st.session_state.all_claims
    
        # Show extracted claims immediately
        st.markdown("## ✅ Claims Extracted")
        
        # Group claims by document
        claims_by_doc = {}
        for claim in all_claims:
            if claim.doc_id not in claims_by_doc:
                claims_by_doc[claim.doc_id] = []
            claims_by_doc[claim.doc_id].append(claim)
        
        # Display claims grouped by document
        for doc_id, doc_claims in claims_by_doc.items():
            with st.expander(f"📋 {doc_id.upper()} CLAIMS ({len(doc_claims)})", expanded=True):
                for i, claim in enumerate(doc_claims):
                    st.markdown(f"**{i+1}. {claim.claim}**")
                    st.markdown("")
    
        # Step 2: Analyze coherence
        if st.session_state.coherence_results is None:
            with st.spinner("🔗 Analyzing claim coherence..."):
                coherence_results = analyze_coherence(all_claims, documents, claims_per_doc)
                st.session_state.coherence_results = coherence_results
        
        coherence_results = st.session_state.coherence_results
    
        # Show coherence analysis
        st.markdown("## ✅ Coherence Analysis Complete")
        
        with st.expander("🔗 COHERENCE ANALYSIS", expanded=True):
            # Document selection checkboxes
            st.markdown("### Document Selection")
            doc_ids = list(set(claim.doc_id for claim in all_claims))
            doc_ids.sort()  # Sort for consistent ordering
            
            # Create checkboxes for each document (all checked by default)
            selected_docs = []
            cols = st.columns(len(doc_ids))
            for i, doc_id in enumerate(doc_ids):
                with cols[i]:
                    is_selected = st.checkbox(doc_id.upper(), value=True, key=f"coherence_{doc_id}")
                    if is_selected:
                        selected_docs.append(doc_id)
            
            # Filter claims based on selected documents
            if selected_docs:
                filtered_claims = [claim for claim in all_claims if claim.doc_id in selected_docs]
                
                # Filter coherence results to only include relationships between filtered claims
                filtered_claim_indices = set(i for i, claim in enumerate(all_claims) if claim.doc_id in selected_docs)
                filtered_coherence = [
                    c for c in coherence_results 
                    if c.claim_i_idx in filtered_claim_indices and c.claim_j_idx in filtered_claim_indices
                ]
                
                # Calculate metrics with filtered data
                conflict_metrics = get_conflict_metrics(filtered_coherence, filtered_claims)
                load_bearing = get_top_load_bearing_claims_filtered(filtered_coherence, all_claims, filtered_claims, n=3)
                
                st.markdown("### Conflict Metrics")
                st.write(f"- **Prevalence**: {conflict_metrics['conflict_prevalence']:.1%} of relationships are negative")
                st.write(f"- **Avg Intensity**: {conflict_metrics['avg_conflict_intensity']:.2f} (among conflicts)")
                st.write(f"- **Max Conflict**: {conflict_metrics['max_conflict']:.2f} (worst contradiction)")
                
                # Display interactive coherence matrix
                st.markdown("### Coherence Matrix")
                st.markdown("Interactive visualization showing how each claim affects the likelihood of other claims:")
                
                matrix_fig = create_coherence_matrix(filtered_coherence, filtered_claims, all_claims, coherence_results)
                if matrix_fig:
                    st.plotly_chart(matrix_fig, use_container_width=True)
                else:
                    st.info("Matrix visualization requires at least one coherence relationship.")
                
                st.markdown("### Top Load-Bearing Claims")
                for i, claim_info in enumerate(load_bearing[:3]):
                    st.write(f"{i+1}. **{claim_info['doc_id']}[{claim_info['claim_idx']}]** (impact: {claim_info['avg_impact']:.2f})")
                    st.write(f"   _{claim_info['claim']}_")
            else:
                st.warning("Please select at least one document to analyze.")
    
        # Step 3: External validation
        if st.session_state.fact_checks is None:
            with st.spinner("✅ Validating claims against external sources..."):
                fact_checks = check_facts(all_claims, documents, claims_per_doc)
                st.session_state.fact_checks = fact_checks
        
        fact_checks = st.session_state.fact_checks
        fact_summary = get_fact_check_summary(fact_checks)
    
        # Show external validation results
        st.markdown("## ✅ External Validation Complete")
        with st.expander("✅ EXTERNAL VALIDATION", expanded=True):
            # Document selection checkboxes
            st.markdown("### Document Selection")
            doc_ids = list(set(claim.doc_id for claim in all_claims))
            doc_ids.sort()  # Sort for consistent ordering
            
            # Create checkboxes for each document (all checked by default)
            selected_docs_fact = []
            cols = st.columns(len(doc_ids))
            for i, doc_id in enumerate(doc_ids):
                with cols[i]:
                    is_selected = st.checkbox(doc_id.upper(), value=True, key=f"fact_check_{doc_id}")
                    if is_selected:
                        selected_docs_fact.append(doc_id)
            
            # Filter fact checks based on selected documents
            if selected_docs_fact:
                # Filter fact checks to only include those from selected documents
                filtered_fact_checks = [fc for fc in fact_checks if fc.doc_id in selected_docs_fact]
                
                # Calculate summary with filtered data
                fact_summary = get_fact_check_summary(filtered_fact_checks)
                
                # Display validation buckets visualization
                st.markdown("### Validation Distribution")
                st.markdown("Interactive visualization showing claim validation score distribution across buckets:")
                
                veracity_fig = create_veracity_buckets(fact_checks, selected_docs_fact)
                if veracity_fig:
                    st.plotly_chart(veracity_fig, use_container_width=True)
                    
                    # Debug section - can be removed once hover is working
                    with st.expander("🐛 Debug: Hover Data", expanded=False):
                        st.write("If hover isn't working, here's the raw data:")
                        filtered_fact_checks = [fc for fc in fact_checks if fc.doc_id in selected_docs_fact]
                        for fc in filtered_fact_checks[:3]:  # Show first 3 for debugging
                            st.write(f"Score: {fc.veracity}/100 | Doc: {fc.doc_id}")
                            st.write(f"Claim: {fc.claim[:200]}...")
                else:
                    st.info("No fact checks available for visualization.")
                
                st.markdown(f"### Summary")
                st.write(f"**Average Validation Score**: {fact_summary['average_veracity']:.1f}/100")
                
                if fact_summary.get('most_accurate_claims'):
                    st.markdown("### Most Validated Claims")
                    for claim_info in fact_summary['most_accurate_claims'][:3]:
                        st.write(f"**{claim_info['veracity']}/100** - {claim_info['claim']}")
                        st.write(f"_{claim_info['explanation']}_")
                        st.write("")
                
                if fact_summary.get('least_accurate_claims'):
                    st.markdown("### Least Validated Claims")
                    for claim_info in fact_summary['least_accurate_claims'][:3]:
                        st.write(f"**{claim_info['veracity']}/100** - {claim_info['claim']}")
                        st.write(f"_{claim_info['explanation']}_")
                        st.write("")
            else:
                st.warning("Please select at least one document to analyze.")
        

if __name__ == "__main__":
    main()