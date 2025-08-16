"""Analysis pipeline functions with progressive display."""

import streamlit as st
from claim_extractor import extract_claims_for_docs
from claim_coherence import analyze_coherence
from external_fact_checking import check_facts, get_fact_check_summary
from .utils import get_conflict_metrics, get_top_load_bearing_claims_filtered, get_most_contradicted_claims
from doc_titler import title_documents
from .visualizations import create_coherence_matrix, create_veracity_buckets
from .navigation_panel import render_navigation_panel, render_load_bearing_claim, render_contradicted_claim, render_validated_claim, render_load_bearing_document, render_contradicted_document, render_validated_document
from .utils_document_aggregation import get_document_load_bearing_scores, get_document_contradiction_scores, get_document_validation_scores


def run_title_generation_step(documents, claims_per_doc):
    """Step 1: Generate document titles and display them progressively."""
    if st.session_state.titled_documents is None:
        with st.spinner("📝 Generating document titles..."):
            titled_documents = title_documents(documents, claims_per_doc)
            
            if not titled_documents:
                st.error("No document titles could be generated")
                return None
            
            st.session_state.titled_documents = titled_documents
    
    titled_documents = st.session_state.titled_documents
    
    # Show generated titles in compact format
    titles = [doc.title for doc in titled_documents]
    st.markdown(f"**Docs:** {',   '.join(titles)}")
    
    return titled_documents


def run_claim_extraction_step(titled_documents, selected_model, claims_per_doc):
    """Step 1: Extract claims and display them progressively."""
    if st.session_state.all_claims is None:
        with st.spinner("🔍 Extracting claims from documents..."):
            # Extract claims from all documents using multithreading
            all_claims = extract_claims_for_docs(titled_documents, claims_per_doc, model=selected_model)
            
            if not all_claims:
                st.error("No claims could be extracted from the documents")
                return None
            
            st.session_state.all_claims = all_claims
    
    all_claims = st.session_state.all_claims
    
    # Show extracted claims immediately (progressive display)
    st.markdown("## ✅ Claims Extracted")
    
    # Group claims by document title
    claims_by_doc = {}
    for claim in all_claims:
        if claim.doc_title not in claims_by_doc:
            claims_by_doc[claim.doc_title] = []
        claims_by_doc[claim.doc_title].append(claim)
    
    # Display claims grouped by document
    for doc_title, doc_claims in claims_by_doc.items():
        with st.expander(f"📋 {doc_title.upper()} ({len(doc_claims)} claims)", expanded=True):
            for i, claim in enumerate(doc_claims):
                st.markdown(f"**{i+1}. {claim.claim}**")
                st.markdown("")
    
    return all_claims


def run_coherence_analysis_step(all_claims, documents, claims_per_doc, selected_model):
    """Step 2: Analyze coherence and display results progressively."""
    if st.session_state.coherence_results is None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def coherence_progress_callback(completed, total):
            progress = completed / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"🔗 Analyzing coherence: {completed}/{total} claims complete")
        
        status_text.text("🔗 Starting coherence analysis...")
        coherence_results = analyze_coherence(
            all_claims, documents, claims_per_doc, 
            model=selected_model, 
            progress_callback=coherence_progress_callback
        )
        
        progress_bar.empty()
        status_text.empty()
        st.session_state.coherence_results = coherence_results
    
    coherence_results = st.session_state.coherence_results
    
    # Show coherence analysis (progressive display)
    st.markdown("## ✅ Coherence Analysis Complete")
    
    return coherence_results


def render_coherence_section(coherence_results, all_claims):
    """Render the coherence analysis section with document filtering."""
    with st.expander("🔗 COHERENCE ANALYSIS", expanded=True):
        # Document selection checkboxes
        st.markdown("### Document Selection")
        doc_ids = list(set(claim.doc_id for claim in all_claims))
        doc_ids.sort()  # Sort for consistent ordering
        
        # Create mapping from doc_id to doc_title
        doc_id_to_title = {claim.doc_id: claim.doc_title for claim in all_claims}
        
        # Create checkboxes for each document (all checked by default)
        selected_docs = []
        cols = st.columns(len(doc_ids))
        for i, doc_id in enumerate(doc_ids):
            with cols[i]:
                doc_title = doc_id_to_title[doc_id]
                is_selected = st.checkbox(doc_title.upper(), value=True, key=f"coherence_{doc_id}")
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
            load_bearing = get_top_load_bearing_claims_filtered(filtered_coherence, all_claims, filtered_claims)
            contradicted = get_most_contradicted_claims(filtered_coherence, all_claims, filtered_claims)
            
            st.markdown("### Conflict Metrics")
            st.write(f"- **Prevalence**: {conflict_metrics['conflict_prevalence']:.1%} of relationships are negative")
            st.write(f"- **Avg Intensity**: {conflict_metrics['avg_conflict_intensity']:.2f} (among conflicts)")
            st.write(f"- **Max Conflict**: {conflict_metrics['max_conflict']:.2f} (worst contradiction)")
            
            # # Display interactive coherence matrix
            matrix_fig = create_coherence_matrix(filtered_coherence, filtered_claims, all_claims, coherence_results)
            if matrix_fig:
                st.plotly_chart(matrix_fig, use_container_width=True)
            else:
                st.info("Matrix visualization requires at least one coherence relationship.")
            
            # Coherence granularity toggle
            coherence_granularity = st.radio(
                "📊 Analyze by:",
                ["Claims", "Documents"],
                key="coherence_granularity",
                horizontal=True
            )
            
            # Load-bearing panel (claims or documents)
            if coherence_granularity == "Claims":
                render_navigation_panel(
                    load_bearing, 
                    "Most Load-Bearing Claims",
                    "load_bearing_index",
                    render_load_bearing_claim
                )
            else:  # Documents
                doc_load_bearing = get_document_load_bearing_scores(filtered_coherence, all_claims, filtered_claims)
                render_navigation_panel(
                    doc_load_bearing,
                    "Most Load-Bearing Documents",
                    "load_bearing_doc_index",
                    render_load_bearing_document
                )
            
            # Contradicted panel (claims or documents)
            if coherence_granularity == "Claims":
                if contradicted:
                    render_navigation_panel(
                        contradicted,
                        "Most Contradicted Claims", 
                        "contradicted_index",
                        render_contradicted_claim
                    )
                else:
                    st.markdown("### Most Contradicted Claims")
                    st.write("No contradictions found - all claims are mutually supportive or neutral.")
            else:  # Documents
                doc_contradicted = get_document_contradiction_scores(filtered_coherence, all_claims, filtered_claims)
                if doc_contradicted:
                    render_navigation_panel(
                        doc_contradicted,
                        "Most Contradicted Documents",
                        "contradicted_doc_index", 
                        render_contradicted_document
                    )
                else:
                    st.markdown("### Most Contradicted Documents")
                    st.write("No contradictions found - all documents are mutually supportive or neutral.")
        else:
            st.warning("Please select at least one document to analyze.")


def run_fact_checking_step(all_claims, documents, claims_per_doc, selected_model):
    """Step 3: Run fact checking and display results progressively."""
    if st.session_state.fact_checks is None:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def fact_check_progress_callback(completed, total):
            progress = completed / total if total > 0 else 0
            progress_bar.progress(progress)
            status_text.text(f"✅ Running external validation: {completed}/{total} claims complete")
        
        status_text.text("✅ Starting external validation...")
        fact_checks = check_facts(
            all_claims, documents, claims_per_doc, 
            model=selected_model,
            progress_callback=fact_check_progress_callback
        )
        
        progress_bar.empty()
        status_text.empty()
        st.session_state.fact_checks = fact_checks
    
    fact_checks = st.session_state.fact_checks
    
    # Show external validation results (progressive display)
    st.markdown("## ✅ External Validation Complete")
    
    return fact_checks


def render_fact_checking_section(fact_checks, all_claims):
    """Render the fact checking section with document filtering."""
    with st.expander("✅ EXTERNAL VALIDATION", expanded=True):
        # Document selection checkboxes
        st.markdown("### Document Selection")
        doc_ids = list(set(claim.doc_id for claim in all_claims))
        doc_ids.sort()  # Sort for consistent ordering
        
        # Create mapping from doc_id to doc_title
        doc_id_to_title = {claim.doc_id: claim.doc_title for claim in all_claims}
        
        # Create checkboxes for each document (all checked by default)
        selected_docs_fact = []
        cols = st.columns(len(doc_ids))
        for i, doc_id in enumerate(doc_ids):
            with cols[i]:
                doc_title = doc_id_to_title[doc_id]
                is_selected = st.checkbox(doc_title.upper(), value=True, key=f"fact_check_{doc_id}")
                if is_selected:
                    selected_docs_fact.append(doc_id)
        
        # Filter fact checks based on selected documents
        if selected_docs_fact:
            # Filter fact checks to only include those from selected documents
            filtered_fact_checks = [fc for fc in fact_checks if fc.doc_id in selected_docs_fact]
            
            # Calculate summary with filtered data (only used for average score display)
            fact_summary = get_fact_check_summary(filtered_fact_checks)
            
            # # Display validation buckets visualization
            veracity_fig = create_veracity_buckets(fact_checks, selected_docs_fact)
            if veracity_fig:
                st.plotly_chart(veracity_fig, use_container_width=True)
            else:
                st.info("No fact checks available for visualization.")
            
            st.markdown(f"### Summary")
            st.write(f"**Average Validation Score**: {fact_summary['average_veracity']:.1f}/100")
            
            # Validation granularity toggle
            validation_granularity = st.radio(
                "📊 Analyze by:",
                ["Claims", "Documents"],
                key="validation_granularity",
                horizontal=True
            )
            
            # Create full ranked claim lists from ALL filtered fact checks
            if filtered_fact_checks:
                # Sort all filtered fact checks by veracity (highest to lowest)
                all_claims_by_veracity = sorted(filtered_fact_checks, key=lambda x: x.veracity, reverse=True)
                
                # Convert to the format expected by render functions
                most_validated_claims = [
                    {
                        "claim": fc.claim,
                        "veracity": fc.veracity,
                        "explanation": fc.explanation,
                        "doc_title": fc.doc_title,
                        "claim_idx": fc.claim_idx
                    }
                    for fc in all_claims_by_veracity
                ]
                
                # Least validated is the same list reversed (lowest to highest)
                least_validated_claims = list(reversed(most_validated_claims))
            else:
                most_validated_claims = []
                least_validated_claims = []
            
            # Most validated panel (claims or documents)
            if validation_granularity == "Claims":
                if most_validated_claims:
                    render_navigation_panel(
                        most_validated_claims,
                        "Most Validated Claims",
                        "most_validated_index", 
                        render_validated_claim
                    )
            else:  # Documents
                doc_validation = get_document_validation_scores(filtered_fact_checks)
                if doc_validation['most_validated']:
                    render_navigation_panel(
                        doc_validation['most_validated'],
                        "Most Validated Documents",
                        "most_validated_doc_index",
                        render_validated_document
                    )
            
            # Least validated panel (claims or documents)
            if validation_granularity == "Claims":
                if least_validated_claims:
                    render_navigation_panel(
                        least_validated_claims,
                        "Least Validated Claims",
                        "least_validated_index",
                        render_validated_claim
                    )
            else:  # Documents
                doc_validation = get_document_validation_scores(filtered_fact_checks)
                if doc_validation['least_validated']:
                    render_navigation_panel(
                        doc_validation['least_validated'],
                        "Least Validated Documents", 
                        "least_validated_doc_index",
                        render_validated_document
                    )
        else:
            st.warning("Please select at least one document to analyze.")


def run_analysis_pipeline(documents, selected_model, claims_per_doc):
    """Run the complete 4-step analysis pipeline with progressive display."""
    
    # Initialize session state for incremental results
    if 'titled_documents' not in st.session_state:
        st.session_state.titled_documents = None
    if 'all_claims' not in st.session_state:
        st.session_state.all_claims = None
    if 'coherence_results' not in st.session_state:
        st.session_state.coherence_results = None
    if 'fact_checks' not in st.session_state:
        st.session_state.fact_checks = None

    # Step 1: Generate document titles (shows immediately when complete)
    titled_documents = run_title_generation_step(documents, claims_per_doc)
    if not titled_documents:
        return

    # Step 2: Extract claims (shows immediately when complete)  
    all_claims = run_claim_extraction_step(titled_documents, selected_model, claims_per_doc)
    if not all_claims:
        return

    # Step 3: Analyze coherence (shows immediately when complete)
    coherence_results = run_coherence_analysis_step(all_claims, documents, claims_per_doc, selected_model)
    render_coherence_section(coherence_results, all_claims)

    # Step 4: External validation (shows immediately when complete)
    fact_checks = run_fact_checking_step(all_claims, documents, claims_per_doc, selected_model)
    render_fact_checking_section(fact_checks, all_claims)