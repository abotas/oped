"""Analysis pipeline functions with progressive display."""

import streamlit as st
from claim_extractor import extract_claims_for_docs
from claim_coherence import analyze_coherence
from external_fact_checking import check_facts, get_fact_check_summary
from .utils import get_conflict_metrics, get_top_load_bearing_claims_filtered, get_most_contradicted_claims
from doc_titler import title_documents
from .visualizations import create_coherence_matrix, create_veracity_buckets


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
    
    # Show generated titles immediately (progressive display)
    st.markdown("## ✅ Document Titles Generated")
    
    for i, doc in enumerate(titled_documents):
        # preview = doc.text[:200] + "..." if len(doc.text) > 200 else doc.text
        st.markdown(f"**{doc.title}**")
        if i < len(titled_documents) - 1:  # Add separator except for last item
            st.markdown("---")
    
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
            load_bearing = get_top_load_bearing_claims_filtered(filtered_coherence, all_claims, filtered_claims, n=3)
            contradicted = get_most_contradicted_claims(filtered_coherence, all_claims, filtered_claims, n=3)
            
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
            st.write('-'*100)
            for i, claim_info in enumerate(load_bearing[:3]):
                st.write(f"**{claim_info['doc_title'].split('.')[1]}** (Doc {claim_info['doc_title'].split('.')[0]})")
                st.write(f"Claim {claim_info['claim_idx'] + 1}: _{claim_info['claim']}_")
                st.write(f"Avg impact: {claim_info['avg_impact']:.2f} ")
                st.write('-'*100)
            
            if contradicted:
                st.markdown("### Most Contradicted Claims")
                st.write('-'*100)
                for i, claim_info in enumerate(contradicted[:3]):
                    st.write(f"**{claim_info['doc_title'].split('.')[1]}** (Doc {claim_info['doc_title'].split('.')[0]})")
                    st.write(f"Claim {claim_info['claim_idx'] + 1}: _{claim_info['claim']}_")
                    st.write(f"({claim_info['num_contradictions']} claims oppose this with an average contradiction of {claim_info['avg_contradiction']:.2f}")
                    st.write('-'*100)
            else:
                st.markdown("### Most Contradicted Claims")
                st.write("No contradictions found - all claims are mutually supportive or neutral.")
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
            
            # Calculate summary with filtered data
            fact_summary = get_fact_check_summary(filtered_fact_checks)
            
            # Display validation buckets visualization
            st.markdown("### Validation Distribution")
            st.markdown("Interactive visualization showing claim validation score distribution across buckets:")
            
            veracity_fig = create_veracity_buckets(fact_checks, selected_docs_fact)
            if veracity_fig:
                st.plotly_chart(veracity_fig, use_container_width=True)
            else:
                st.info("No fact checks available for visualization.")
            
            st.markdown(f"### Summary")
            st.write(f"**Average Validation Score**: {fact_summary['average_veracity']:.1f}/100")
            
            if fact_summary.get('most_accurate_claims'):
                st.markdown("### Most Validated Claims")
                st.write('-'*100)
                for claim_info in fact_summary['most_accurate_claims'][:3]:
                    st.write(f"**{claim_info['doc_title'].split('.')[1]}** (Doc {claim_info['doc_title'].split('.')[0]})")
                    st.write(f"Claim {claim_info['claim_idx'] + 1}: _{claim_info['claim']}_")
                    st.write(f"Validation score: {claim_info['veracity']}/100")
                    st.write(f"_{claim_info['explanation']}_")
                    st.write('-'*100)
            
            if fact_summary.get('least_accurate_claims'):
                st.markdown("### Least Validated Claims")
                st.write('-'*100)
                for claim_info in fact_summary['least_accurate_claims'][:3]:
                    st.write(f"**{claim_info['doc_title'].split('.')[1]}** (Doc {claim_info['doc_title'].split('.')[0]})")
                    st.write(f"Claim {claim_info['claim_idx'] + 1}: _{claim_info['claim']}_")
                    st.write(f"Validation score: {claim_info['veracity']}/100")
                    st.write(f"_{claim_info['explanation']}_")
                    st.write('-'*100)
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