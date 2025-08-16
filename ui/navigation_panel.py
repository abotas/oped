"""Reusable navigation panel component for displaying lists with arrow navigation."""

import streamlit as st


def render_navigation_panel(items: list, title: str, session_key: str, render_item_func: callable):
    """
    Render a navigation panel using native Streamlit components and styling.
    
    Args:
        items: List of items to navigate through
        title: Title to display above the panel
        session_key: Unique session state key for this panel's current index
        render_item_func: Function that takes (item, index) and renders the item content
    """
    if not items:
        st.info(f"**{title}**: No items to display")
        return
    
    # Initialize session state for this panel
    if session_key not in st.session_state:
        st.session_state[session_key] = 0
    
    current_index = st.session_state[session_key]
    total_items = len(items)
    
    # Ensure index is within bounds
    current_index = max(0, min(current_index, total_items - 1))
    st.session_state[session_key] = current_index
    
    # Use Streamlit's subheader for the title
    st.subheader(title)
    
    # Navigation using prev/next buttons only
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("← Prev", 
                    key=f"{session_key}_prev", 
                    disabled=current_index <= 0,
                    use_container_width=True):
            st.session_state[session_key] = current_index - 1
            st.rerun()
    
    with col2:
        st.markdown(f"<div style='text-align: center; padding: 8px 0;'><strong>{current_index + 1} of {len(items)}</strong></div>", 
                   unsafe_allow_html=True)
    
    with col3:
        if st.button("Next →", 
                    key=f"{session_key}_next", 
                    disabled=current_index >= len(items) - 1,
                    use_container_width=True):
            st.session_state[session_key] = current_index + 1
            st.rerun()
    
    # Use Streamlit's container for the content area
    with st.container():
        st.markdown("---")  # Use native divider
        
        # Render the current item
        current_item = items[current_index]
        render_item_func(current_item, current_index)
        
        st.markdown("---")  # Use native divider


def render_load_bearing_claim(claim_info: dict, index: int):
    """Render function for load-bearing claims using native Streamlit components."""
    doc_title = claim_info['doc_title'].split('.')[1] if '.' in claim_info['doc_title'] else claim_info['doc_title']
    
    # Document header
    st.markdown(f"**{doc_title}**")
    
    # Claim content using native info box
    st.info(f"**Claim {claim_info['claim_idx'] + 1}:** {claim_info['claim']}")
    
    # Impact metric using native metric
    st.metric("Load-bearing Impact", f"{claim_info['avg_impact']:.2f}", help="Average impact on other claims")


def render_contradicted_claim(claim_info: dict, index: int):
    """Render function for contradicted claims using native Streamlit components."""
    doc_title = claim_info['doc_title'].split('.')[1] if '.' in claim_info['doc_title'] else claim_info['doc_title']
    
    # Document header
    st.markdown(f"**{doc_title}**")
    
    # Claim content using native warning box
    st.warning(f"**Claim {claim_info['claim_idx'] + 1}:** {claim_info['claim']}")
    
    # Contradiction metrics using native columns
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Opposing Claims", claim_info['num_contradictions'])
    with col2:
        st.metric("Avg Contradiction", f"{claim_info['avg_contradiction']:.2f}")


def render_validated_claim(claim_info: dict, index: int):
    """Render function for validated claims using native Streamlit components."""
    doc_title = claim_info['doc_title'].split('.')[1] if '.' in claim_info['doc_title'] else claim_info['doc_title']
    
    # Document header
    st.markdown(f"**{doc_title}**")
    
    # Choose message type based on veracity score
    if claim_info['veracity'] >= 80:
        st.success(f"**Claim {claim_info['claim_idx'] + 1}:** {claim_info['claim']}")
    elif claim_info['veracity'] >= 60:
        st.warning(f"**Claim {claim_info['claim_idx'] + 1}:** {claim_info['claim']}")
    else:
        st.error(f"**Claim {claim_info['claim_idx'] + 1}:** {claim_info['claim']}")
    
    # Validation score using native metric
    delta = None
    if claim_info['veracity'] >= 80:
        delta = "High confidence"
    elif claim_info['veracity'] >= 60:
        delta = "Medium confidence"
    else:
        delta = "Low confidence"
    
    st.metric("Validation Score", f"{claim_info['veracity']}/100", delta=delta)
    
    # Explanation using native expander
    with st.expander("View Explanation"):
        st.write(claim_info['explanation'])