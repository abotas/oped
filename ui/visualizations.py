"""Visualization functions for the Op-Ed Analyzer UI."""

import plotly.graph_objects as go
import textwrap
from claim_coherence import ClaimCoherence, coherence_to_matrix


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
            remapped_c = ClaimCoherence(
                claim_i_idx=original_to_filtered[c.claim_i_idx],
                claim_j_idx=original_to_filtered[c.claim_j_idx],
                delta_prob=c.delta_prob,
                reasoning=c.reasoning
            )
            remapped_coherence.append(remapped_c)
    
    # Generate the matrix with remapped coherence
    matrix = coherence_to_matrix(remapped_coherence, len(claims))
    
    # Group claims by document
    doc_groups = {}
    doc_titles = {}
    
    for i, claim in enumerate(claims):
        if claim.doc_id not in doc_groups:
            doc_groups[claim.doc_id] = []
            doc_titles[claim.doc_id] = f"Doc {claim.doc_title[:20]}..."
        doc_groups[claim.doc_id].append(i)
    
    # Create tick positions and labels for document groups
    tick_positions = []
    tick_labels = []
    
    current_pos = 0
    for doc_id in sorted(doc_groups.keys()):
        group_size = len(doc_groups[doc_id])
        # Position label at the center of the document group
        center_pos = current_pos + (group_size - 1) / 2
        tick_positions.append(center_pos)
        tick_labels.append(doc_titles[doc_id])
        current_pos += group_size
    
    # Create temporary claim labels for hover text (not used for axes)
    temp_claim_labels = []
    for i, claim in enumerate(claims):
        claim_num = len([c for c in claims[:i+1] if c.doc_id == claim.doc_id])
        title_num = claim.doc_title.split('.')[0] if '.' in claim.doc_title else claim.doc_title[:3]
        temp_claim_labels.append(f"{title_num}[{claim_num-1}]")
    
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
                    f"<b>Source Claim A ({temp_claim_labels[i]}):</b><br>"
                    f"{claim_i_wrapped}<br><br>"
                    f"<b>→ Target Claim B ({temp_claim_labels[j]}):</b><br>"
                    f"{claim_j_wrapped}<br><br>"
                    f"<b>Reasoning:</b><br>"
                    f"{reasoning_wrapped}"
                )
            hover_row.append(hover_text_cell)
        hover_text.append(hover_row)
    
    # Create the heatmap with no initial tick labels (we'll set them manually)
    fig = go.Figure(data=go.Heatmap(
        z=matrix,
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
        showscale=False
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
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            # tickangle=45,
            tickfont=dict(size=12, color="white"),
            gridcolor="rgba(255,255,255,0.3)",
            linecolor="rgba(255,255,255,0.5)"
        ),
        yaxis=dict(
            autorange='reversed',  # Reverse y-axis to match matrix convention
            tickmode='array',
            tickvals=tick_positions,
            ticktext=tick_labels,
            tickangle=45,
            tickfont=dict(size=12, color="white"),
            gridcolor="rgba(255,255,255,0.3)", 
            linecolor="rgba(255,255,255,0.5)"
        ),
        plot_bgcolor='rgba(255,255,255,0.05)',
        paper_bgcolor='rgba(255,255,255,0.05)',
        font=dict(color="white"),
        margin=dict(l=80, r=80, t=80, b=80)
    )
    
    return fig