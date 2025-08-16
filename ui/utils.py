"""
UI utility functions for the Op-Ed Analyzer
"""

def get_conflict_metrics(coherence_results, claims):
    """Calculate conflict metrics for coherence analysis."""
    negative_rels = [c for c in coherence_results if c.delta_prob < 0]
    total_rels = len(claims) * (len(claims) - 1)
    
    if not total_rels:
        return {"conflict_prevalence": 0, "avg_conflict_intensity": 0, "max_conflict": 0}
    
    return {
        "conflict_prevalence": len(negative_rels) / total_rels,
        "avg_conflict_intensity": sum(abs(c.delta_prob) for c in negative_rels) / len(negative_rels) if negative_rels else 0,
        "max_conflict": min((c.delta_prob for c in negative_rels), default=0)
    }

def get_most_contradicted_claims(coherence_results, claims):
    """Get claims that are most contradicted by other claims (most negative impact received).
    
    This finds the claims that are made LESS likely when other claims are true.
    These correspond to the columns with the most red in the coherence matrix.
    """
    contradiction_scores = {}  # How much this claim is contradicted by others
    contradiction_counts = {}
    
    for c in coherence_results:
        # claim_j_idx is the TARGET (column in matrix) - the one being affected
        target_idx = c.claim_j_idx
        
        # Only count negative relationships (contradictions)
        if c.delta_prob < 0:
            if target_idx not in contradiction_scores:
                contradiction_scores[target_idx] = 0
                contradiction_counts[target_idx] = 0
            contradiction_scores[target_idx] += abs(c.delta_prob)  # Sum of negative impacts
            contradiction_counts[target_idx] += 1
    
    if not contradiction_scores:
        return []  # No contradictions found
    
    # Calculate average contradiction per relationship
    avg_contradiction = {idx: contradiction_scores[idx] / contradiction_counts[idx] 
                        for idx in contradiction_scores}
    
    # Sort by total contradiction (most contradicted first)
    top_indices = sorted(contradiction_scores.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {
            "claim": claims[idx].claim,
            "doc_id": claims[idx].doc_id,
            "doc_title": claims[idx].doc_title,
            "claim_idx": claims[idx].claim_idx,
            "total_contradiction": score,
            "avg_contradiction": avg_contradiction[idx],
            "num_contradictions": contradiction_counts[idx]
        }
        for idx, score in top_indices
    ]

def get_top_load_bearing_claims_filtered(coherence_results, claims):
    """Get all claims sorted by highest total impact from filtered coherence results."""
    impact_scores = {}
    impact_counts = {}
    
    for c in coherence_results:
        if c.claim_i_idx not in impact_scores:
            impact_scores[c.claim_i_idx] = 0
            impact_counts[c.claim_i_idx] = 0
        impact_scores[c.claim_i_idx] += abs(c.delta_prob)  # Use absolute magnitude
        impact_counts[c.claim_i_idx] += 1
    
    avg_impact = {idx: impact_scores[idx] / impact_counts[idx] for idx in impact_scores}
    top_indices = sorted(avg_impact.items(), key=lambda x: x[1], reverse=True)
    
    return [
        {
            "claim": claims[idx].claim,
            "doc_id": claims[idx].doc_id,
            "doc_title": claims[idx].doc_title,
            "claim_idx": claims[idx].claim_idx,
            "avg_impact": score,
            "total_impact": impact_scores[idx],
            "num_relationships": impact_counts[idx]
        }
        for idx, score in top_indices
    ]
