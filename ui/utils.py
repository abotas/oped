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

def get_top_load_bearing_claims(coherence_results, claims, n=3):
    """Get claims with highest total impact (absolute magnitude) on other claims."""
    impact_scores = {}
    impact_counts = {}
    
    for c in coherence_results:
        if c.claim_i_idx not in impact_scores:
            impact_scores[c.claim_i_idx] = 0
            impact_counts[c.claim_i_idx] = 0
        impact_scores[c.claim_i_idx] += abs(c.delta_prob)  # Use absolute magnitude
        impact_counts[c.claim_i_idx] += 1
    
    avg_impact = {idx: impact_scores[idx] / impact_counts[idx] for idx in impact_scores}
    top_indices = sorted(avg_impact.items(), key=lambda x: x[1], reverse=True)[:n]
    
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

def get_top_load_bearing_claims_filtered(coherence_results, all_claims, filtered_claims, n=3):
    """Get claims with highest total impact from filtered coherence results."""
    impact_scores = {}
    impact_counts = {}
    
    for c in coherence_results:
        if c.claim_i_idx not in impact_scores:
            impact_scores[c.claim_i_idx] = 0
            impact_counts[c.claim_i_idx] = 0
        impact_scores[c.claim_i_idx] += abs(c.delta_prob)  # Use absolute magnitude
        impact_counts[c.claim_i_idx] += 1
    
    avg_impact = {idx: impact_scores[idx] / impact_counts[idx] for idx in impact_scores}
    top_indices = sorted(avg_impact.items(), key=lambda x: x[1], reverse=True)[:n]
    
    return [
        {
            "claim": all_claims[idx].claim,
            "doc_id": all_claims[idx].doc_id,
            "doc_title": all_claims[idx].doc_title,
            "claim_idx": all_claims[idx].claim_idx,
            "avg_impact": score,
            "total_impact": impact_scores[idx],
            "num_relationships": impact_counts[idx]
        }
        for idx, score in top_indices
    ]