"""Document-level aggregation functions for the Op-Ed Analyzer UI."""


def get_document_load_bearing_scores(coherence_results, all_claims, filtered_claims):
    """Get documents ranked by average load-bearing impact of their claims."""
    from .utils import get_top_load_bearing_claims_filtered
    
    # First get all claim scores
    claim_scores = get_top_load_bearing_claims_filtered(coherence_results, all_claims, filtered_claims)
    
    # Group by document and calculate averages
    doc_scores = {}
    doc_counts = {}
    
    for claim_info in claim_scores:
        doc_title = claim_info['doc_title']
        if doc_title not in doc_scores:
            doc_scores[doc_title] = 0
            doc_counts[doc_title] = 0
        doc_scores[doc_title] += claim_info['avg_impact']
        doc_counts[doc_title] += 1
    
    # Calculate average scores per document
    doc_averages = []
    for doc_title in doc_scores:
        avg_score = doc_scores[doc_title] / doc_counts[doc_title]
        doc_averages.append({
            'doc_title': doc_title,
            'avg_impact': avg_score,
            'num_claims': doc_counts[doc_title]
        })
    
    # Sort by average impact
    doc_averages.sort(key=lambda x: x['avg_impact'], reverse=True)
    return doc_averages


def get_document_contradiction_scores(coherence_results, all_claims, filtered_claims):
    """Get documents ranked by average contradiction of their claims."""
    from .utils import get_most_contradicted_claims
    
    # First get all claim scores
    claim_scores = get_most_contradicted_claims(coherence_results, all_claims, filtered_claims)
    
    if not claim_scores:
        return []
    
    # Group by document and calculate averages
    doc_scores = {}
    doc_counts = {}
    
    for claim_info in claim_scores:
        doc_title = claim_info['doc_title']
        if doc_title not in doc_scores:
            doc_scores[doc_title] = 0
            doc_counts[doc_title] = 0
        doc_scores[doc_title] += claim_info['avg_contradiction']
        doc_counts[doc_title] += 1
    
    # Calculate average scores per document
    doc_averages = []
    for doc_title in doc_scores:
        avg_score = doc_scores[doc_title] / doc_counts[doc_title]
        doc_averages.append({
            'doc_title': doc_title,
            'avg_contradiction': avg_score,
            'num_claims': doc_counts[doc_title]
        })
    
    # Sort by average contradiction
    doc_averages.sort(key=lambda x: x['avg_contradiction'], reverse=True)
    return doc_averages


def get_document_validation_scores(fact_checks):
    """Get documents ranked by average validation scores using ALL fact checks."""
    # Group ALL fact checks by document
    doc_scores = {}
    doc_counts = {}
    
    # Process ALL fact checks (not just the summary ones)
    for fact_check in fact_checks:
        doc_title = fact_check.doc_title
        if doc_title not in doc_scores:
            doc_scores[doc_title] = 0
            doc_counts[doc_title] = 0
        doc_scores[doc_title] += fact_check.veracity
        doc_counts[doc_title] += 1
    
    # Calculate average scores per document
    doc_averages = []
    for doc_title in doc_scores:
        avg_score = doc_scores[doc_title] / doc_counts[doc_title]
        doc_averages.append({
            'doc_title': doc_title,
            'avg_veracity': avg_score,
            'num_claims': doc_counts[doc_title]
        })
    
    # Return both most and least validated versions
    most_validated = sorted(doc_averages, key=lambda x: x['avg_veracity'], reverse=True)
    least_validated = sorted(doc_averages, key=lambda x: x['avg_veracity'])
    
    return {
        'most_validated': most_validated,
        'least_validated': least_validated
    }