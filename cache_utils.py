"""Shared cache utilities for consistent hashing across modules."""

import hashlib
from models import ExtractedClaim



def generate_unified_hash_from_config(documents: list[dict], claims_per_doc: int) -> str:
    """Generate hash from document configuration (for claim_extractor).
    
    Args:
        documents: List of document dicts with 'id' and 'text' fields
        claims_per_doc: Number of claims to extract per document
        
    Returns:
        12-character hash string
    """
    # Create hash input from content hash + claims per doc for each document
    doc_hashes = []
    for doc in sorted(documents, key=lambda x: x['id']):
        content_hash = hashlib.sha256(doc['text'].encode()).hexdigest()[:8]
        doc_hashes.append(f"{content_hash}:{claims_per_doc}")
    
    hash_input = "|".join(doc_hashes)
    return hashlib.sha256(hash_input.encode()).hexdigest()[:12]