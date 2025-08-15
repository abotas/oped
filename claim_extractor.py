"""Simplified claim extractor - extracts the N most central claims from documents."""

from openai import OpenAI
import json
from pathlib import Path
import hashlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from cache_utils import generate_unified_hash_from_config
from models import ExtractedClaim

load_dotenv()
client = None

# Configuration
MAX_WORKERS = 4


PROMPT_CLAIM_EXTRACTION = """You are an expert at parsing text for central claims and arguments. Extract the {n} most CENTRAL and IMPORTANT claims from the provided text(s).

A central claim should:
- Be a substantive assertion or argument made in the text
- Contain all necessary context to understand it independently
- Include geographic/jurisdictional scope when relevant
- Include temporal context when relevant  
- Be specific rather than vague (avoid phrases like 'the proposal' - say which proposal)

Focus on the core arguments and assertions rather than minor supporting details.

Respond with a valid JSON matching this schema, where claim_1 is the most central claim, claim_2 is the second most central, etc:
{{"claim_1": "str", "claim_2": "str", ...}}

If there are fewer than {n} substantive claims, return only what you find.
ONLY return the JSON object without markdown or extra text.

Here is the text to parse:

{text}
"""


def _extract_claims(doc: str, doc_id: str, n: int, unified_hash: str) -> list[ExtractedClaim]:
    """Extract the N most central claims from a single document.
    
    Args:
        doc: Document text to analyze
        doc_id: Unique identifier for this document
        n: Number of top claims to extract
        unified_hash: Pre-computed unified hash for this analysis
        
    Returns:
        List of ExtractedClaim objects
    """
    # Use provided unified cache key
    cache_dir = Path("data/cache") / unified_hash / "claims"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{doc_id}.json"
    
    if cache_file.exists():
        print(f"Loading cached claims for {doc_id}")
        claims_data = json.loads(cache_file.read_text())
        return [ExtractedClaim(**claim) for claim in claims_data]
    
    print(f"Extracting top {n} claims from document '{doc_id}'")
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": PROMPT_CLAIM_EXTRACTION.format(n=n, text=doc)}
        ]
    )
    raw_content = response.choices[0].message.content
    print(f"Raw API response: {repr(raw_content[:500])}")  # First 500 chars
    
    # Strip potential markdown formatting
    if raw_content.startswith("```"):
        # Remove markdown code blocks
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()
    
    claims_json = json.loads(raw_content)
    claims = []
    
    for idx, (key, claim_text) in enumerate(claims_json.items()):
        claim = ExtractedClaim(
            doc_id=doc_id,
            claim_idx=idx,
            claim=claim_text,
            document_text=doc
        )
        claims.append(claim)
    
    # Cache results
    cache_file.write_text(json.dumps([claim.model_dump() for claim in claims], indent=2))
    print(f"Extracted {len(claims)} claims, cached to {cache_file}")
    
    return claims


def extract_claims_for_docs(documents: list[dict], claims_per_doc: int = 10) -> list[ExtractedClaim]:
    """Extract claims from multiple documents using multithreading.
    
    Args:
        documents: List of document dicts with 'id' and 'text' fields
        claims_per_doc: Number of claims to extract per document
        
    Returns:
        List of ExtractedClaim objects from all documents
    """
    # Generate unified hash for ALL documents (used by all modules)
    unified_hash = generate_unified_hash_from_config(documents, claims_per_doc)
    print(f"Using unified hash: {unified_hash}")
    
    all_claims = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all extraction tasks
        future_to_doc = {
            executor.submit(_extract_claims, doc["text"], doc["id"], claims_per_doc, unified_hash): doc 
            for doc in documents
        }
        
        # Collect results as they complete
        for future in future_to_doc:
            doc = future_to_doc[future]
            claims = future.result()
            all_claims.extend(claims)
    
    return all_claims