"""Simplified claim extractor - extracts the N most central claims from documents."""

from openai import OpenAI
import json
from pathlib import Path
import hashlib
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from cache_utils import generate_unified_hash_from_config
from models import ExtractedClaim, TitledDocument

client = OpenAI()

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


def _extract_claims(titled_doc: TitledDocument, n: int, unified_hash: str, model: str = "gpt-5-mini") -> list[ExtractedClaim]:
    """Extract the N most central claims from a single document.
    
    Args:
        titled_doc: TitledDocument with text, id, and title
        n: Number of top claims to extract
        unified_hash: Pre-computed unified hash for this analysis
        model: Model to use for extraction
        
    Returns:
        List of ExtractedClaim objects
    """
    # Use provided unified cache key
    cache_dir = Path("data/cache") / unified_hash / "claims"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{titled_doc.id}.json"
    
    if cache_file.exists():
        print(f"Loading cached claims for {titled_doc.title}")
        claims_data = json.loads(cache_file.read_text())
        return [ExtractedClaim(**claim) for claim in claims_data]
    
    print(f"Extracting top {n} claims from document '{titled_doc.title}' using model: {model}")
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": PROMPT_CLAIM_EXTRACTION.format(n=n, text=titled_doc.text)}
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
            doc_id=titled_doc.id,
            doc_title=titled_doc.title,
            claim_idx=idx,
            claim=claim_text,
            document_text=titled_doc.text
        )
        claims.append(claim)
    
    # Cache results
    cache_file.write_text(json.dumps([claim.model_dump() for claim in claims], indent=2))
    print(f"Extracted {len(claims)} claims, cached to {cache_file}")
    
    return claims


def extract_claims_for_docs(titled_documents: list[TitledDocument], claims_per_doc: int = 10, model: str = "gpt-5-mini") -> list[ExtractedClaim]:
    """Extract claims from multiple documents using multithreading.
    
    Args:
        titled_documents: List of TitledDocument objects
        claims_per_doc: Number of claims to extract per document
        model: Model to use for extraction
        
    Returns:
        List of ExtractedClaim objects from all documents
    """
    # Convert TitledDocuments to dicts for hashing (backward compatibility)
    documents_for_hash = [{"id": doc.id, "text": doc.text} for doc in titled_documents]
    
    # Generate unified hash for ALL documents (used by all modules)
    unified_hash = generate_unified_hash_from_config(documents_for_hash, claims_per_doc)
    print(f"Using unified hash: {unified_hash}")
    
    all_claims = []
    
    # Use ThreadPoolExecutor for parallel processing
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all extraction tasks
        future_to_doc = {
            executor.submit(_extract_claims, doc, claims_per_doc, unified_hash, model): doc 
            for doc in titled_documents
        }
        
        # Collect results as they complete
        for future in future_to_doc:
            doc = future_to_doc[future]
            claims = future.result()
            all_claims.extend(claims)
    
    return all_claims