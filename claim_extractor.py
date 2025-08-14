"""Simplified claim extractor - extracts the N most central claims from documents."""

from openai import OpenAI
import json
from pydantic import BaseModel
from pathlib import Path
import hashlib

client = OpenAI()


class ExtractedClaim(BaseModel):
    doc_id: str
    claim_idx: int
    claim: str


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


def extract_claims(doc: str, doc_id: str, n: int = 10) -> list[ExtractedClaim]:
    """Extract the N most central claims from a single document.
    
    Args:
        doc: Document text to analyze
        doc_id: Unique identifier for this document
        n: Number of top claims to extract (default 10)
        
    Returns:
        List of ExtractedClaim objects
    """
    # Get document hash for caching
    doc_hash = hashlib.sha256(doc.encode()).hexdigest()[:12]
    
    # Check cache
    cache_dir = Path("data/cache/claims")
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / f"{doc_hash}_{n}.json"
    
    if cache_file.exists():
        print(f"Loading cached claims for hash {doc_hash}")
        claims_data = json.loads(cache_file.read_text())
        return [ExtractedClaim(**claim) for claim in claims_data]
    
    print(f"Extracting top {n} claims from document '{doc_id}'")
    
    response = client.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "user", "content": PROMPT_CLAIM_EXTRACTION.format(n=n, text=doc)}
        ]
    )
    
    # Debug: print what we got from the API
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
            claim=claim_text
        )
        claims.append(claim)
    
    # Cache results
    cache_file.write_text(json.dumps([claim.model_dump() for claim in claims], indent=2))
    print(f"Extracted {len(claims)} claims, cached to {cache_file}")
    
    return claims