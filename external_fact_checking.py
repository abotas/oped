"""External fact checking module - verifies claims against external sources."""

from openai import OpenAI
import json
from pydantic import BaseModel
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import ExtractedClaim
from cache_utils import generate_unified_hash_from_config

client = OpenAI()
MAX_WORKERS = 4


class FactCheck(BaseModel):
    doc_id: str
    claim_idx: int
    claim: str
    veracity: int  # 0-100 scale
    explanation: str
    sources: list[str]


PROMPT_FACT_CHECK = """You are an expert fact-checker. You rate the overall veracity of claims using a
scale of 0 (completely inaccurate) to 100 (completely accurate) using your knowledge and trustworthy,
accurate sources you find online. ALWAYS cite sources. You also provide a brief explanation of your
response/reasoning. If the fact mentions the UK, make sure your fact-check specifically pertains to the
UK (not the US). You must respond with a valid JSON matching the schema provided, where veracity
is the 0-100 veracity score for the first task and "explanation" is your concise rationale for the score.
Only return the JSON object without markdown json code block with backticks or extra text before
or after the JSON object: {{"veracity": "int", "explanation": "str"}}. Here is the claim to evaluate: {claim}
"""


def _check_single_claim(claim: ExtractedClaim, cache_dir: Path) -> FactCheck:
    """Check a single claim against external sources. Used for multithreading."""
    # Check if this claim's fact check is already cached
    claim_cache_file = cache_dir / f"{claim.doc_id}_{claim.claim_idx}.json"
    if claim_cache_file.exists():
        print(f"  Loading cached: {claim.doc_id}[{claim.claim_idx}]")
        cached_data = json.loads(claim_cache_file.read_text())
        return FactCheck(**cached_data)
    
    # Use GPT-5 with web search capability using responses API
    response = client.responses.create(
        model="gpt-5",
        tools=[{"type": "web_search_preview"}],
        input=PROMPT_FACT_CHECK.format(claim=claim.claim)
    )
    
    result = json.loads(response.output_text)
    
    fact_check = FactCheck(
        doc_id=claim.doc_id,
        claim_idx=claim.claim_idx,
        claim=claim.claim,
        veracity=int(result["veracity"]),
        explanation=result["explanation"],
        sources=[]  # GPT-5 web search preview doesn't return sources in the same format
    )
    
    # Save this claim's fact check immediately
    claim_cache_file.write_text(json.dumps(fact_check.model_dump(), indent=2))
    print(f"  Checked and saved: {claim.doc_id}[{claim.claim_idx}]")
    
    return fact_check


def check_facts(claims: list[ExtractedClaim], documents: list[dict], claims_per_doc: int) -> list[FactCheck]:
    """Check claims against external sources for accuracy.
    
    Args:
        claims: List of ExtractedClaim objects to verify
        documents: Original document configuration (for consistent hashing)
        claims_per_doc: Number of claims per document (for consistent hashing)
        
    Returns:
        List of FactCheck objects with veracity scores
    """
    # Generate unified cache key using same method as claim extraction
    unified_hash = generate_unified_hash_from_config(documents, claims_per_doc)
    
    # Setup cache directory for this set of claims
    cache_dir = Path("data/cache") / unified_hash / "fact_checks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fact-checking {len(claims)} claims using hash {unified_hash} and {MAX_WORKERS} workers")
    print(f"Cache directory: {cache_dir}")
    fact_checks = []
    
    # Process claims in parallel
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(claims))) as executor:
        futures = [executor.submit(_check_single_claim, claim, cache_dir) for claim in claims]
        
        completed = 0
        for future in as_completed(futures):
            fact_check = future.result()
            fact_checks.append(fact_check)
            completed += 1
            if completed % 5 == 0 or completed == len(claims):
                print(f"  Progress: {completed}/{len(claims)} claims checked")
    
    print(f"Checked {len(fact_checks)} claims")
    
    return fact_checks


def get_fact_check_summary(fact_checks: list[FactCheck]) -> dict:
    """Generate summary statistics from fact checking.
    
    Args:
        fact_checks: List of FactCheck objects
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "average_veracity": 0.0,
        "most_accurate_claims": [],
        "least_accurate_claims": [],
        "claims_at_odds_with_reality": []
    }
    
    if not fact_checks:
        return summary
    
    # Calculate average veracity
    summary["average_veracity"] = sum(fc.veracity for fc in fact_checks) / len(fact_checks)
    
    # Sort by veracity
    sorted_checks = sorted(fact_checks, key=lambda x: x.veracity, reverse=True)
    
    # Most accurate claims (top 3, regardless of absolute score)
    summary["most_accurate_claims"] = [
        {
            "claim": fc.claim,
            "veracity": fc.veracity,
            "explanation": fc.explanation
        }
        for fc in sorted_checks[:3]  # Top 3 items (highest scores)
    ]
    
    # Least accurate claims (bottom 3, regardless of absolute score)
    summary["least_accurate_claims"] = [
        {
            "claim": fc.claim,
            "veracity": fc.veracity,
            "explanation": fc.explanation
        }
        for fc in sorted_checks[-3:]  # Last 3 items (lowest scores)
    ][::-1]  # Reverse to show lowest first
    
    # Claims most at odds with reality (veracity < 50)
    summary["claims_at_odds_with_reality"] = [
        {
            "claim": fc.claim,
            "veracity": fc.veracity,
            "explanation": fc.explanation,
            "sources": fc.sources
        }
        for fc in fact_checks if fc.veracity < 50
    ]
    
    return summary