"""External fact checking module - verifies claims against external sources."""

from openai import OpenAI
import json
from pydantic import BaseModel
from pathlib import Path
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from models import ExtractedClaim, InputConfig

client = OpenAI()
MAX_WORKERS = 4


class FactCheck(BaseModel):
    doc_id: str
    doc_title: str  # Human-readable title
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


def _check_single_claim(claim: ExtractedClaim, cache_dir: Path, model: str = "gpt-5-mini") -> FactCheck:
    """Check a single claim against external sources. Used for multithreading."""
    # Check if this claim's fact check is already cached
    claim_cache_file = cache_dir / f"{claim.doc_id}_{claim.claim_idx}.json"
    if claim_cache_file.exists():
        print(f"  Loading cached: {claim.doc_id}[{claim.claim_idx}]")
        cached_data = json.loads(claim_cache_file.read_text())
        return FactCheck(**cached_data)
    
    # Use selected model with web search capability using responses API
    response = client.responses.create(
        model=model,
        tools=[{"type": "web_search_preview"}],
        input=PROMPT_FACT_CHECK.format(claim=claim.claim)
    )
    
    result = json.loads(response.output_text)
    
    fact_check = FactCheck(
        doc_id=claim.doc_id,
        doc_title=claim.doc_title,
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


def check_facts(claims: list[ExtractedClaim], config: InputConfig, progress_callback: callable = None) -> list[FactCheck]:
    """Check claims against external sources using config object.
    
    Args:
        claims: List of ExtractedClaim objects to verify
        config: AnalysisConfig object with all parameters
        progress_callback: Optional callback function(completed, total) for progress updates
        
    Returns:
        List of FactCheck objects with veracity scores
    """
    # Generate unified cache key from config
    unified_hash = config.generate_cache_hash()
    
    # Setup cache directory for this set of claims
    cache_dir = Path("data/cache") / unified_hash / "fact_checks"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Fact-checking {len(claims)} claims using hash {unified_hash} and {MAX_WORKERS} workers")
    print(f"Cache directory: {cache_dir}")
    fact_checks = []
    
    # Process claims in parallel
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(claims))) as executor:
        futures = [executor.submit(_check_single_claim, claim, cache_dir, config.model) for claim in claims]
        
        completed = 0
        for future in as_completed(futures):
            fact_check = future.result()
            fact_checks.append(fact_check)
            completed += 1
            if completed % 5 == 0 or completed == len(claims):
                print(f"  Progress: {completed}/{len(claims)} claims checked")
            if progress_callback:
                progress_callback(completed, len(claims))
    
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
    
    # Most accurate claims (all claims sorted by highest veracity)
    summary["most_accurate_claims"] = [
        {
            "claim": fc.claim,
            "veracity": fc.veracity,
            "explanation": fc.explanation,
            "doc_title": fc.doc_title,
            "claim_idx": fc.claim_idx
        }
        for fc in sorted_checks  # All claims from highest to lowest score
    ]
    
    # Least accurate claims (all claims sorted by lowest veracity)
    summary["least_accurate_claims"] = [
        {
            "claim": fc.claim,
            "veracity": fc.veracity,
            "explanation": fc.explanation,
            "doc_title": fc.doc_title,
            "claim_idx": fc.claim_idx
        }
        for fc in reversed(sorted_checks)  # All claims from lowest to highest score
    ]
    
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