"""Claim coherence analyzer - evaluates how claims affect each other's likelihood."""

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


class ClaimCoherence(BaseModel):
    claim_i_idx: int
    claim_j_idx: int
    delta_prob: float  # How much more/less likely claim_j is if claim_i is true
    reasoning: str


PROMPT_COHERENCE_ANALYSIS = """You are an expert at analyzing logical relationships between claims.

Given that Claim A is TRUE, evaluate how this affects the likelihood of each other claim being true.

Use a scale from -1.0 to +1.0:
- +1.0: Claim A being true makes the other claim almost certainly true
- +0.5: Claim A being true makes the other claim significantly more likely
- 0.0: Claim A has no effect on the other claim's likelihood
- -0.5: Claim A being true makes the other claim significantly less likely  
- -1.0: Claim A being true makes the other claim almost certainly false

CLAIM A (assume this is TRUE): {claim_a}

OTHER CLAIMS TO EVALUATE:
{other_claims}

Respond with a valid JSON array where each element has:
- "claim_idx": the index of the other claim (0-based)
- "delta_prob": the probability change (-1.0 to +1.0)
- "reasoning": brief explanation of the relationship

ONLY return the JSON array without markdown or extra text:
[{{"claim_idx": 0, "delta_prob": 0.5, "reasoning": "..."}}]
"""


def _analyze_single_claim(work_item: tuple, cache_dir: Path, model: str = "gpt-5-mini") -> list[ClaimCoherence]:
    """Analyze how one claim affects all others. Used for multithreading."""
    i, claim_i, other_claims_text, other_claims = work_item
    
    # Check if this claim's analysis is already cached
    claim_cache_file = cache_dir / f"{claim_i.doc_id}_{claim_i.claim_idx}.json"
    if claim_cache_file.exists():
        print(f"  Loading cached: {claim_i.doc_id}[{claim_i.claim_idx}]")
        cached_data = json.loads(claim_cache_file.read_text())
        return [ClaimCoherence(**item) for item in cached_data]
    
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": PROMPT_COHERENCE_ANALYSIS.format(
                claim_a=claim_i.claim,
                other_claims=other_claims_text
            )}
        ],
    )
    
    # Debug and strip potential markdown
    raw_content = response.choices[0].message.content
    if raw_content.startswith("```"):
        raw_content = raw_content.split("```")[1]
        if raw_content.startswith("json"):
            raw_content = raw_content[4:]
        raw_content = raw_content.strip()
    
    relationships = json.loads(raw_content)
    claim_results = []
    
    # Create mapping from LLM's sequential indices to original claim indices
    llm_to_original_idx = []
    for j in range(len(other_claims) + 1):  # +1 for the original claims count
        if j < i:
            llm_to_original_idx.append(j)  # Claims before claim_i keep their original index
        elif j > i:
            llm_to_original_idx.append(j)  # Claims after claim_i keep their original index
        # Skip j == i since that's the claim being analyzed
    
    for rel in relationships:
        # Map LLM's sequential index to original claim index
        llm_idx = rel["claim_idx"]
        if 0 <= llm_idx < len(llm_to_original_idx):
            actual_j_idx = llm_to_original_idx[llm_idx]
            
            coherence = ClaimCoherence(
                claim_i_idx=i,
                claim_j_idx=actual_j_idx,
                delta_prob=float(rel["delta_prob"]),
                reasoning=rel["reasoning"]
            )
            claim_results.append(coherence)
    
    # Save this claim's results immediately
    claim_cache_file.write_text(json.dumps([c.model_dump() for c in claim_results], indent=2))
    print(f"  Analyzed and saved: {claim_i.doc_id}[{claim_i.claim_idx}]")
    
    return claim_results


def analyze_coherence(claims: list[ExtractedClaim], documents: list[dict], claims_per_doc: int, model: str = "gpt-5-mini", progress_callback: callable = None) -> list[ClaimCoherence]:
    """Analyze coherence between claims - how each claim affects others' likelihood.
    
    Args:
        claims: List of ExtractedClaim objects to analyze
        documents: Original document configuration (for consistent hashing)
        claims_per_doc: Number of claims per document (for consistent hashing)
        model: The model to use for analysis
        progress_callback: Optional callback function(completed, total) for progress updates
        
    Returns:
        List of ClaimCoherence objects showing relationships
    """
    # Generate unified cache key using same method as claim extraction
    unified_hash = generate_unified_hash_from_config(documents, claims_per_doc)
    
    # Setup cache directory for this set of claims
    cache_dir = Path("data/cache") / unified_hash / "coherence"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Analyzing coherence for {len(claims)} claims using {MAX_WORKERS} workers")
    coherence_results = []
    
    # Create work items for each claim
    work_items = []
    for i, claim_i in enumerate(claims):
        # Prepare other claims for analysis (use sequential indices for LLM)
        other_claims = [claim_j for j, claim_j in enumerate(claims) if j != i]
        other_claims_text = "\n".join([
            f"{idx}. {claim.claim}"
            for idx, claim in enumerate(other_claims)
        ])
        
        # Skip if no other claims
        if not other_claims_text:
            continue
        
        work_items.append((i, claim_i, other_claims_text, other_claims))
    
    # Process claims in parallel
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(work_items))) as executor:
        futures = [executor.submit(_analyze_single_claim, work_item, cache_dir, model) for work_item in work_items]
        
        completed = 0
        for future in as_completed(futures):
            claim_relationships = future.result()
            coherence_results.extend(claim_relationships)
            completed += 1
            if completed % 5 == 0 or completed == len(work_items):
                print(f"  Progress: {completed}/{len(work_items)} claims processed")
            if progress_callback:
                progress_callback(completed, len(work_items))
    
    print(f"Analyzed {len(coherence_results)} claim relationships")
    
    return coherence_results


def coherence_to_matrix(coherence_results: list[ClaimCoherence], n_claims: int) -> list[list[float]]:
    """Convert coherence results to an NxN matrix.
    
    Args:
        coherence_results: List of ClaimCoherence objects
        n_claims: Number of claims (size of matrix)
        
    Returns:
        NxN matrix where matrix[i][j] is the delta_prob of claim i on claim j
        Diagonal elements are 1.0 (claim always supports itself)
    """
    # Initialize matrix with zeros
    matrix = [[0.0 for _ in range(n_claims)] for _ in range(n_claims)]
    
    # Set diagonal to 1.0 (claim coherence with itself)
    for i in range(n_claims):
        matrix[i][i] = 1.0
    
    # Fill in the coherence values, with bounds checking
    for c in coherence_results:
        if 0 <= c.claim_i_idx < n_claims and 0 <= c.claim_j_idx < n_claims:
            matrix[c.claim_i_idx][c.claim_j_idx] = c.delta_prob
    
    return matrix


def format_coherence_matrix(coherence_results: list[ClaimCoherence], claims: list) -> str:
    """Format coherence matrix as a clean table string.
    
    Args:
        coherence_results: List of ClaimCoherence objects
        claims: List of claims for labeling
        
    Returns:
        Formatted string representation of the matrix
    """
    matrix = coherence_to_matrix(coherence_results, len(claims))
    
    # Create labels
    labels = [f"{claim.doc_id[:3]}[{claim.claim_idx}]" for claim in claims]
    
    # Calculate column width (max of label width + 2)
    col_width = max(len(label) for label in labels) + 2
    
    lines = []
    lines.append("=== COHERENCE MATRIX ===")
    lines.append("(rows = if claim i is true, columns = effect on claim j)")
    lines.append("Values: -1.0 (contradicts) to +1.0 (strongly supports)")
    lines.append("")
    
    # Header row
    header = " " * (col_width + 2)  # Space for row labels
    for label in labels:
        header += f"{label:>{col_width}}"
    lines.append(header)
    
    # Data rows
    for i, row_label in enumerate(labels):
        row = f"{row_label:>{col_width}}: "
        for j in range(len(claims)):
            val = matrix[i][j]
            row += f"{val:>{col_width}.2f}"
        lines.append(row)
    
    lines.append("")
    lines.append("Note: Diagonal is 1.0 (claims support themselves)")
    lines.append("Legend: doc_id[claim_idx] format")
    
    return "\n".join(lines)