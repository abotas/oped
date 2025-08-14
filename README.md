
We're building an oped analyzer

## Op-Ed Analyzer

Analyzes op-ed documents to extract claims, analyze their coherence, and fact-check them.

### Core Functionality

For each document:
- Extract the N most central claims 
- Analyze claim coherence across all documents (how claims affect each other's likelihood)
- External fact checking of all claims
- Generate analysis reports with metrics

### API Structure

**claim_extractor.py**: Extract claims from a single document
```python
extract_claims(doc: str, doc_id: str, n: int = 10) -> list[ExtractedClaim]
```

**claim_coherence.py**: Analyze relationships between claims
```python
analyze_coherence(claims: list[ExtractedClaim]) -> list[ClaimCoherence]
coherence_to_matrix(coherence_results, n_claims) -> list[list[float]]
```

**external_fact_checking.py**: Fact-check claims against external sources
```python
check_facts(claims: list[ExtractedClaim]) -> list[FactCheck]
```

### Multi-Document Workflow

1. Load multiple documents with unique doc_ids
2. Extract N claims from each document individually 
3. Combine all claims for cross-document coherence analysis
4. Fact-check all claims together
5. Generate unified analysis showing relationships across documents

### Data Models

Claims include `doc_id` to track source document, enabling analysis across multiple op-eds while maintaining traceability.

### Caching

All modules use document hashing for idempotent caching - crashes are recoverable without re-running expensive LLM calls.

