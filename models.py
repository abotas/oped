"""Shared data models for the op-ed analyzer."""

from pydantic import BaseModel, Field
import hashlib


class TitledDocument(BaseModel):
    id: str  # snake_case ID, max 10 chars
    title: str  # Human-readable title like "1. Machines of Loving Grace"


class ExtractedClaim(BaseModel):
    doc_id: str
    doc_title: str  # Human-readable title
    claim_idx: int
    claim: str


class InputConfig(BaseModel):
    """Configuration for an analysis run. Immutable and hashable."""
    
    documents: list[dict]
    claims_per_doc: int
    topic: str | None = None
    model: str = "gpt-5-mini"
    
    class Config:
        frozen = True  # Make immutable
    
    def generate_cache_hash(self) -> str:
        """Generate unified hash for this analysis configuration.
        
        Uses all InputConfig fields for deterministic, unique hashing.
        
        Returns:
            12-character hash string
        """
        # Create hash input from all config fields
        hash_components = []
        
        # Add documents (index + first 100 chars of each)
        doc_hashes = []
        for i, doc in enumerate(self.documents):
            text_preview = doc['text'][:100]
            content_hash = hashlib.sha256(text_preview.encode()).hexdigest()[:8]
            doc_hashes.append(f"{i}:{content_hash}")
        hash_components.append("|".join(doc_hashes))
        
        # Add claims_per_doc (once, not per document)
        hash_components.append(f"claims:{self.claims_per_doc}")
        
        # Add model
        hash_components.append(f"model:{self.model}")
        
        # Add topic if provided
        if self.topic:
            topic_hash = hashlib.sha256(self.topic.encode()).hexdigest()[:8]
            hash_components.append(f"topic:{topic_hash}")
        
        hash_input = "|".join(hash_components)
        return hashlib.sha256(hash_input.encode()).hexdigest()[:12]
    
    def save_to_cache(self) -> str:
        """Save this config to cache directory and return the cache hash.
        
        Returns:
            Cache hash for this configuration
        """
        from pathlib import Path
        import json
        
        cache_hash = self.generate_cache_hash()
        cache_dir = Path("data/cache") / cache_hash
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        config_file = cache_dir / "analysis_config.json"
        config_file.write_text(self.model_dump_json(indent=2))
        
        return cache_hash
    
    @classmethod
    def load_from_cache(cls, cache_hash: str) -> "InputConfig":
        """Load config from cache directory.
        
        Args:
            cache_hash: The cache hash to load from
            
        Returns:
            InputConfig object
        """
        from pathlib import Path
        import json
        
        config_file = Path("data/cache") / cache_hash / "analysis_config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No config found for cache hash {cache_hash}")
        
        config_data = json.loads(config_file.read_text())
        return cls(**config_data)