"""Shared data models for the op-ed analyzer."""

from pydantic import BaseModel


class ExtractedClaim(BaseModel):
    doc_id: str
    claim_idx: int
    claim: str
    document_text: str