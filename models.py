"""Shared data models for the op-ed analyzer."""

from pydantic import BaseModel


class TitledDocument(BaseModel):
    text: str
    id: str  # snake_case ID, max 10 chars
    title: str  # Human-readable title like "1. Machines of Loving Grace"


class ExtractedClaim(BaseModel):
    doc_id: str
    doc_title: str  # Human-readable title
    claim_idx: int
    claim: str
    document_text: str