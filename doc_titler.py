"""Document titler - generates titles and IDs for documents."""

from openai import OpenAI
import re
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from models import TitledDocument
from cache_utils import generate_unified_hash_from_config

load_dotenv()
client = OpenAI()

MAX_WORKERS = 4

PROMPT_TITLE_GENERATION = """Generate a concise, descriptive title for this document that captures its main topic or argument.
The title should be no more than 8 words. If an author name is clearly present, include it in the title.

Document:
{text}"""


def _generate_single_title(doc_text: str, index: int) -> tuple[int, str]:
    """Generate a title for a single document. Returns (index, title) for ordering."""
    # Truncate very long documents for title generation
    truncated_text = doc_text[:3000] if len(doc_text) > 3000 else doc_text
    
    response = client.chat.completions.create(
        model="gpt-5-nano",
        messages=[
            {"role": "user", "content": PROMPT_TITLE_GENERATION.format(text=truncated_text)}
        ]
    )
    
    title = response.choices[0].message.content.strip()
    return (index, title)


def _generate_titles(documents: list[dict]) -> list[str]:
    """Generate titles for documents using gpt-5-nano with multithreading.
    
    Args:
        documents: List of document dicts with 'text' field
        
    Returns:
        List of titled strings like ['1. Title One', '2. Title Two']
    """
    print(f"Generating titles for {len(documents)} documents using gpt-5-nano")
    
    # Process documents in parallel
    with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(documents))) as executor:
        futures = [
            executor.submit(_generate_single_title, doc["text"], i) 
            for i, doc in enumerate(documents)
        ]
        
        # Collect results maintaining order
        results = [future.result() for future in futures]
        results.sort(key=lambda x: x[0])  # Sort by original index
        
    # Add 1-based indexing to titles
    indexed_titles = [f"{i+1}. {title}" for i, (_, title) in enumerate(results)]
    
    print(f"Generated titles: {indexed_titles}")
    return indexed_titles


def _title_to_id(title: str) -> str:
    """Convert a title to a snake_case ID, max 10 chars.
    
    Args:
        title: Human-readable title like "1. Machines of Loving Grace"
        
    Returns:
        Snake-cased ID like "machines_o" (max 10 chars)
    """
    # Remove the index prefix if present (e.g., "1. ")
    title_without_index = re.sub(r'^\d+\.\s*', '', title)
    
    # Remove non-alphanumeric chars and convert to lowercase
    cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', title_without_index)
    
    # Split into words and join with underscores
    words = cleaned.lower().split()
    snake_case = '_'.join(words)
    
    # Truncate to 10 characters
    return snake_case[:10]


def title_documents(documents: list[dict], claims_per_doc: int = 10, topic: str = None) -> list[TitledDocument]:
    """Generate titles and IDs for documents with caching support.
    
    This is the main public interface for the doc_titler module.
    Takes raw documents and returns them with generated titles and IDs.
    Caches results for interruptability and reuse.
    
    Args:
        documents: List of dicts with 'text' field
        claims_per_doc: Number of claims per doc (used for cache key consistency)
        topic: Optional topic for consistent cache hashing
        
    Returns:
        List of TitledDocument objects with:
        - id: snake_case ID (max 10 chars) for internal use
        - title: Human-readable title like "1. Machines of Loving Grace"
        - text: Original document text
    """
    # Generate unified hash for caching (same approach as other modules)
    unified_hash = generate_unified_hash_from_config(
        [{"id": f"doc_{i+1}", "text": doc["text"]} for i, doc in enumerate(documents)], 
        claims_per_doc,
        topic=topic
    )
    
    # Setup cache directory
    cache_dir = Path("data/cache") / unified_hash / "titled_documents"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / "documents.json"
    
    # Check if already cached
    if cache_file.exists():
        print(f"Loading cached titled documents from {cache_file}")
        cached_data = json.loads(cache_file.read_text())
        return [TitledDocument(**doc) for doc in cached_data]
    
    print(f"Generating titles and caching to {cache_file}")
    titles = _generate_titles(documents)
    
    processed_docs = []
    for i, (doc, title) in enumerate(zip(documents, titles)):
        base_id = _title_to_id(title)
        
        # Always prefix with document index (1-based)
        doc_id = f"{i+1}_{base_id[:8]}"  # Format: "1_machines", "2_some_oth", etc.
        
        processed_docs.append(TitledDocument(
            text=doc['text'],
            id=doc_id,
            title=title
        ))
    
    # Cache the results
    cache_file.write_text(json.dumps([doc.model_dump() for doc in processed_docs], indent=2))
    print(f"Cached {len(processed_docs)} titled documents")
    
    return processed_docs