"""Document titler - generates titles and IDs for documents."""

from openai import OpenAI
import re
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv
from models import TitledDocument

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


def title_documents(documents: list[dict]) -> list[TitledDocument]:
    """Generate titles and IDs for documents.
    
    This is the main public interface for the doc_titler module.
    Takes raw documents and returns them with generated titles and IDs.
    
    Args:
        documents: List of dicts with 'text' field
        
    Returns:
        List of TitledDocument objects with:
        - id: snake_case ID (max 10 chars) for internal use
        - title: Human-readable title like "1. Machines of Loving Grace"
        - text: Original document text
    """
    titles = _generate_titles(documents)
    
    processed_docs = []
    for i, (doc, title) in enumerate(zip(documents, titles)):
        doc_id = _title_to_id(title)
        
        # Ensure unique IDs by appending index if needed
        used_ids = {d.id for d in processed_docs}
        if doc_id in used_ids:
            doc_id = f"{doc_id[:8]}_{i+1}"  # Leave room for underscore and number
        
        processed_docs.append(TitledDocument(
            text=doc['text'],
            id=doc_id,
            title=title
        ))
    
    return processed_docs