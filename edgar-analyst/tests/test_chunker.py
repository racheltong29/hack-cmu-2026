from edgar_analyst.ingestion.chunker import chunk_document, split_sentences
from edgar_analyst.models import Document


def make_doc(text: str) -> Document:
    return Document(
        doc_id="T-10K-FY1-item7",
        ticker="T",
        form_type="10-K",
        section="Item 7",
        period="FY1",
        text=text,
    )


def test_sentences_are_never_split_mid_number():
    doc = make_doc("Revenue was $20.1 billion in fiscal 2025. Margins improved. " * 40)
    chunks = chunk_document(doc, size=200, overlap=50)
    for c in chunks:
        assert "$20.1 billion" in c.text or "Margins" in c.text
        # No chunk starts mid-sentence with a dangling fragment
        assert c.text[0].isupper() or c.text[0].isdigit() or c.text[0] == "$"


def test_chunk_ids_deterministic_and_metadata_propagates():
    doc = make_doc("First sentence here. Second sentence here. Third sentence here.")
    a = chunk_document(doc, size=40, overlap=0)
    b = chunk_document(doc, size=40, overlap=0)
    assert [c.chunk_id for c in a] == [c.chunk_id for c in b]
    assert all(c.doc_id == "T-10K-FY1-item7" and c.section == "Item 7" for c in a)


def test_overlap_carries_context():
    text = " ".join(f"Sentence number {i} is right here." for i in range(20))
    chunks = chunk_document(make_doc(text), size=150, overlap=60)
    assert len(chunks) > 1
    # Consecutive chunks share at least one sentence
    for prev, nxt in zip(chunks, chunks[1:]):
        prev_sents = set(split_sentences(prev.text))
        nxt_sents = set(split_sentences(nxt.text))
        assert prev_sents & nxt_sents


def test_empty_document_yields_no_chunks():
    assert chunk_document(make_doc("")) == []
