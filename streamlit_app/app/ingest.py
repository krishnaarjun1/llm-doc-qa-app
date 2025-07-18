import fitz  # PyMuPDF
import io

def extract_text_from_pdf(uploaded_file):
    # Read PDF from uploaded file bytes
    pdf_bytes = uploaded_file.read()
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")

    return "\n".join([page.get_text() for page in doc])

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
