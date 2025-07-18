from transformers import pipeline

# Load QA pipeline with SQuAD2 model
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

def answer_question(chunks, question):
    """
    Uses the top-ranked document chunks and answers the question using Hugging Face QA model.
    """
    # Join all chunks into one large context or top-N relevant ones
    # Ideally, use semantic ranking to select top 3–5 chunks
    context = "\n".join(chunks[:3])  # Use top 3 chunks (assuming pre-ranked)

    # Format input as expected by the QA pipeline
    result = qa_pipeline(question=question, context=context)

    return result["answer"]
