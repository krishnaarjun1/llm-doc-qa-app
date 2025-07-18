from transformers import pipeline

# Use the correct pipeline type
qa_pipeline = pipeline("text2text-generation", model="google/flan-t5-base")

def build_prompt(context, question):
    return f"""Answer the question using only the context below.

Context:
{context}

Question: {question}
Answer:"""

def answer_question(chunks, question):
    context = "\n".join(chunks)
    prompt = build_prompt(context, question)
    result = qa_pipeline(prompt, max_new_tokens=300)

    # Defensive return (some versions may return 'text' instead of 'generated_text')
    return result[0].get('generated_text') or result[0].get('text', '[No answer generated]')
