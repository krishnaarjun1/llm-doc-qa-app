from transformers import pipeline

qa_pipeline = pipeline("summarization", model="google/flan-t5-base")

def build_prompt(context, question):
    return f"""Answer the question using only the context below.

Context:
{context}

Question: {question}
Answer:"""

def answer_question(chunks, question):
    context = "\n".join(chunks)
    prompt = build_prompt(context, question)
    return qa_pipeline(prompt, max_new_tokens=300)[0]['generated_text']
