from transformers import pipeline

qa_pipeline = pipeline("text2text-generation", model="mistralai/Mistral-7B-Instruct-v0.2")

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
