import google.generativeai as genai
from dotenv import load_dotenv
load_dotenv()
import os
api_key = os.getenv("GOOGLE_API_KEY")

def ask_llm(question, context):
    prompt = f"Context: {context}\n\nQuestion: {question}\n\nAnswer:"
    # Use the correct model name; "gemini-pro" is commonly available
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(prompt)
    return response.text
