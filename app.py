import os
import json
import re
from flask import Flask, request, jsonify, render_template, session
from groq import Groq
from typing import List
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.secret_key = os.urandom(24)  # Required for session management

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """Split text into chunks at sentence boundaries."""
    # Split into sentences (handles both Bengali and English)
    sentences = re.split(r'([।\n]|\.\s)', text)
    
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        if not sentence.strip():
            continue
            
        # Rough estimate of tokens (characters / 3 for Bengali)
        sentence_length = len(sentence) // 3
        
        if current_length + sentence_length > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [sentence]
            current_length = sentence_length
        else:
            current_chunk.append(sentence)
            current_length += sentence_length
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

def find_relevant_chunks(query: str, chunks: List[str], top_k: int = 2) -> str:
    """Find the most relevant chunks for a given query."""
    # Simple keyword matching (can be improved with better similarity measures)
    query_words = set(query.lower().split())
    chunk_scores = []
    
    for chunk in chunks:
        chunk_words = set(chunk.lower().split())
        score = len(query_words.intersection(chunk_words))
        chunk_scores.append((score, chunk))
    
    # Sort by score and get top_k chunks
    relevant_chunks = [chunk for score, chunk in sorted(chunk_scores, reverse=True)[:top_k]]
    return "\n\n".join(relevant_chunks)

# Load configuration
working_dir = os.path.dirname(os.path.abspath(__file__))
config_data = json.load(open(f"{working_dir}/config.json"))
GROQ_API_KEY = config_data["GROQ_API_KEY"]

# Load and chunk FAQ content
def load_and_chunk_faq():
    try:
        with open(r"book.txt", encoding='utf-8') as file:
            content = file.read()
        return chunk_text(content)
    except FileNotFoundError:
        print("FAQ.txt file not found. Please ensure it exists in the working directory.")
        return []

faq_chunks = load_and_chunk_faq()

# Initialize Groq client
os.environ["GROQ_API_KEY"] = GROQ_API_KEY
client = Groq()

def create_system_prompt(query: str) -> str:
    """Create a system prompt using relevant FAQ chunks."""
    relevant_content = find_relevant_chunks(query, faq_chunks)
    return f"""You are a helpful assistant specialized in answering questions about Bengali family health. 
    Answer based on these relevant FAQ sections:

    {relevant_content}

    Guidelines:
    1. Answer based solely on the provided FAQ content
    2. If the information isn't in the provided sections, say so
    3. Respond in the same language as the user's question (Bengali or English)
    4. Keep responses clear and concise
    5. Stay focused on the specific question asked
    6. Maintain conversation context and refer back to previous messages when relevant"""

@app.route('/')
def index():
    # Initialize or reset conversation history when loading the main page
    session['conversation_history'] = []
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_prompt = data.get('query', '')
    
    # Initialize conversation history if it doesn't exist
    if 'conversation_history' not in session:
        session['conversation_history'] = []
    
    # Create system prompt with relevant content
    system_prompt = create_system_prompt(user_prompt)
    
    # Prepare messages with conversation history
    messages = [
        {"role": "system", "content": system_prompt},
    ]
    
    # Add conversation history (limited to last 5 exchanges to manage context window)
    messages.extend(session['conversation_history'][-10:])
    
    # Add current user message
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model="gemma2-9b-it",
            messages=messages,
            temperature=0.7,
            max_tokens=700,
        )

        assistant_response = response.choices[0].message.content
        
        # Update conversation history
        session['conversation_history'].extend([
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": assistant_response}
        ])
        
        # Ensure session is saved
        session.modified = True
        
        return jsonify({
            "response": assistant_response,
            "conversation_history": session['conversation_history']
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/reset', methods=['POST'])
def reset_conversation():
    """Reset the conversation history."""
    session['conversation_history'] = []
    return jsonify({"message": "Conversation history reset successfully"})

if __name__ == '__main__':
    app.run(debug=True)