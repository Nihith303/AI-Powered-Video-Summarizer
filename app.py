import os
import re
import json
import time
import tempfile
import speech_recognition as sr
from typing import List, Tuple, Optional
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename

import numpy as np
import faiss
from moviepy.editor import VideoFileClip
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

app = Flask(__name__)

# Load environment variables from .env if present
load_dotenv()

# ---------------------------
# Configuration
# ---------------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
# Allow overriding model slugs via env vars (provide sensible defaults)
QWEN_MODEL = os.getenv("QWEN_MODEL", "deepseek/deepseek-r1-distill-llama-70b:free")
LLAMA_MODEL = os.getenv("LLAMA_MODEL", "meta-llama/llama-3.3-70b-instruct:free")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCHgW4Ho5yCXCpcZ5ZVtEu1lV9XJqTwTDs")

# File upload configuration
UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm', 'wmv', 'flv'}
MAX_CONTENT_LENGTH = 100 * 1024 * 1024  # 100MB max file size

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

def allowed_file(filename: str) -> bool:
    """Check if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path: str) -> str:
    """Extract audio from video file and return the path to the audio file."""
    video = VideoFileClip(video_path)
    audio_path = f"{video_path}.wav"
    video.audio.write_audiofile(audio_path, logger=None)
    video.close()
    return audio_path

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using speech recognition."""
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio_data = recognizer.record(source)
        try:
            text = recognizer.recognize_google(audio_data)
            return text
        except sr.UnknownValueError:
            return ""
        except sr.RequestError as e:
            raise Exception(f"Could not request results; {e}")

def process_video_file(video_path: str) -> str:
    """Process video file and return extracted text."""
    try:
        # Extract audio from video
        audio_path = extract_audio(video_path)
        
        # Transcribe audio to text
        text = transcribe_audio(audio_path)
        
        # Clean up temporary files
        try:
            os.remove(audio_path)
        except OSError:
            pass
            
        return text
    except Exception as e:
        raise Exception(f"Error processing video: {str(e)}")
    finally:
        # Clean up video file
        try:
            os.remove(video_path)
        except OSError:
            pass

# ---------------------------
# Embedding and Vector Index (in-memory for demo)
# ---------------------------
_embed_model = SentenceTransformer("all-MiniLM-L6-v2")
_faiss_index = None  # type: ignore
_chunks: List[str] = []
_chunk_embeddings = None


# ---------------------------
# Helpers
# ---------------------------

def extract_youtube_id(url: str) -> str:
    patterns = [
        r"v=([\w-]{11})",
        r"youtu\.be/([\w-]{11})",
        r"youtube\.com/embed/([\w-]{11})",
    ]
    for p in patterns:
        m = re.search(p, url)
        if m:
            return m.group(1)
    # Fallback: last 11 chars if looks like an ID
    if len(url) >= 11 and re.match(r"^[\w-]{11}$", url[-11:]):
        return url[-11:]
    raise ValueError("Could not parse YouTube video ID from URL")


def fetch_transcript(video_id: str) -> str:
    """
    Fetches video details (snippet info) from YouTube using the Data API.
    
    Args:
        video_id (str): The ID of the YouTube video (e.g., 'AXOjZIFOTeE')
    
    Returns:
        dict: The snippet information if successful, otherwise None.
    """
    try:
        youtube = build("youtube", "v3", developerKey=GEMINI_API_KEY)
        request = youtube.videos().list(
            part="snippet",
            id=video_id
        )
        response = request.execute()

        # If response contains data
        if "items" in response and len(response["items"]) > 0:
            description = response["items"][0]["snippet"]["description"]
            return description
        else:
            print("⚠️ No video details found for the given ID.")
            return None

    except HttpError as e:
        print(f"❌ An HTTP error occurred: {e}")
        return None

    except Exception as e:
        print(f"⚠️ An unexpected error occurred: {e}")
        return None


def chunk_text(text: str, max_tokens: int = 400) -> List[str]:
    # Approximate tokens as words; 1 token ~ 0.75 words; keep simple here
    words = text.split()
    chunk_size = max_tokens
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def build_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, np.ndarray]:
    if not chunks:
        raise ValueError("No chunks to index")
    embeddings = _embed_model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    return index, embeddings


def retrieve(query: str, top_k: int = 5) -> List[Tuple[int, float]]:
    global _faiss_index, _chunks
    if _faiss_index is None:
        return []
    q_emb = _embed_model.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    scores, ids = _faiss_index.search(q_emb, top_k)
    results = []
    for i, s in zip(ids[0], scores[0]):
        if int(i) >= 0 and int(i) < len(_chunks):
            results.append((int(i), float(s)))
    return results


def openrouter_chat(model: str, messages: List[dict], temperature: float = 0.2, max_tokens: int = 800) -> str:
    if not OPENROUTER_API_KEY:
        raise RuntimeError("OPENROUTER_API_KEY not set in environment")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "http://localhost/",
        "X-Title": "YouTube Transcript Summarizer",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=120)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenRouter error {resp.status_code}: {resp.text[:500]}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return json.dumps(data)[:2000]


# ---------------------------
# Routes
# ---------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/favicon.ico')
def favicon():
    return ("", 204)


@app.route("/process", methods=["POST"])
def process():
    global _faiss_index, _chunks, _chunk_embeddings
    data = request.get_json(force=True)
    url = data.get("url", "").strip()
    if not url:
        return jsonify({"error": "Missing YouTube URL"}), 400
    try:
        vid = extract_youtube_id(url)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

    try:
        transcript = fetch_transcript(vid)
    except RuntimeError as e:
        # Known fetch issues (disabled/none/parse)
        return jsonify({"error": str(e)}), 502
    except Exception as e:
        return jsonify({"error": f"Unexpected error fetching transcript: {str(e)}"}), 500

    if not transcript.strip():
        return jsonify({"error": "Transcript not available for this video."}), 404

    try:
        _chunks = chunk_text(transcript, max_tokens=400)
        _faiss_index, _chunk_embeddings = build_index(_chunks)
    except Exception as e:
        return jsonify({"error": f"Index build failed: {str(e)}"}), 500

    return jsonify({
        "message": "Transcript processed and vector index built.",
        "chunks": len(_chunks)
    })


@app.route("/process_text", methods=["POST"])
def process_text():
    global _faiss_index, _chunks, _chunk_embeddings
    
    data = request.get_json()
    text = data.get('text', '').strip()
    
    if not text:
        return jsonify({'error': 'No text provided'}), 400
    
    try:
        # Split text into chunks
        _chunks = chunk_text(text)
        
        # Build index
        _faiss_index, _chunk_embeddings = build_index(_chunks)
        
        return jsonify({
            'status': 'success',
            'chunks': len(_chunks)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route("/upload_video", methods=["POST"])
def upload_video():
    global _chunks, _faiss_index, _chunk_embeddings
    
    # Check if the post request has the file part
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    
    # If user does not select file, browser also
    # submit an empty part without filename
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if not file or not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400
    
    try:
        # Save the uploaded file temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, filename)
        file.save(video_path)
        
        # Process the video file (extract audio and transcribe)
        text = process_video_file(video_path)
        
        if not text:
            return jsonify({'error': 'Could not extract text from video'}), 400
        
        # Split text into chunks
        _chunks = chunk_text(text)
        
        # Build index
        _faiss_index, _chunk_embeddings = build_index(_chunks)
        
        return jsonify({
            'status': 'success',
            'chunks': len(_chunks)
        })
        
    except Exception as e:
        app.logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': f'Error processing video: {str(e)}'}), 500
    finally:
        # Clean up the temporary directory
        try:
            if 'temp_dir' in locals():
                for file in os.listdir(temp_dir):
                    os.remove(os.path.join(temp_dir, file))
                os.rmdir(temp_dir)
        except Exception as e:
            app.logger.error(f"Error cleaning up temp files: {str(e)}")


@app.route("/summarize", methods=["POST"])
def summarize():
    if not _chunks:
        return jsonify({"error": "No transcript processed yet."}), 400

    # Map: summarize chunks individually (batching naive loop)
    partial_summaries = []
    for ch in _chunks:
        prompt = [
            {"role": "system", "content": "You are a concise expert summarizer. Produce short, faithful summaries."},
            {"role": "user", "content": f"Summarize the following transcript segment in 3-5 bullet points.\n\nSegment:\n{ch}"}
        ]
        try:
            s = openrouter_chat(LLAMA_MODEL, prompt, temperature=0.2, max_tokens=300)
        except Exception as e:
            return jsonify({"error": f"Summarization failed: {str(e)}"}), 500
        partial_summaries.append(s)
        # light throttling to be polite
        time.sleep(0.2)

    # Reduce: combine partial summaries
    combined_text = "\n\n".join(partial_summaries)
    reduce_prompt = [
        {"role": "system", "content": "You are a world-class editor. Create a single cohesive summary."},
        {"role": "user", "content": f"Combine the following bullet-point summaries into one clear summary with sections: Overview, Key Points, and Takeaways. Keep it under 250 words.\n\n{combined_text}"}
    ]
    try:
        final_summary = openrouter_chat(LLAMA_MODEL, reduce_prompt, temperature=0.2, max_tokens=350)
    except Exception as e:
        return jsonify({"error": f"Final summarization failed: {str(e)}"}), 500

    return jsonify({"summary": final_summary})


@app.route("/ask", methods=["POST"])
def ask():
    if _faiss_index is None or not _chunks:
        return jsonify({"error": "No transcript processed yet."}), 400
    data = request.get_json(force=True)
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "Missing question"}), 400

    hits = retrieve(question, top_k=5)
    context_blocks = []
    for idx, score in hits:
        context_blocks.append(f"[Chunk {idx} | score={score:.3f}]\n{_chunks[idx]}")
    context = "\n\n".join(context_blocks)

    messages = [
        {"role": "system", "content": "You are Q&A assistant. Only use the provided context. If absent, say you don't know."},
        {"role": "user", "content": f"Answer the question strictly based on the context.\n\nContext:\n{context}\n\nQuestion: {question}"}
    ]
    try:
        answer = openrouter_chat(QWEN_MODEL, messages, temperature=0.1, max_tokens=600)
    except Exception as e:
        return jsonify({"error": f"Q&A failed: {str(e)}"}), 500

    return jsonify({"answer": answer, "sources": [i for i, _ in hits]})


if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
