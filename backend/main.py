from fastapi import FastAPI, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import requests
import json
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Review Analyzer API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def query_ollama(prompt: str, max_retries: int = 3):
    """Query Ollama with error handling and retries"""
    for attempt in range(max_retries):
        try:
            response = requests.post(
                "http://localhost:11434/api/generate",
                json={
                    "model": "mistral",
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.1,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
        except requests.exceptions.ConnectionError:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=503,
                    detail="Cannot connect to Ollama. Make sure Ollama is running on localhost:11434"
                )
        except requests.exceptions.Timeout:
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=504,
                    detail="Ollama request timed out. Please try again."
                )
        except Exception as e:
            logger.error(f"Error querying Ollama: {str(e)}")
            if attempt == max_retries - 1:
                raise HTTPException(
                    status_code=500,
                    detail="Error processing request with AI model"
                )
    
@app.get("/")
def read_root():
    return {"message": "Review Analyzer API is running"}

@app.get("/health")
def health_check():
    """Check if Ollama is accessible"""
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        response.raise_for_status()
        return {"status": "healthy", "ollama": "connected"}
    except:
        return {"status": "unhealthy", "ollama": "disconnected"}

@app.post("/analyze/")
def analyze_review(text: str = Form(...)):
    """Analyze a single review for sentiment, topic, and summary"""
    if not text.strip():
        raise HTTPException(status_code=400, detail="Review text cannot be empty")
    
    # Sentiment analysis prompt
    sentiment_prompt = f"""Analyze the sentiment of this product review and respond with only one word: Positive, Neutral, or Negative.

Review: "{text}"

Sentiment:"""

    # Topic detection prompt
    topic_prompt = f"""Identify the main topic or issue discussed in this product review. Choose from these categories or provide a brief 2-3 word description:
- Product Quality
- Delivery/Shipping  
- Price/Value
- Customer Service
- Design/Appearance
- Functionality
- Durability

Review: "{text}"

Main Topic:"""

    # Summary prompt
    summary_prompt = f"""Summarize this product review in one concise sentence (maximum 15 words):

Review: "{text}"

Summary:"""

    try:
        # Get responses from Ollama
        sentiment = query_ollama(sentiment_prompt)
        topic = query_ollama(topic_prompt)
        summary = query_ollama(summary_prompt)
        
        # Clean up responses
        sentiment_clean = sentiment.split('\n')[0].strip()
        topic_clean = topic.split('\n')[0].strip()
        summary_clean = summary.split('\n')[0].strip()
        
        return {
            "sentiment": sentiment_clean,
            "topic": topic_clean,
            "summary": summary_clean
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)