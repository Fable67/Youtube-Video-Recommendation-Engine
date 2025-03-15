from flask import Flask, render_template, request
import pandas as pd
import re
import sys
import os 
sys.path.insert(0, os.path.abspath('../'))
from search_videos import build_faiss_index, search_videos  # your previously defined function

app = Flask(__name__)

# Load the precomputed embeddings and videos.csv once at startup.
# Ensure that df_embeddings.csv contains at least columns "video_index" and "embedding"
df_embeddings = pd.read_csv("../videos_topics_embedded.csv")
videos_df = pd.read_csv("../videos.csv")  # Should include columns: id, title, url

# Build the FAISS index.
faiss_index, _ = build_faiss_index(df_embeddings, debug=False)

def convert_to_embed_url(url: str) -> str:
    """
    Converts a standard YouTube URL to its embed URL.
    For example: 
      https://www.youtube.com/watch?v=VIDEO_ID  --> https://www.youtube.com/embed/VIDEO_ID
    """
    # Use regex to extract video ID from URL.
    match = re.search(r'v=([^&]+)', url)
    if match:
        video_id = match.group(1)
        return f"https://www.youtube.com/embed/{video_id}"
    return url  # fallback to the original url if pattern not found

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/search", methods=["POST"])
def search():
    query = request.form.get("query")
    if not query:
        return render_template("index.html", error="Please enter a query.")
    
    # Get the top 10 similar videos using your search function.
    results_df = search_videos(query, faiss_index, df_embeddings, videos_df, top_k=10, debug=False)
    
    # Create an embed_url column by converting the original video URL.
    results_df["embed_url"] = results_df["url"].apply(convert_to_embed_url)
    
    results = results_df.to_dict("records")
    return render_template("results.html", query=query, results=results)

if __name__ == "__main__":
    app.run(debug=True)
