from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import re
import sys
import os
sys.path.insert(0, os.path.abspath('../'))
from search_videos import build_faiss_index, search_videos  # your previously defined function

app = Flask(__name__)
CORS(app)

df_embeddings = pd.read_csv("../videos_topics_embedded.csv")
videos_df = pd.read_csv("../videos.csv")  # Should include columns: id, title, url

faiss_index, _ = build_faiss_index(df_embeddings, debug=False)

def convert_to_embed_url(url: str) -> str:
    match = re.search(r'v=([^&]+)', url)
    if match:
        video_id = match.group(1)
        return f"https://www.youtube.com/embed/{video_id}"
    return url

@app.route("/api/search", methods=["POST"])
def api_search():
    data = request.get_json()
    query = data.get("query", "")
    if not query:
        return jsonify({"error": "Query is required."}), 400

    results_df = search_videos(query, faiss_index, df_embeddings, videos_df, top_k=10, debug=False)
    results_df["embed_url"] = results_df["url"].apply(convert_to_embed_url)
    return jsonify(results_df.to_dict("records"))

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="4999", debug=False)
