import numpy as np
import faiss
import pandas as pd
import ast
import os
from openai import OpenAI

client = OpenAI()

def embed_query(query: str, model: str = "text-embedding-3-large", debug: bool = False) -> np.ndarray:
    """
    Embeds the query string using the OpenAI embedding API.
    Returns a normalized numpy array.
    """
    response = client.embeddings.create(input=query,
    model=model)
    embedding = np.array(response.data[0].embedding, dtype=np.float32)
    if debug:
        print("Query embedding shape:", embedding.shape)
    # Normalize the embedding for cosine similarity.
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding /= norm
    return embedding

def build_faiss_index(df_embeddings: pd.DataFrame, debug: bool = False) -> (faiss.IndexFlatIP, np.ndarray):
    """
    Builds a FAISS index from the embeddings stored in df_embeddings.
    If an embedding is stored as a string, it is converted to a list of floats.
    Returns the index and the numpy array of embeddings.
    """
    embedding_list = []
    for x in df_embeddings["embedding"]:
        if isinstance(x, str):
            # Convert the string representation of a list to an actual list
            embedding = ast.literal_eval(x)
        else:
            embedding = x
        embedding_list.append(embedding)
    X = np.array(embedding_list, dtype=np.float32)
    if debug:
        print("Building FAISS index on embeddings array of shape:", X.shape)
    faiss.normalize_L2(X)
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X)
    return index, X

def search_videos(query: str, index: faiss.IndexFlatIP, df_embeddings: pd.DataFrame, videos_df: pd.DataFrame, top_k: int = 10, return_transcript: bool = False, debug: bool = False) -> pd.DataFrame:
    """
    Given a query string, embeds it and searches the topic embeddings to find the closest videos.
    Returns a DataFrame of the top videos with their video_index, title, url, similarity, topics, and topic_index.
    """
    query_emb = embed_query(query, debug=debug).reshape(1, -1)
    distances, indices = index.search(query_emb, top_k*3)
    if debug:
        print("FAISS search distances:", distances)
        print("FAISS search indices:", indices)

    video_similarities = {}
    topic_indices = {}
    for sim, idx in zip(distances[0], indices[0]):
        video_index = int(df_embeddings.iloc[idx]["video_index"])
        topic_index = int(df_embeddings.iloc[idx]["topic_number"]) - 1
        if video_index not in video_similarities or sim > video_similarities[video_index]:
            video_similarities[video_index] = sim
            topic_indices[video_index] = topic_index  # store index of the most similar topic

    sorted_videos = sorted(video_similarities.items(), key=lambda x: x[1], reverse=True)[:top_k]
    if debug:
        print("Top video indices and similarities:", sorted_videos)

    results = []
    for video_index, sim in sorted_videos:
        row = videos_df[videos_df["id"] == video_index]
        if row.empty:
            if debug:
                print(f"Video with id {video_index} not found in videos.csv.")
            continue
        title = row.iloc[0]["title"]
        url = row.iloc[0]["url"]
        topics = df_embeddings[df_embeddings["video_index"] == video_index]["topic"].tolist()
        result = {
            "video_index": video_index,
            "title": title,
            "url": url,
            "similarity": sim,
            "topics": topics,
            "topic_index": topic_indices[video_index]
        }
        if return_transcript:
            with open(os.path.abspath(row.iloc[0]["transcript_file"])) as transcript_file:
                result["transcript"] = transcript_file.read()

        results.append(result)

    return pd.DataFrame(results)

if __name__ == "__main__":
    df_embeddings = pd.read_csv("videos_topics_embedded.csv")
    videos_df = pd.read_csv("videos.csv")
    query_str = "How can meditation reduce stress?"
    index, _ = build_faiss_index(df_embeddings, debug=True)
    results_df = search_videos(query_str, index, df_embeddings, videos_df, top_k=10, return_transcript=True,  debug=True)
    print(results_df)
