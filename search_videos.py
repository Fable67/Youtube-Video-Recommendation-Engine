import numpy as np
import faiss
import pandas as pd
import ast
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

def search_videos(query: str, index: faiss.IndexFlatIP, df_embeddings: pd.DataFrame, videos_df: pd.DataFrame, top_k: int = 10, debug: bool = False) -> pd.DataFrame:
    """
    Given a query string, embeds it and searches the topic embeddings to find the closest videos.
    
    It does the following:
      1. Embeds the query and normalizes it.
      2. Builds a FAISS index from the provided df_embeddings (which must have columns:
         'video_index' and 'embedding').
      3. Performs a search to retrieve the top 20 topic matches.
      4. Aggregates the matches by video (taking the maximum cosine similarity per video).
      5. Sorts the videos by similarity and selects the top_k videos.
      6. Looks up video details (title and url) from videos.csv, which is assumed to have
         columns "id", "title", and "url".
    
    Returns a DataFrame of the top videos with their video_index, title, url, and similarity score.
    """
    # Step 1: Embed the query.
    query_emb = embed_query(query, debug=debug)
    query_emb = query_emb.reshape(1, -1)

    # Step 2: Search for the top 20 topic matches.
    distances, indices = index.search(query_emb, top_k*3)  # distances: cosine similarities
    if debug:
        print("FAISS search distances:", distances)
        print("FAISS search indices:", indices)

    # Step 3: Map each retrieved topic to its corresponding video.
    video_similarities = {}
    for sim, idx in zip(distances[0], indices[0]):
        # Get the video_index for the matched topic.
        video_index = int(df_embeddings.iloc[idx]["video_index"])
        # Aggregate by taking the maximum similarity per video.
        video_similarities[video_index] = max(video_similarities.get(video_index, 0), sim)

    # Step 4: Sort videos by similarity and select the top_k.
    sorted_videos = sorted(video_similarities.items(), key=lambda x: x[1], reverse=True)
    top_videos = sorted_videos[:top_k]
    if debug:
        print("Top video indices and similarities:", top_videos)

    # Step 5: Match video indices to video details.
    results = []
    for video_index, sim in top_videos:
        # Assume the videos.csv "id" column corresponds to video_index.
        row = videos_df[videos_df["id"] == video_index]
        if row.empty:
            if debug:
                print(f"Video with id {video_index} not found in videos.csv.")
            continue
        title = row.iloc[0]["title"]
        url = row.iloc[0]["url"]
        topics = df_embeddings[df_embeddings["video_index"] == video_index]["topic"]
        results.append({
            "video_index": video_index,
            "title": title,
            "url": url,
            "similarity": sim,
            "topics": topics
        })

    return pd.DataFrame(results)

# Example usage:
if __name__ == "__main__":
    # Assume df_embeddings is created from previous steps and has at least these columns:
    # 'video_index' and 'embedding'. For demonstration, we simulate a small DataFrame.
    df_embeddings = pd.read_csv("videos_topics_embedded.csv")

    # For testing, create a sample videos.csv.
    videos_df = pd.read_csv("videos.csv")

    # Query string.
    query_str = "How can meditation reduce stress?"

    # Build the FAISS index.
    index, _ = build_faiss_index(df_embeddings, debug=True)

    # Search for top videos.
    results_df = search_videos(query_str, index, df_embeddings, videos_df, top_k=10, debug=True)
    print(results_df)
