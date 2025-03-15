import pandas as pd

def flatten_batched_embeddings_and_topics(batched_embeddings: list[list[list[list[float]]]], batched_topics: list[list[list[str]]], debug: bool = False) -> list[list[list[float]]]:
    flattened_videos, flattened_topics = [], []
    for batch_index, batch in enumerate(batched_embeddings):
        flattened_videos.extend(batch)
    for batch_index, batch in enumerate(batched_topics):
        flattened_topics.extend(batch)
    assert len(flattened_videos) == len(flattened_topics)
    if debug:
        print(f"Total videos after flattening: {len(flattened_videos)}")
    return flattened_videos, flattened_topics

def create_embeddings_dataframe(flattened_videos: list[list[list[float]]], flattened_topics: list[list[str]], debug: bool = False) -> pd.DataFrame:
    rows = []
    for video_index, (topics_embeddings, topics) in enumerate(zip(flattened_videos, flattened_topics)):
        if debug:
            print(f"Processing video {video_index} with {len(topics)} topics.")
        for topic_index, (embedding, topic) in enumerate(zip(topics_embeddings, topics)):
            rows.append({
                "video_index": video_index,
                "topic_number": topic_index + 1,  # start numbering from 1
                "topic": topic,
                "embedding": embedding
            })
    df = pd.DataFrame(rows)
    return df

def create_embeddings_dataframe_from_batched(batched_embeddings: list[list[list[list[float]]]], batched_topics: list[list[list[str]]], debug: bool = False) -> pd.DataFrame:
    """
    Flattens the batched embeddings and then creates a DataFrame.
    
    Args:
        batched_embeddings (list[list[list[list[float]]]]): Nested list of embeddings with structure:
            batches -> videos -> topics -> embedding.
        debug (bool): If True, prints debug information.
        
    Returns:
        pd.DataFrame: A DataFrame with columns 'video_index', 'topic_number', 'topic', and 'embedding'.
    """
    flattened_videos, flattened_topics = flatten_batched_embeddings_and_topics(batched_embeddings, batched_topics, debug)
    return create_embeddings_dataframe(flattened_videos, flattened_topics, debug)