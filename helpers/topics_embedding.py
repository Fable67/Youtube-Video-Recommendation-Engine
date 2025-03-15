import time
import openai
from openai import OpenAI

client = OpenAI()

def call_embedding_api_with_retry(topic: str, model: str, debug: bool = False) -> list[float]:
    """
    Calls the OpenAI Embedding API for the given topic using exponential backoff for rate limits.
    Returns the embedding vector as a list of floats.
    """
    retries = 0
    max_retries = 5
    while retries < max_retries:
        try:
            response = client.embeddings.create(input=topic,
            model=model)
            embedding = response.data[0].embedding
            if debug:
                print(f"Embedding retrieved for topic: {topic[:50]}...")
            return embedding
        except openai.RateLimitError as e:
            sleep_time = 2 ** retries
            if debug:
                print(f"Embedding rate limit hit; retrying in {sleep_time} seconds...")
            time.sleep(sleep_time)
            retries += 1
    raise Exception("Embedding API call failed after retries.")

def get_topics_embeddings(topics: list[list[str]], debug: bool = False) -> list[list[list[float]]]:
    """
    For each transcript's list of topics, obtains embeddings for each topic using OpenAI's embedding API.
    
    Args:
        topics (list[list[str]]): A list where each element is a list of topic strings (per transcript).
        debug (bool): If True, prints debug information.
        
    Returns:
        list[list[list[float]]]: A list for each transcript; each transcript is represented as a list of 
                                  embedding vectors (each embedding vector is a list of floats).
                                  
    Uses the "text-embedding-3-large" model as per cost analysis and intended use.
    """
    embedding_model = "text-embedding-3-large"
    all_embeddings = []

    for transcript_idx, transcript_topics in enumerate(topics):
        if debug:
            print(f"\nProcessing embeddings for transcript {transcript_idx} with {len(transcript_topics)} topics...")
        transcript_embeddings = []
        for topic in transcript_topics:
            if debug:
                print(f"Embedding topic: {topic}")
            embedding = call_embedding_api_with_retry(topic, embedding_model, debug)
            transcript_embeddings.append(embedding)
        all_embeddings.append(transcript_embeddings)

    return all_embeddings

# Example usage:
if __name__ == "__main__":
    # Example topics: list for each transcript, each with a few topics.
    sample_topics = [
        ["Topic 1: Meditation", "Topic 2: Mindfulness", "Topic 3: Stress Reduction"],
        ["Topic 1: Metta", "Topic 2: Anapanasati", "Topic 3: Compassion"],
        ["Topic 1: Mental Health", "Topic 2: Emotional Regulation"]  # even fewer topics is fine
    ]
    embeddings = get_topics_embeddings(sample_topics, debug=True)
    for i, emb_list in enumerate(embeddings):
        print(f"\nTranscript {i} embeddings:")
        for emb in emb_list:
            print(emb[:5], "...")  # print first few numbers for brevity
