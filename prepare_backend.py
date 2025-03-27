import os
import ast
from tqdm import tqdm
import pandas as pd

from helpers.topic_extraction import get_topics_from_transcripts
from helpers.topics_embedding import get_topics_embeddings
from helpers.embeddings_dataframe import (
    flatten_batched_embeddings_and_topics,
    create_embeddings_dataframe,
    create_embeddings_dataframe_from_batched
)

"""
STEP 1:
    Load the csv which contains videos in each row with information:
        - video title
        - video url
        - file name of the transcipt
    Example:
        title,                           url,                         transcript_file
        Metta Has To Start With Your..., https://www.youtube.com/..., metta-has-to-start-with-your-own...
        We Make Things Difficult...,     https://www.youtube.com/..., we-make-things-difficult-dhamma-dasa...
        Investigating | Shamil #2 ...,   https://www.youtube.com/..., investigating-shamil-2-090224
        ...

    Load this into a pandas dataframe
STEP 2:
    Add an id to each video in the dataframe

STEP 3:
    Loop over the videos in batches:
    Load the transcripts

    STEP 4:
        Pass the transcripts to OpenAI API to extract the topics
        (Topics should be described in a relatively short text)
        Example response:
            - Topic 1 is metta
            - Topic 2 is anapanasati
            - Topic 3 is about mental problems
            - ...
    STEP 5:
      Split the topics for each video
      Embed the topics separately
STEP 6:
    Save into new dataframe these rows:
        - video id, topic 1 num, topic 1 embedding
        - video id, topic 2 num, topic 2 embedding
        - video id, topic 3 num, topic 3 embedding
        ...
STEP 7:
    Save the embeddings dataframe

Caching logic:
    - If FORCE_REBUILD is False and videos_topics_embedded.csv exists, load it
      and determine which video indices have already been processed.
    - Then only process videos (in batches) that have not been processed yet.
    - Finally, concatenate cached results with new rows and save the file.
"""

def load_videos_information_csv(file_path: str, debug=False) -> pd.DataFrame:
    # Load the CSV and get the file names
    if not os.path.exists(file_path):
        print("ERROR: videos.csv not found")
        exit(1)
    videos_df = pd.read_csv(file_path)
    if debug: print(f"Number of videos is {len(videos_df)}")
    # Accept either structure with or without 'id'
    expected1 = ["title", "url", "transcript_file"]
    expected2 = ["title", "url", "transcript_file", "id"]
    if videos_df.columns.tolist() not in [expected1, expected2]:
        print("ERROR: videos.csv file does not have the correct structure: title, url, transcript_file, [id]")
        exit(1)
    return videos_df


def batch_load_transcripts(files: list[str], debug=False) -> list[str]:
    batch_transcripts_ = []
    if debug: print(f"DEBUG: Number of transcript files in batch: {len(files)}")
    for file in files:
        if not os.path.exists(file):
            print(f"ERROR: File {file} does not exist in transcript folder")
            exit(1)
        with open(file, "r") as f:
            batch_transcripts_.append(f.read())
    return batch_transcripts_


if __name__ == "__main__":
    # HYPERPARAMETERS
    videos_csv_path = "videos.csv"
    EMBEDDINGS_CACHE_FILE = "videos_topics_embedded.csv"
    transcripts_folder_path = "transcripts/"
    FORCE_REBUILD = False
    DEBUG = False
    BATCHSIZE = 2
    NUM_VIDEOS = 254

    if NUM_VIDEOS % BATCHSIZE != 0:
        print(f"NUM_VIDEOS ({NUM_VIDEOS}) must be divisible by BATCHSIZE ({BATCHSIZE}).")
        exit(1)

    # STEP 1: Load videos.csv
    videos_df = load_videos_information_csv(videos_csv_path, debug=DEBUG)

    # STEP 2: Ensure there is an 'id' column
    videos_df["id"] = videos_df.index
    videos_df.to_csv("videos.csv", index=False)
    # Limit to NUM_VIDEOS
    if videos_df.shape[0] > NUM_VIDEOS:
        videos_df = videos_df.iloc[:NUM_VIDEOS]
    

    # STEP 3: Ensure all transcript files exist.
    for i in range(0, videos_df.shape[0]):
        transcript_file = videos_df.iloc[i, 2]
        if not os.path.exists(transcript_file):
            print(f"ERROR: File {transcript_file} does not exist. Stopping...")
            exit(1)

    # Load cache if available and not forcing a rebuild.
    if not FORCE_REBUILD and os.path.exists(EMBEDDINGS_CACHE_FILE):
        cached_df = pd.read_csv(EMBEDDINGS_CACHE_FILE)
        # Convert embeddings column from string to list, if necessary.
        cached_df["embedding"] = cached_df["embedding"].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
        processed_ids = set(cached_df["video_index"].unique())
        if DEBUG:
            print("Cached video indices:", processed_ids)
    else:
        cached_df = pd.DataFrame(columns=["video_index", "topic_number", "topic", "embedding"])
        processed_ids = set()

    # ------------------
    # Processing New Videos in Batches
    # ------------------
    new_batched_topics = []       # List of batched topics (structure: batch -> video -> topics list)
    new_batched_embeddings = []   # List of batched embeddings (structure: batch -> video -> topics embeddings)
    new_batches_processed = 0

    # Process videos in batches (only new ones)
    for i in tqdm(range(0, videos_df.shape[0], BATCHSIZE)):
        batch_videos = videos_df.iloc[i:i+BATCHSIZE]
        batch_video_ids = batch_videos["id"].tolist()
        # Select only videos that are not already processed.
        new_video_ids = [vid for vid in batch_video_ids if vid not in processed_ids]
        if not new_video_ids:
            if DEBUG:
                print(f"Batch starting at index {i}: all videos already processed. Skipping batch.")
            continue

        # Filter the batch for new videos.
        new_batch_videos = batch_videos[batch_videos["id"].isin(new_video_ids)]
        transcript_files = new_batch_videos["transcript_file"].tolist()
        batch_transcripts = batch_load_transcripts(transcript_files, debug=DEBUG)

        # STEP 4: Extract topics for the new batch of transcripts.
        topics = get_topics_from_transcripts(batch_transcripts, debug=DEBUG)
        new_batched_topics.append(topics)

        # STEP 5: Compute embeddings for the topics.
        embeddings = get_topics_embeddings(topics, debug=DEBUG)
        new_batched_embeddings.append(embeddings)

        new_batches_processed += 1

    if new_batches_processed > 0:
        # Create new embeddings dataframe for the new videos.
        new_embeddings_df = create_embeddings_dataframe_from_batched(new_batched_embeddings, new_batched_topics, debug=DEBUG)
        new_embeddings_df["video_index"] = new_embeddings_df["video_index"] + (NUM_VIDEOS - new_batches_processed*BATCHSIZE)
        # Combine with cached results.
        final_df = pd.concat([cached_df, new_embeddings_df], ignore_index=True)
        if DEBUG:
            print(f"Processed {new_batches_processed} new batch(es).")
    else:
        final_df = cached_df
        if DEBUG:
            print("No new videos to process.")


    # STEP 7: Save the combined embeddings dataframe.
    final_df.to_csv(EMBEDDINGS_CACHE_FILE, index=False)
    print("Embeddings file saved!")






