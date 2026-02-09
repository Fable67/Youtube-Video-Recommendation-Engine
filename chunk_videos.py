from helpers.chunking import chunk_transcript_semantic

import os
import time

import pandas as pd
from tqdm import tqdm
import unicodedata
import json
import requests
import numpy as np 

DEBUG = False

def batch_embed_fn(texts):
    response = requests.post("https://ollama.sailehd.systems/api/embed", json={"model": "qwen3-embedding:4b", "input": texts, "dimensions": 512})

    embeddings = response.json()["embeddings"]

    return [np.array(embeddings[i]) for i in range(len(embeddings))]

def chunk_transcripts(csv_path, num_videos=None, on_pick_up=False):
    """
    Chunks the transcripts.

    Parameters:
        csv_path (str): Path to the csv file with tab as separator and title and url columns.
        num_videos (int, optional): The number of videos to transcribe. If None, all videos will be transcribed.
        on_pick_up(bool, optional): Skip videos already transcribed and continue with next ones.
    """

    new_csv_path = f"{csv_path.split('.')[0]}_chunked.csv"

    try:
        # Load the csv file into memory
        # videos_df = pd.read_csv(csv_path, sep=";;;")
        # videos = videos_df.to_numpy()
        # videos_df["transcript_file"] = ""

        if on_pick_up and os.path.exists(new_csv_path):
            videos_df = pd.read_csv(new_csv_path, sep="@")
            videos = videos_df.to_numpy()
        else:
            # Load the csv file into memory
            videos_df = pd.read_csv(csv_path, sep="@")
            videos = videos_df.to_numpy()
            videos_df["chunk_file"] = ""

    except Exception as e:
        print(f"""Error: Could not load the csv file. 
        This may be due to one or more of the following reasons:
        - The path {os.path.abspath(csv_path)} is wrong.
        - The file is not a csv file.
        - The csv file used the wrong separator. Use '@' as separator!
        
        The full error message is:
        {str(e)}""")
        return

    print(videos)

    # Limit to the specified number of videos if provided
    if num_videos:
        videos = videos[:num_videos]

    # Skip already transcribed 
    if on_pick_up:
        last_idx = videos_df.loc[videos_df['chunk_file'].notna() & (videos_df['chunk_file'] != '')].index.max()
        if DEBUG:
            print("Last index with chunks: ", last_idx)

    print(f"Total videos to chunk: {len(videos)}")

    # Directory to save transcripts
    output_dir = "chunks"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    try:
        # Progress bar
        for i, columns in enumerate(tqdm(videos, desc="Chunking videos", unit="video")):
            if (on_pick_up and i < last_idx) or columns[2] is np.nan:
                continue
            video_title = columns[0]
            transcript_file = columns[2]
            SUCCESS = False
            num_tries = 0
            while not SUCCESS and num_tries <= 6:
                for _ in tqdm(range(2**num_tries), desc=f"Cooldown - Try {num_tries+1}", leave=False):
                    time.sleep(1)

                with open(transcript_file, "r") as f:
                    try:
                        chunks = chunk_transcript_semantic(json.loads(f.read()), batch_embed_fn=batch_embed_fn, debug=DEBUG)
                    except Exception as e:
                        error = str(e)
                        print(error)
                        SUCCESS = False
                        num_tries += 1

                # Create a json file transcript chunks
                chunk_file = os.path.join(output_dir, f"{transcript_file.split('/')[-1][:-4]}.json")
                
                with open(chunk_file, 'w', encoding='utf-8') as file:
                    file.write(json.dumps(chunks))

                videos_df.loc[videos_df["title"] == video_title, "chunk_file"] = chunk_file
                
                SUCCESS = True

                

            if not SUCCESS:
                print("Can't chunk more transcripts. Quitting...")
                break

    finally:
        print("Saving...")
        videos_df.to_csv(new_csv_path, sep="@", index=False)



def main():
    """
    The main function to handle user inputs and initiate the transcription process.
    """
    print("Transcript Chunker")
    csv_path = input("Enter the path to the csv file containing video titles, urls, transcript_file_path (videos_transcribed.csv): ")
    if not csv_path:
        csv_path = "videos_transcribed.csv"

    # Get the number of videos to transcribe
    num_videos = input("Enter the number of transcript to chunk (leave blank for all videos): ")
    num_videos = int(num_videos) if num_videos.strip().isdigit() else None

    # Pick up where you left off
    on_pick_up = input("Pick up where you left of (Y/N)? ")
    on_pick_up = on_pick_up == "Y"

    # Fetch the video transcripts
    chunk_transcripts(csv_path, num_videos, on_pick_up)

if __name__ == "__main__":
    main()
