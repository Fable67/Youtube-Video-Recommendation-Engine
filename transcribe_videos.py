import os
import time

import pandas as pd
from youtube_transcript_api import YouTubeTranscriptApi, NoTranscriptFound
from tqdm import tqdm
import unicodedata
import re
import json

DEBUG = True

def slugify(value, allow_unicode=False):
    """
    Taken from https://github.com/django/django/blob/master/django/utils/text.py
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize('NFKC', value)
    else:
        value = unicodedata.normalize('NFKD', value).encode('ascii', 'ignore').decode('ascii')
    value = re.sub(r'[^\w\s-]', '', value.lower())
    return re.sub(r'[-\s]+', '-', value).strip('-_')


def fetch_video_transcripts(csv_path, num_videos=None, on_pick_up=False):
    """
    Fetches the transcripts of videos from youtube links.

    Parameters:
        csv_path (str): Path to the csv file with tab as separator and title and url columns.
        num_videos (int, optional): The number of videos to transcribe. If None, all videos will be transcribed.
        on_pick_up(bool, optional): Skip videos already transcribed and continue with next ones.
    """

    new_csv_path = f"{csv_path.split('.')[0]}_transcribed.csv"

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
            videos_df = pd.read_csv(csv_path, sep=";;;")
            videos = videos_df.to_numpy()
            videos_df["transcript_file"] = ""

    except Exception as e:
        print(f"""Error: Could not load the csv file. 
        This may be due to one or more of the following reasons:
        - The path {os.path.abspath(csv_path)} is wrong.
        - The file is not a csv file.
        - The csv file used the wrong separator. Use ';;;' as separator!
        
        The full error message is:
        {str(e)}""")
        return

    print(videos)

    # Limit to the specified number of videos if provided
    if num_videos:
        videos = videos[:num_videos]

    # Skip already transcribed 
    if on_pick_up:
        last_idx = videos_df.loc[videos_df['transcript_file'].notna() & (videos_df['transcript_file'] != '')].index.max()
        if DEBUG:
            print("Last index with transcript: ", last_idx)

    print(f"Total videos to transcribe: {len(videos)}")

    # Directory to save transcripts
    output_dir = "transcripts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    yt = YouTubeTranscriptApi()

    try:
        # Progress bar
        transcript_files = []
        for i, columns in enumerate(tqdm(videos, desc="Transcribing videos", unit="video")):
            if on_pick_up and i <= last_idx:
                continue
            video_title = columns[0]
            video_url = columns[1]
            SUCCESS = False
            num_tries = 0
            while not SUCCESS and num_tries <= 6:
                try:
                    for _ in tqdm(range(2**num_tries), desc=f"Cooldown - Try {num_tries+1}", leave=False):
                        time.sleep(1)
                    video_id = video_url.split('=')[-1]
                    if not DEBUG:
                        transcript = yt.fetch(video_id).to_raw_data()
                    else:
                        transcript = [{"text": "debug transcript"}]

                    # Create a text file for each video transcript
                    transcript_file = os.path.join(output_dir, f"{slugify(video_title.replace('/', ''))}.txt")
                    
                    with open(transcript_file, 'w', encoding='utf-8') as file:
                        file.write(json.dumps(transcript))

                    videos_df.loc[videos_df["title"] == video_title, "transcript_file"] = transcript_file
                    
                    transcript_files.append(transcript_file)

                    SUCCESS = True

                except NoTranscriptFound:
                    print(f"Warning: No transcript found for video {video_url}. Skipping.")
                    videos_df.drop(i, inplace=True)
                except Exception as e:
                    error = str(e)
                    if "The video is no longer available" in error:
                        print(f"Error: Could not fetch transcript for video {video_url}. The video is no longer available.")
                        SUCCESS = True
                    else:
                        print(f"Error: Could not fetch transcript for video {video_url}. ({error})")
                        SUCCESS = False
                        num_tries += 1

            if not SUCCESS:
                print("Can't fetch more transcripts. Quitting...")
                break

    finally:
        print("Saving...")
        # videos_df["transcript_file"] = transcript_files
        videos_df.to_csv(new_csv_path, sep="@", index=False)



def main():
    """
    The main function to handle user inputs and initiate the transcription process.
    """
    print("YouTube Channel Video Transcription")
    channel_url = input("Enter the path to the csv file containing video titles and urls: ")

    # Get the number of videos to transcribe
    num_videos = input("Enter the number of videos to transcribe (leave blank for all videos): ")
    num_videos = int(num_videos) if num_videos.strip().isdigit() else None

    # Pick up where you left off
    on_pick_up = input("Pick up where you left of (Y/N)? ")
    on_pick_up = on_pick_up == "Y"

    # Fetch the video transcripts
    fetch_video_transcripts(channel_url, num_videos, on_pick_up)

if __name__ == "__main__":
    main()
