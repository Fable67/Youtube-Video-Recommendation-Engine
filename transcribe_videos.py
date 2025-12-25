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


def fetch_video_transcripts(csv_path, num_videos=None):
    """
    Fetches the transcripts of videos from youtube links.

    Parameters:
        csv_path (str): Path to the csv file with tab as separator and title and url columns.
        num_videos (int, optional): The number of videos to transcribe. If None, all videos will be transcribed.
    """
    try:
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

    print(f"Total videos to transcribe: {len(videos)}")

    # Directory to save transcripts
    output_dir = "transcripts"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    yt = YouTubeTranscriptApi()

    # Progress bar
    transcript_files = []
    for i, (video_title, video_url) in enumerate(tqdm(videos, desc="Transcribing videos", unit="video")):
        try:
            video_id = video_url.split('=')[-1]
            if not DEBUG:
                transcript = yt.fetch(video_id).to_raw_data()
            else:
                transcript = [{"text": "debug transcript"}]
            time.sleep(1)

            # video_title.replace("/", "")

            # Create a text file for each video transcript
            transcript_file = os.path.join(output_dir, f"{slugify(video_title.replace('/', ''))}.txt")
            
            with open(transcript_file, 'w', encoding='utf-8') as file:
                file.write(json.dumps(transcript))

            videos_df.loc[videos_df["title"] == video_title, "transcript_file"] = transcript_file
            
            transcript_files.append(transcript_file)

        except NoTranscriptFound:
            print(f"Warning: No transcript found for video {video_url}. Skipping.")
            videos_df.drop(i, inplace=True)
        except Exception as e:
            print(f"Error: Could not fetch transcript for video {video_url}. ({str(e)})")

    # videos_df["transcript_file"] = transcript_files
    videos_df.to_csv(f"{csv_path}_transcribed.csv", sep=",", index=False)



def main():
    """
    The main function to handle user inputs and initiate the transcription process.
    """
    print("YouTube Channel Video Transcription")
    channel_url = input("Enter the path to the csv file containing video titles and urls: ")

    # Get the number of videos to transcribe
    num_videos = input("Enter the number of videos to transcribe (leave blank for all videos): ")
    num_videos = int(num_videos) if num_videos.strip().isdigit() else None

    # Fetch the video transcripts
    fetch_video_transcripts(channel_url, num_videos)

if __name__ == "__main__":
    main()
