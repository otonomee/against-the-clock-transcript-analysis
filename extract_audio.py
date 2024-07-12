import csv
import os
import subprocess


# Function to download audio from YouTube video
def download_audio(video_url, video_title):
    # Ensure the directory path is included in the output template
    output_template = os.path.join("extracted_audio", "%(title)s.%(ext)s")

    # Corrected command to download audio using yt-dlp with the specified output template
    command = (
        f'yt-dlp --no-part -x --audio-format mp3 -o "{output_template}" {video_url}'
    )
    subprocess.run(command, shell=True)


# Read the CSV and download audio for each video
with open("videos.csv", mode="r") as file:
    csv_reader = csv.DictReader(file)
    for row in csv_reader:
        video_url = row["video_url"]
        video_title = row["title"]
        download_audio(video_url, video_title)

print("Download completed.")
