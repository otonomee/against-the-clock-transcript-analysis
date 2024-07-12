import csv
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from urllib.parse import unquote


def sanitize_title(title):
    # Remove characters not allowed in filenames and trim to a reasonable length
    return "".join(x for x in title if x.isalnum() or x in " _-").rstrip()[:50]


def download_audio(video_url, video_title):
    # Sanitize the video title to create a safe filename
    sanitized_title = sanitize_title(video_title)
    output_dir = "extracted_audio"
    os.makedirs(output_dir, exist_ok=True)

    # Check if a file starting with the sanitized title already exists
    if any(f.startswith(sanitized_title) for f in os.listdir(output_dir)):
        print(f"Skipping download, audio for '{video_title}' already exists.")
        return

    # Corrected command to download audio using yt-dlp with the specified output template
    output_template = os.path.join(output_dir, "%(title)s.%(ext)s")
    command = (
        f'yt-dlp --no-part -x --audio-format mp3 -o "{output_template}" {video_url}'
    )
    subprocess.run(command, shell=True)


def download_videos_concurrently(video_details):
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(lambda details: download_audio(*details), video_details)


if __name__ == "__main__":
    video_details = []
    with open("videos.csv", mode="r") as file:
        csv_reader = csv.DictReader(file)
        for row in csv_reader:
            video_details.append((row["video_url"], row["title"]))

    download_videos_concurrently(video_details)
    print("Download completed.")
