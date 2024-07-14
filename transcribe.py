import os
import whisper
from datetime import datetime
from preprocess import (
    remove_timestamps,
    remove_special_characters,
    remove_duplicate_lines,
)


def transcribe_audio():
    input_folder = "normalized_audio"
    output_folder = "transcripts"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the Whisper model
    model = whisper.load_model("medium")

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}.txt"
            )

            # Transcribe the audio
            result = model.transcribe(input_path)

            # Save the transcript in the standard Whisper format
            with open(output_path, "w", encoding="utf-8") as f:
                for segment in result["segments"]:
                    start_time = datetime.utcfromtimestamp(segment["start"]).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    text = segment["text"].strip()
                    cleaned_text = remove_duplicate_lines(
                        remove_special_characters(remove_timestamps(text))
                    )
                    f.write(f"{start_time}: {cleaned_text}\n")
                    # Add an empty line for segments with no speech
                    if not cleaned_text:
                        f.write("\n")

            print(f"Transcribed: {filename}")


# Usage
transcribe_audio()
