import os
import whisper
import json


def transcribe_audio():
    input_folder = "normalized_audio"
    output_folder = "transcripts"

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Load the Whisper model
    model = whisper.load_model("large")

    for filename in os.listdir(input_folder):
        if filename.endswith(".mp3"):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(
                output_folder, f"{os.path.splitext(filename)[0]}.json"
            )

            # Transcribe the audio
            result = model.transcribe(input_path)

            # Save the transcript as JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, ensure_ascii=False, indent=4)

            print(f"Transcribed: {filename}")


# Usage
transcribe_audio()
