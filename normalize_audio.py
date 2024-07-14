import os
from pydub import AudioSegment
import numpy as np

def normalize_audio(target_dBFS=-20):
    input_folder = 'extracted_audio'
    output_folder = 'normalized_audio'

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for filename in os.listdir(input_folder):
        if filename.endswith('.mp3'):
            input_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, filename)
            
            audio = AudioSegment.from_mp3(input_path)
            change_in_dBFS = target_dBFS - audio.dBFS
            normalized_audio = audio.apply_gain(change_in_dBFS)
            
            normalized_audio.export(output_path, format='mp3')
            print(f"Normalized: {filename}")

# Usage
normalize_audio()
