import os

# Assuming preprocess.py is in the same directory as this script
from preprocess import preprocess_text

def process_transcripts(input_dir="transcripts", output_dir="preprocessed_transcripts"):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all .txt files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(".txt"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            # Read the content of the input file
            with open(input_path, "r", encoding="utf-8") as file:
                content = file.read()

            # Process the content
            processed_content = preprocess_text(content)

            # Write the processed content to the output file
            with open(output_path, "w", encoding="utf-8") as file:
                file.write(processed_content)

            print(f"Processed {filename}")

if __name__ == "__main__":
    process_transcripts()
