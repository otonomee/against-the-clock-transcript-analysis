import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

# Download necessary NLTK data
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)


def remove_timestamps(text):
    return re.sub(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}: ", "", text)


def remove_special_characters(text):
    return re.sub(r"[^a-zA-Z0-9\s]", "", text)


def remove_stopwords(text):
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    return " ".join([word for word in word_tokens if word.lower() not in stop_words])


def remove_duplicate_lines(text):
    lines = text.split("\n")
    unique_lines = []
    seen = set()
    for line in lines:
        cleaned_line = remove_timestamps(line.strip())
        if cleaned_line not in seen:
            seen.add(cleaned_line)
            unique_lines.append(line)
    return "\n".join(unique_lines)


def preprocess_text(text):
    text = remove_duplicate_lines(text)
    text = text.lower()
    text = remove_timestamps(text)
    text = remove_special_characters(text)
    text = remove_stopwords(text)
    return text


def chunk_text(text, chunk_size=1000, overlap=200):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks
