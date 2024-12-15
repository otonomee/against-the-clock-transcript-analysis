import os
from collections import defaultdict
import pandas as pd
from pathlib import Path
import re
from typing import Dict, List, Tuple


class TranscriptAnalyzer:
    def __init__(self, transcript_dir: str):
        self.transcript_dir = transcript_dir
        self.time_buckets = defaultdict(list)
        self.producer_actions = defaultdict(list)

    def parse_timestamp_line(self, line: str) -> Tuple[int, str]:
        """Extract timestamp and text from a line"""
        parts = line.split(": ", 1)
        if len(parts) != 2:
            return None, None

        timestamp_str, text = parts
        try:
            h, m, s = map(int, timestamp_str.split(":"))
            total_seconds = h * 3600 + m * 60 + s
            return total_seconds, text.strip()
        except:
            return None, None

    def should_skip_line(self, text: str) -> bool:
        """Check if line should be skipped (filter noise)"""
        skip_phrases = {
            "let's go",
            "wow",
            "applause",
            "yeah",
            "okay",
            "cool",
            "all right",
            "mm-hmm",
            "uh-huh",
            "um",
            "uh",
        }
        return (
            text.lower() in skip_phrases
            or len(text) < 5
            or text.count(" ") < 2  # Skip very short phrases
        )

    def extract_key_actions(self, text: str) -> List[str]:
        """Extract key production-related actions from text"""
        key_terms = {
            "drum",
            "beat",
            "synth",
            "bass",
            "kick",
            "snare",
            "sample",
            "sequence",
            "automation",
            "filter",
            "effect",
            "delay",
            "reverb",
            "midi",
            "tempo",
            "pattern",
            "mix",
            "eq",
            "compress",
            "melody",
            "pad",
            "chord",
        }

        words = set(text.lower().split())
        if any(term in words for term in key_terms):
            return [text]
        return []

    def process_file(self, filepath: str) -> None:
        """Process a single transcript file"""
        producer_name = Path(filepath).stem

        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()

        for line in content.split("\n"):
            seconds, text = self.parse_timestamp_line(line)
            if seconds is None or self.should_skip_line(text):
                continue

            # Group into 30-second buckets
            bucket = (seconds // 30) * 30
            actions = self.extract_key_actions(text)

            if actions:
                self.time_buckets[bucket].extend(actions)
                self.producer_actions[producer_name].append((bucket, text))

    def process_all_files(self) -> None:
        """Process all transcript files in directory"""
        for filename in os.listdir(self.transcript_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.transcript_dir, filename)
                self.process_file(filepath)

    def get_temporal_analysis(self) -> pd.DataFrame:
        """Generate temporal analysis DataFrame"""
        rows = []
        for time, actions in sorted(self.time_buckets.items()):
            minutes = time // 60
            seconds = time % 60
            time_str = f"{minutes:02d}:{seconds:02d}"

            # Count most common actions in this time bucket
            action_summary = " | ".join(actions[:3])  # Limit to top 3 actions

            rows.append(
                {"time": time_str, "count": len(actions), "actions": action_summary}
            )

        return pd.DataFrame(rows)

    def get_producer_patterns(self) -> Dict[str, List[Tuple[int, str]]]:
        """Get patterns for each producer"""
        return dict(self.producer_actions)


def main():
    analyzer = TranscriptAnalyzer("./transcripts")
    analyzer.process_all_files()

    # Get temporal analysis
    df = analyzer.get_temporal_analysis()

    print("\nTemporal Analysis of Producer Actions:")
    print(df.to_string(index=False))

    # Print some summary statistics
    print("\nMost Active Time Periods:")
    most_active = df.nlargest(3, "count")
    print(most_active.to_string(index=False))

    # Get producer patterns
    patterns = analyzer.get_producer_patterns()
    print(f"\nAnalyzed {len(patterns)} producers")

    return df, patterns


if __name__ == "__main__":
    df, patterns = main()
