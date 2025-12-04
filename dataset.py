"""Dataset handler for NEJM Image Challenge data."""

import base64
import hashlib
import json
import random
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

from config import CACHE_DIR, DATASET_URL, RANDOM_SEED, SAMPLE_SIZE, DIFFICULTY_MODE, BRIER_THRESHOLDS

# NEJM image server base URL
NEJM_IMAGE_BASE = "https://csvc.nejm.org/ContentServer/images?id=IC"


def parse_nejm_date(date_str: str) -> str:
    """Convert date like 'apr-01-2010' to 'YYYYMMDD' format."""
    # Handle formats like "apr-01-2010" or "dec-22-2022"
    try:
        dt = datetime.strptime(date_str, "%b-%d-%Y")
        return dt.strftime("%Y%m%d")
    except ValueError:
        # Try alternative format if needed
        try:
            dt = datetime.strptime(date_str, "%B-%d-%Y")
            return dt.strftime("%Y%m%d")
        except ValueError:
            return ""


def construct_image_url(date_str: str) -> str:
    """Construct NEJM image URL from date string."""
    date_code = parse_nejm_date(date_str)
    if date_code:
        return f"{NEJM_IMAGE_BASE}{date_code}"
    return ""


class Challenge:
    """Represents a single NEJM Image Challenge."""

    def __init__(self, data: dict[str, Any]):
        # Parse the actual dataset structure
        self.image_id = data.get("image_id", "")
        self.id = str(self.image_id)
        self.date = data.get("date", "")

        # Construct image URL from date
        self.image_url = construct_image_url(self.date)

        # Clinical description is in "question" field
        self.clinical_description = data.get("question", "")

        # Parse options (option_A, option_B, etc.)
        self.options = {}
        for letter in ["A", "B", "C", "D", "E"]:
            key = f"option_{letter}"
            if key in data:
                self.options[letter] = data[key]

        # Correct answer is a letter (A, B, C, D, or E)
        self.correct_answer_letter = data.get("correct_answer", "")

        # Get the actual text of the correct answer
        self.correct_answer = self.options.get(self.correct_answer_letter, "")

        # Vote distribution - parse from "option_X_votes" fields (e.g., "12%")
        self.vote_distribution = {}
        for letter in ["A", "B", "C", "D", "E"]:
            vote_key = f"option_{letter}_votes"
            if vote_key in data:
                vote_str = data[vote_key]
                # Parse "12%" to 12.0
                if isinstance(vote_str, str) and vote_str.endswith("%"):
                    self.vote_distribution[letter] = float(vote_str.rstrip("%"))
                elif isinstance(vote_str, (int, float)):
                    self.vote_distribution[letter] = float(vote_str)

        # Use pre-calculated Brier score if available in dataset
        self._brier_score_cached = data.get("brier_score")

        self._image_base64: str | None = None

    @property
    def brier_score(self) -> float:
        """Get Brier score (lower = easier case).

        Uses pre-calculated value from dataset if available, otherwise
        calculates from vote distribution.

        Brier score measures the accuracy of probabilistic predictions.
        For each option, we calculate (predicted_probability - actual_outcome)^2
        where actual_outcome is 1 for the correct answer and 0 for others.
        """
        # Use pre-calculated value if available
        if self._brier_score_cached is not None:
            return float(self._brier_score_cached)

        # Calculate from vote distribution
        if not self.vote_distribution or not self.correct_answer_letter:
            return 0.5  # Default to medium difficulty if data missing

        brier = 0.0
        for letter in ["A", "B", "C", "D", "E"]:
            vote_pct = self.vote_distribution.get(letter, 0)
            actual = 1.0 if letter == self.correct_answer_letter else 0.0
            brier += (vote_pct / 100.0 - actual) ** 2
        return brier

    @property
    def physician_accuracy(self) -> float:
        """Return the percentage of physicians who selected the correct answer."""
        if not self.vote_distribution or not self.correct_answer_letter:
            return 0.0
        return self.vote_distribution.get(self.correct_answer_letter, 0)

    def __repr__(self) -> str:
        answer_preview = self.correct_answer[:30] if self.correct_answer else "N/A"
        return f"Challenge(id={self.id}, date={self.date}, answer={answer_preview}...)"


class NEJMDataset:
    """Handles loading and sampling NEJM Image Challenge dataset."""

    def __init__(self, cache_dir: str = CACHE_DIR):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.challenges: list[Challenge] = []

    async def load(self) -> None:
        """Load the dataset from GitHub or local cache."""
        dataset_cache = self.cache_dir / "dataset.json"

        if dataset_cache.exists():
            print("Loading dataset from cache...")
            with open(dataset_cache, "r", encoding="utf-8") as f:
                data = json.load(f)
        else:
            print(f"Downloading dataset from {DATASET_URL}...")
            async with httpx.AsyncClient() as client:
                response = await client.get(DATASET_URL, timeout=60.0)
                response.raise_for_status()
                data = response.json()

            # Cache the dataset
            with open(dataset_cache, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"Dataset cached to {dataset_cache}")

        # Parse challenges - filter out any with missing image URLs
        all_challenges = [Challenge(item) for item in data]
        self.challenges = [c for c in all_challenges if c.image_url and c.correct_answer]

        print(f"Loaded {len(self.challenges)} valid challenges (out of {len(all_challenges)} total)")

    def sample(
        self, n: int = SAMPLE_SIZE, seed: int = RANDOM_SEED, difficulty: str = DIFFICULTY_MODE
    ) -> list[Challenge]:
        """Randomly sample n challenges from the dataset, filtered by difficulty.

        Args:
            n: Number of challenges to sample
            seed: Random seed for reproducibility
            difficulty: Difficulty mode - "all", "hard", or "expert"
                - "all": No filtering, use all cases
                - "hard": Brier >= 0.087 (physician accuracy ~30%)
                - "expert": Brier >= 0.123 (physician accuracy ~27%)
        """
        if not self.challenges:
            raise ValueError("Dataset not loaded. Call load() first.")

        # Filter by difficulty
        candidates = self.challenges
        if difficulty == "hard":
            threshold = BRIER_THRESHOLDS.get("hard", 0.087)
            candidates = [c for c in self.challenges if c.brier_score >= threshold]
            print(f"Filtered to {len(candidates)} hard cases (Brier >= {threshold})")
        elif difficulty == "expert":
            threshold = BRIER_THRESHOLDS.get("expert", 0.123)
            candidates = [c for c in self.challenges if c.brier_score >= threshold]
            print(f"Filtered to {len(candidates)} expert cases (Brier >= {threshold})")
        else:
            print(f"Using all {len(candidates)} cases (no difficulty filter)")

        if len(candidates) < n:
            print(f"Warning: Only {len(candidates)} cases available, requested {n}")
            n = len(candidates)

        random.seed(seed)
        sampled = random.sample(candidates, min(n, len(candidates)))

        # Print difficulty stats for sampled cases
        avg_brier = sum(c.brier_score for c in sampled) / len(sampled) if sampled else 0
        avg_physician_acc = sum(c.physician_accuracy for c in sampled) / len(sampled) if sampled else 0
        print(f"Sampled {len(sampled)} challenges (seed={seed})")
        print(f"  Avg Brier score: {avg_brier:.3f}")
        print(f"  Avg physician accuracy: {avg_physician_acc:.1f}%")

        return sampled

    async def download_image(self, challenge: Challenge) -> str:
        """Download and cache an image, return base64 encoded string."""
        if challenge._image_base64:
            return challenge._image_base64

        if not challenge.image_url:
            raise ValueError(f"No image URL for challenge {challenge.id}")

        # Create a filename from URL hash
        url_hash = hashlib.md5(challenge.image_url.encode()).hexdigest()
        image_cache = self.cache_dir / f"{url_hash}.jpg"

        if image_cache.exists():
            with open(image_cache, "rb") as f:
                image_bytes = f.read()
        else:
            print(f"Downloading image for challenge {challenge.id} ({challenge.date})...")
            async with httpx.AsyncClient() as client:
                # NEJM images may need headers to download
                headers = {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Referer": "https://www.nejm.org/",
                }
                response = await client.get(
                    challenge.image_url, headers=headers, timeout=30.0, follow_redirects=True
                )
                response.raise_for_status()
                image_bytes = response.content

            # Cache the image
            with open(image_cache, "wb") as f:
                f.write(image_bytes)

        # Encode to base64
        challenge._image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        return challenge._image_base64

    async def prepare_challenges(self, challenges: list[Challenge]) -> list[Challenge]:
        """Download all images for a list of challenges."""
        print(f"Preparing {len(challenges)} challenges (downloading images)...")
        successful = 0
        for i, challenge in enumerate(challenges):
            try:
                await self.download_image(challenge)
                successful += 1
                print(f"  [{i+1}/{len(challenges)}] OK - {challenge.id} ({challenge.date})")
            except Exception as e:
                print(f"  [{i+1}/{len(challenges)}] FAILED - {challenge.id}: {e}")

        print(f"Successfully downloaded {successful}/{len(challenges)} images")
        return challenges


async def load_and_sample_dataset(
    n: int = SAMPLE_SIZE, seed: int = RANDOM_SEED, difficulty: str = DIFFICULTY_MODE
) -> list[Challenge]:
    """Convenience function to load dataset and return sampled challenges with images.

    Args:
        n: Number of challenges to sample
        seed: Random seed for reproducibility
        difficulty: Difficulty mode - "all", "hard", or "expert"
    """
    dataset = NEJMDataset()
    await dataset.load()
    sampled = dataset.sample(n, seed, difficulty)
    await dataset.prepare_challenges(sampled)
    return sampled
