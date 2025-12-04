"""Binary scoring for multiple choice responses."""

import re
from dataclasses import dataclass, field

from dataset import Challenge
from models import ModelResponse


def extract_letter(response_text: str) -> str | None:
    """Extract the selected letter (A-E) from a model response.

    Handles various response formats:
    - Just the letter: "B"
    - Letter with period: "B."
    - Letter at start: "B) Amelanotic melanoma"
    - "The answer is B"
    - "I select option B"
    """
    if not response_text:
        return None

    text = response_text.strip().upper()

    # Check for just a single letter
    if len(text) == 1 and text in "ABCDE":
        return text

    # Check for letter at the very start (possibly followed by punctuation)
    if text and text[0] in "ABCDE":
        # Make sure it's not part of a word like "AND"
        if len(text) == 1 or text[1] in " .):\n\t":
            return text[0]

    # Look for common patterns
    patterns = [
        r"(?:the\s+)?answer\s+is\s+([A-E])",
        r"(?:i\s+)?(?:select|choose|pick)\s+(?:option\s+)?([A-E])",
        r"option\s+([A-E])",
        r"^([A-E])\s*[\)\.:\-]",
        r"\b([A-E])\b(?:\s*[\)\.:]|\s+is\s+(?:the\s+)?(?:correct|best|most\s+likely))",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Last resort: find any standalone letter A-E
    match = re.search(r"\b([A-E])\b", text)
    if match:
        return match.group(1).upper()

    return None


def score_response(response_text: str, correct_letter: str) -> tuple[int, str, str]:
    """Score a model response using binary scoring.

    Args:
        response_text: The model's response
        correct_letter: The correct answer letter (A-E)

    Returns:
        Tuple of (score, category, selected_letter)
        - score: 1 if correct, 0 if incorrect
        - category: "Correct" or "Incorrect" or "Invalid"
        - selected_letter: The letter extracted from the response
    """
    selected = extract_letter(response_text)

    if selected is None:
        return 0, "Invalid", ""

    if selected == correct_letter.upper():
        return 1, "Correct", selected

    return 0, "Incorrect", selected


@dataclass
class JudgeScore:
    """Score for a single challenge response."""

    challenge_id: str
    model_id: str
    model_response: str
    correct_answer_letter: str
    correct_answer_text: str
    selected_letter: str
    score: int  # 0 or 1
    category: str  # "Correct", "Incorrect", or "Invalid"
    success: bool
    error: str | None = None


@dataclass
class ModelScores:
    """Aggregated scores for a single model."""

    model_name: str
    scores: list[JudgeScore] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        """Percentage of correct answers (0-100)."""
        valid_scores = [s for s in self.scores if s.success]
        if not valid_scores:
            return 0.0
        return sum(s.score for s in valid_scores) / len(valid_scores) * 100

    @property
    def correct_count(self) -> int:
        """Number of correct answers."""
        return sum(s.score for s in self.scores if s.success)

    @property
    def total_count(self) -> int:
        """Total number of evaluated responses."""
        return sum(1 for s in self.scores if s.success)

    @property
    def category_breakdown(self) -> dict[str, int]:
        """Count of scores by category."""
        breakdown: dict[str, int] = {"Correct": 0, "Incorrect": 0, "Invalid": 0}
        for s in self.scores:
            if s.success and s.category in breakdown:
                breakdown[s.category] += 1
        return breakdown

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.scores if s.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for s in self.scores if not s.success)

    # Legacy property for backwards compatibility
    @property
    def average_score(self) -> float:
        """Legacy: Returns accuracy as a 0-10 scale for backwards compatibility."""
        return self.accuracy / 10

    @property
    def score_distribution(self) -> dict[int, int]:
        """Binary distribution: count of 0s and 1s."""
        dist = {0: 0, 1: 0}
        for s in self.scores:
            if s.success:
                dist[s.score] = dist.get(s.score, 0) + 1
        return dist


class JudgeClient:
    """Client for scoring multiple choice responses using binary scoring.

    Note: This no longer uses an LLM judge. Scoring is done by simple
    letter matching for the multiple choice format.
    """

    def __init__(self, **kwargs):
        """Initialize the judge client.

        Args are accepted for backwards compatibility but ignored.
        """
        pass

    async def judge_response(
        self, challenge: Challenge, model_response: ModelResponse
    ) -> JudgeScore:
        """Score a single model response."""
        if not model_response.success:
            return JudgeScore(
                challenge_id=challenge.id,
                model_id=model_response.model_id,
                model_response=model_response.response_text,
                correct_answer_letter=challenge.correct_answer_letter,
                correct_answer_text=challenge.correct_answer,
                selected_letter="",
                score=0,
                category="Failed",
                success=False,
                error=model_response.error,
            )

        score, category, selected = score_response(
            model_response.response_text, challenge.correct_answer_letter
        )

        return JudgeScore(
            challenge_id=challenge.id,
            model_id=model_response.model_id,
            model_response=model_response.response_text,
            correct_answer_letter=challenge.correct_answer_letter,
            correct_answer_text=challenge.correct_answer,
            selected_letter=selected,
            score=score,
            category=category,
            success=True,
        )

    async def judge_all_responses(
        self,
        challenges: list[Challenge],
        model_responses: dict[str, list[ModelResponse]],
    ) -> dict[str, ModelScores]:
        """Score all responses for all models."""
        results: dict[str, ModelScores] = {}

        # Create a lookup for challenges by ID
        challenge_lookup = {c.id: c for c in challenges}

        total_judgments = sum(len(responses) for responses in model_responses.values())
        completed = 0

        for model_name, responses in model_responses.items():
            print(f"\nScoring {model_name} responses...")
            model_scores = ModelScores(model_name=model_name)

            for response in responses:
                challenge = challenge_lookup.get(response.challenge_id)
                if not challenge:
                    continue

                score = await self.judge_response(challenge, response)
                model_scores.scores.append(score)
                completed += 1

                if score.success:
                    status_icon = "+" if score.score == 1 else "x"
                    selected_info = f"Selected {score.selected_letter}" if score.selected_letter else "No letter"
                    correct_info = f"Correct: {score.correct_answer_letter}"
                    status = f"[{status_icon}] {selected_info}, {correct_info}"
                else:
                    status = f"FAILED: {score.error}"

                print(f"  [{completed}/{total_judgments}] {response.challenge_id}: {status}")

            results[model_name] = model_scores

        return results
