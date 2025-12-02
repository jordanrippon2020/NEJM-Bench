"""Judge model for evaluating diagnostic responses."""

import asyncio
import json
import re
from dataclasses import dataclass, field
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    JUDGE_MODEL,
    JUDGE_PROMPT_TEMPLATE,
    MAX_RETRIES,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
)
from dataset import Challenge
from models import ModelResponse


@dataclass
class JudgeScore:
    """Score from the judge model."""

    challenge_id: str
    model_id: str
    model_response: str
    correct_answer: str
    score: float
    category: str
    reasoning: str
    success: bool
    error: str | None = None


@dataclass
class ModelScores:
    """Aggregated scores for a single model."""

    model_name: str
    scores: list[JudgeScore] = field(default_factory=list)

    @property
    def average_score(self) -> float:
        valid_scores = [s.score for s in self.scores if s.success]
        return sum(valid_scores) / len(valid_scores) if valid_scores else 0.0

    @property
    def score_distribution(self) -> dict[int, int]:
        """Count of scores by integer value (0-10)."""
        dist = {i: 0 for i in range(11)}
        for s in self.scores:
            if s.success:
                dist[int(s.score)] += 1
        return dist

    @property
    def category_breakdown(self) -> dict[str, int]:
        """Count of scores by category."""
        breakdown: dict[str, int] = {}
        for s in self.scores:
            if s.success:
                cat = s.category
                breakdown[cat] = breakdown.get(cat, 0) + 1
        return breakdown

    @property
    def success_count(self) -> int:
        return sum(1 for s in self.scores if s.success)

    @property
    def failure_count(self) -> int:
        return sum(1 for s in self.scores if not s.success)


class JudgeClient:
    """Client for the judge model to evaluate responses."""

    def __init__(self, api_key: str = OPENROUTER_API_KEY, judge_model: str = JUDGE_MODEL):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")
        self.api_key = api_key
        self.judge_model = judge_model
        self.base_url = OPENROUTER_BASE_URL

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nejm-benchmark",
            "X-Title": "NEJM Image Challenge Benchmark - Judge",
        }

    def _build_judge_prompt(self, correct_answer: str, model_response: str) -> str:
        return JUDGE_PROMPT_TEMPLATE.format(
            correct_answer=correct_answer, model_response=model_response
        )

    def _parse_judge_response(self, response_text: str) -> dict[str, Any]:
        """Parse JSON from judge response, handling various formats."""
        # Try to extract JSON from the response
        # Sometimes models wrap JSON in markdown code blocks
        json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group())
            except json.JSONDecodeError:
                pass

        # Try parsing the whole response as JSON
        try:
            return json.loads(response_text)
        except json.JSONDecodeError:
            pass

        # Fallback: try to extract score from text
        score_match = re.search(r"(?:score|Score)[:\s]*(\d+(?:\.\d+)?)", response_text)
        if score_match:
            return {
                "score": float(score_match.group(1)),
                "category": "Unknown",
                "reasoning": response_text[:200],
            }

        raise ValueError(f"Could not parse judge response: {response_text[:200]}")

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY, min=1, max=10),
    )
    async def _make_request(self, prompt: str) -> str:
        """Make an API request to the judge model."""
        payload = {
            "model": self.judge_model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 300,
            "temperature": 0.0,
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result = response.json()
            return result["choices"][0]["message"]["content"]

    async def judge_response(
        self, challenge: Challenge, model_response: ModelResponse
    ) -> JudgeScore:
        """Judge a single model response."""
        if not model_response.success:
            return JudgeScore(
                challenge_id=challenge.id,
                model_id=model_response.model_id,
                model_response=model_response.response_text,
                correct_answer=challenge.correct_answer,
                score=0.0,
                category="Failed",
                reasoning="Model failed to generate a response",
                success=False,
                error=model_response.error,
            )

        try:
            prompt = self._build_judge_prompt(
                correct_answer=challenge.correct_answer,
                model_response=model_response.response_text,
            )
            response_text = await self._make_request(prompt)
            parsed = self._parse_judge_response(response_text)

            return JudgeScore(
                challenge_id=challenge.id,
                model_id=model_response.model_id,
                model_response=model_response.response_text,
                correct_answer=challenge.correct_answer,
                score=float(parsed.get("score", 0)),
                category=str(parsed.get("category", "Unknown")),
                reasoning=str(parsed.get("reasoning", "")),
                success=True,
            )

        except Exception as e:
            return JudgeScore(
                challenge_id=challenge.id,
                model_id=model_response.model_id,
                model_response=model_response.response_text,
                correct_answer=challenge.correct_answer,
                score=0.0,
                category="Error",
                reasoning="",
                success=False,
                error=str(e),
            )

    async def judge_all_responses(
        self,
        challenges: list[Challenge],
        model_responses: dict[str, list[ModelResponse]],
    ) -> dict[str, ModelScores]:
        """Judge all responses for all models."""
        results: dict[str, ModelScores] = {}

        # Create a lookup for challenges by ID
        challenge_lookup = {c.id: c for c in challenges}

        total_judgments = sum(len(responses) for responses in model_responses.values())
        completed = 0

        for model_name, responses in model_responses.items():
            print(f"\nJudging {model_name} responses...")
            model_scores = ModelScores(model_name=model_name)

            for response in responses:
                challenge = challenge_lookup.get(response.challenge_id)
                if not challenge:
                    continue

                score = await self.judge_response(challenge, response)
                model_scores.scores.append(score)
                completed += 1

                status = f"Score: {score.score}" if score.success else f"FAILED: {score.error}"
                print(f"  [{completed}/{total_judgments}] {response.challenge_id}: {status}")

                # Small delay between requests
                await asyncio.sleep(0.3)

            results[model_name] = model_scores

        return results
