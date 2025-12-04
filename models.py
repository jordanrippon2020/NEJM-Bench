"""OpenRouter API client for querying vision models."""

import asyncio
from dataclasses import dataclass
from typing import Any

import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

from config import (
    MAX_RETRIES,
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    REQUEST_TIMEOUT,
    RETRY_DELAY,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
)
from dataset import Challenge


@dataclass
class ModelResponse:
    """Response from a model query."""

    model_id: str
    challenge_id: str
    response_text: str
    success: bool
    error: str | None = None
    usage: dict[str, Any] | None = None


class OpenRouterClient:
    """Async client for OpenRouter API with vision support."""

    def __init__(self, api_key: str = OPENROUTER_API_KEY):
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Check your .env file.")
        self.api_key = api_key
        self.base_url = OPENROUTER_BASE_URL

    def _get_headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://github.com/nejm-benchmark",
            "X-Title": "NEJM Image Challenge Benchmark",
        }

    def _build_vision_message(
        self, challenge: Challenge, image_base64: str
    ) -> list[dict[str, Any]]:
        """Build the message payload with image and text including multiple choice options."""
        user_prompt = USER_PROMPT_TEMPLATE.format(
            clinical_description=challenge.clinical_description,
            option_A=challenge.options.get("A", "N/A"),
            option_B=challenge.options.get("B", "N/A"),
            option_C=challenge.options.get("C", "N/A"),
            option_D=challenge.options.get("D", "N/A"),
            option_E=challenge.options.get("E", "N/A"),
        )

        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_base64}"
                        },
                    },
                ],
            },
        ]

    @retry(
        stop=stop_after_attempt(MAX_RETRIES),
        wait=wait_exponential(multiplier=RETRY_DELAY, min=1, max=10),
        reraise=True,
    )
    async def _make_request(
        self, model_id: str, messages: list[dict[str, Any]]
    ) -> dict[str, Any]:
        """Make an API request with retry logic."""
        # Base payload - model-specific settings will override these
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
        }

        # ========== GPT-5.1 Configuration ==========
        # Per OpenAI docs: max_completion_tokens up to 128K for reasoning models
        # - "high" reasoning effort for maximum intelligence/reliability
        # - include_reasoning=False to avoid org verification requirement
        if "gpt-5" in model_id:
            payload["temperature"] = 0.0  # Deterministic output
            payload["max_completion_tokens"] = 64000  # MAX: Allow full reasoning + response
            payload["include_reasoning"] = False  # Avoids org verification
            payload["reasoning"] = {"effort": "high"}  # Maximum reasoning depth

        # ========== Gemini 3 Pro Configuration ==========
        # Per Google docs: temperature MUST be 1.0, supports up to 64K output tokens
        # - thinking_level: "high" for maximum reasoning depth (default)
        elif "gemini" in model_id:
            payload["temperature"] = 1.0  # REQUIRED: Google recommends 1.0, lower causes issues
            payload["max_tokens"] = 64000  # MAX: Gemini 3 Pro supports up to 64K output
            # thinking_level defaults to "high" which is what we want

        # ========== Claude Opus 4.5 Configuration ==========
        # Per OpenRouter docs: Use reasoning parameter with effort for extended thinking
        # - "high" effort = budget_tokens up to 32K for reasoning
        # - max_tokens can go up to 64K with extended thinking
        # - temperature 1.0 recommended for reasoning models
        elif "claude" in model_id:
            payload["temperature"] = 1.0  # Recommended for reasoning models
            payload["max_tokens"] = 64000  # MAX: Allow full thinking + response
            payload["reasoning"] = {"effort": "high"}  # Extended thinking on high (32K budget)

        # ========== Default for other models ==========
        else:
            payload["temperature"] = 0.0
            payload["max_tokens"] = 2000

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers=self._get_headers(),
                json=payload,
                timeout=REQUEST_TIMEOUT,
            )
            if response.status_code != 200:
                error_detail = response.text[:500]
                raise httpx.HTTPStatusError(
                    f"HTTP {response.status_code}: {error_detail}",
                    request=response.request,
                    response=response,
                )
            return response.json()

    async def query_model(
        self, model_id: str, challenge: Challenge
    ) -> ModelResponse:
        """Query a model with an image challenge."""
        if not challenge._image_base64:
            return ModelResponse(
                model_id=model_id,
                challenge_id=challenge.id,
                response_text="",
                success=False,
                error="Image not downloaded for this challenge",
            )

        try:
            messages = self._build_vision_message(challenge, challenge._image_base64)
            result = await self._make_request(model_id, messages)

            response_text = result["choices"][0]["message"]["content"]
            usage = result.get("usage")

            return ModelResponse(
                model_id=model_id,
                challenge_id=challenge.id,
                response_text=response_text.strip(),
                success=True,
                usage=usage,
            )

        except httpx.HTTPStatusError as e:
            return ModelResponse(
                model_id=model_id,
                challenge_id=challenge.id,
                response_text="",
                success=False,
                error=f"HTTP {e.response.status_code}: {e.response.text[:200]}",
            )
        except Exception as e:
            return ModelResponse(
                model_id=model_id,
                challenge_id=challenge.id,
                response_text="",
                success=False,
                error=str(e),
            )

    async def query_all_models(
        self, models: dict[str, str], challenges: list[Challenge]
    ) -> dict[str, list[ModelResponse]]:
        """Query all models for all challenges."""
        results: dict[str, list[ModelResponse]] = {name: [] for name in models}

        total_queries = len(models) * len(challenges)
        completed = 0

        for model_name, model_id in models.items():
            print(f"\nQuerying {model_name} ({model_id})...")

            for i, challenge in enumerate(challenges):
                response = await self.query_model(model_id, challenge)
                results[model_name].append(response)
                completed += 1

                status = "OK" if response.success else f"FAILED: {response.error}"
                print(f"  [{completed}/{total_queries}] {challenge.id}: {status}")

                # Small delay between requests to avoid rate limiting
                await asyncio.sleep(0.5)

        return results


async def query_single_model(
    model_id: str, challenge: Challenge, image_base64: str
) -> ModelResponse:
    """Convenience function to query a single model."""
    client = OpenRouterClient()
    challenge._image_base64 = image_base64
    return await client.query_model(model_id, challenge)
