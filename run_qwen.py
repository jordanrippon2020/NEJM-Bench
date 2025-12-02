"""Run Qwen3-VL-235B on the same 50 challenges from the original benchmark."""

import asyncio
import json
from pathlib import Path

import httpx

from config import (
    OPENROUTER_API_KEY,
    OPENROUTER_BASE_URL,
    SYSTEM_PROMPT,
    USER_PROMPT_TEMPLATE,
    JUDGE_MODEL,
    JUDGE_PROMPT_TEMPLATE,
)

QWEN_MODEL = "qwen/qwen3-vl-235b-a22b-thinking"


async def download_image(session: httpx.AsyncClient, url: str) -> str | None:
    """Download image and return base64."""
    import base64
    try:
        resp = await session.get(url, timeout=30.0)
        if resp.status_code == 200:
            return base64.b64encode(resp.content).decode("utf-8")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
    return None


async def query_qwen(
    session: httpx.AsyncClient,
    headers: dict,
    challenge: dict,
    image_b64: str,
) -> str | None:
    """Query Qwen with a challenge."""
    user_prompt = USER_PROMPT_TEMPLATE.format(
        clinical_description=challenge["clinical_description"]
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
                },
            ],
        },
    ]

    payload = {
        "model": QWEN_MODEL,
        "messages": messages,
        "temperature": 0.6,  # Qwen thinking models need temp > 0
        "max_tokens": 32000,
    }

    try:
        resp = await session.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=300.0,  # Longer timeout for thinking model
        )
        if resp.status_code == 200:
            result = resp.json()
            return result["choices"][0]["message"]["content"]
        else:
            print(f"Error {resp.status_code}: {resp.text[:200]}")
    except Exception as e:
        print(f"Request error: {e}")
    return None


async def judge_response(
    session: httpx.AsyncClient,
    headers: dict,
    correct_answer: str,
    model_response: str,
) -> dict | None:
    """Judge a model response."""
    prompt = JUDGE_PROMPT_TEMPLATE.format(
        correct_answer=correct_answer,
        model_response=model_response,
    )

    payload = {
        "model": JUDGE_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 500,
    }

    try:
        resp = await session.post(
            f"{OPENROUTER_BASE_URL}/chat/completions",
            headers=headers,
            json=payload,
            timeout=60.0,
        )
        if resp.status_code == 200:
            result = resp.json()
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
    except Exception as e:
        print(f"Judge error: {e}")
    return None


async def main():
    # Load original benchmark results to get the same challenges
    results_path = Path("results/results_11d90b03.json")
    with open(results_path) as f:
        original_results = json.load(f)

    challenges = original_results["challenges"]
    print(f"Running Qwen3-VL-235B-Thinking on {len(challenges)} challenges...")

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/nejm-benchmark",
        "X-Title": "NEJM Image Challenge Benchmark",
    }

    results = []
    scores = []

    async with httpx.AsyncClient() as session:
        for i, challenge in enumerate(challenges):
            print(f"\n[{i+1}/{len(challenges)}] Challenge {challenge['id']}: {challenge['correct_answer'][:30]}...")

            # Download image
            image_b64 = await download_image(session, challenge["image_url"])
            if not image_b64:
                print("  Failed to download image, skipping")
                continue

            # Query Qwen
            response = await query_qwen(session, headers, challenge, image_b64)
            if not response:
                print("  Failed to get response, skipping")
                continue

            # Extract just the diagnosis (Qwen thinking models include reasoning)
            # Take last line or look for diagnosis
            response_clean = response.strip()
            print(f"  Qwen response: {response_clean[:80]}...")

            # Judge response
            judgment = await judge_response(
                session, headers, challenge["correct_answer"], response_clean
            )
            if judgment:
                score = judgment.get("score", 0)
                category = judgment.get("category", "unknown")
                print(f"  Score: {score}/10 ({category})")
                scores.append(score)
                results.append({
                    "challenge_id": challenge["id"],
                    "correct_answer": challenge["correct_answer"],
                    "qwen_response": response_clean,
                    "score": score,
                    "category": category,
                    "reasoning": judgment.get("reasoning", ""),
                })
            else:
                print("  Failed to judge response")

            # Small delay
            await asyncio.sleep(0.5)

    # Print summary
    print("\n" + "=" * 60)
    print("QWEN3-VL-235B-THINKING RESULTS")
    print("=" * 60)
    if scores:
        avg = sum(scores) / len(scores)
        print(f"Average Score: {avg:.2f}/10")
        print(f"Evaluated: {len(scores)}/{len(challenges)}")

        # Score distribution
        dist = {}
        for s in scores:
            dist[s] = dist.get(s, 0) + 1

        exact = dist.get(10, 0)
        high = sum(dist.get(i, 0) for i in range(7, 10))
        partial = sum(dist.get(i, 0) for i in range(4, 7))
        low = sum(dist.get(i, 0) for i in range(1, 4))
        wrong = dist.get(0, 0)

        print(f"Exact (10): {exact}")
        print(f"High (7-9): {high}")
        print(f"Partial (4-6): {partial}")
        print(f"Low (1-3): {low}")
        print(f"Wrong (0): {wrong}")

    # Save results
    output_path = Path("results/qwen_results.json")
    with open(output_path, "w") as f:
        json.dump({
            "model": QWEN_MODEL,
            "average_score": sum(scores) / len(scores) if scores else 0,
            "total_evaluated": len(scores),
            "results": results,
        }, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
