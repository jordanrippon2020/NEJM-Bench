"""Retry failed Gemini queries and update results."""

import asyncio
import json
from pathlib import Path

from config import MODELS
from dataset import NEJMDataset
from judge import JudgeClient
from models import OpenRouterClient

# Failed challenge IDs from the benchmark run
FAILED_CHALLENGES = ["934", "52", "389", "635", "762"]
RESULTS_FILE = "results/results_5888bf48.json"


async def retry_gemini():
    """Retry Gemini 3 Pro on failed challenges."""
    print("=" * 60)
    print("Retrying Gemini 3 Pro on 5 failed challenges")
    print("=" * 60)

    # Load dataset
    dataset = NEJMDataset()
    await dataset.load()

    # Find the failed challenges
    failed_challenges = [c for c in dataset.challenges if c.id in FAILED_CHALLENGES]
    print(f"Found {len(failed_challenges)} challenges to retry")

    # Download images for these challenges
    await dataset.prepare_challenges(failed_challenges)

    # Query Gemini
    client = OpenRouterClient()
    model_id = MODELS["gemini-3-pro"]
    print(f"\nQuerying {model_id}...")

    responses = []
    for i, challenge in enumerate(failed_challenges):
        print(f"  [{i+1}/{len(failed_challenges)}] Challenge {challenge.id}...")
        response = await client.query_model(model_id, challenge)
        responses.append(response)
        status = "OK" if response.success else f"FAILED: {response.error}"
        print(f"    {status}")
        await asyncio.sleep(1)  # Rate limiting

    # Score responses
    judge = JudgeClient()
    print("\nScoring responses...")
    scores = []
    for response in responses:
        challenge = next(c for c in failed_challenges if c.id == response.challenge_id)
        score = await judge.judge_response(challenge, response)
        scores.append(score)
        if score.success:
            icon = "+" if score.score == 1 else "x"
            print(f"  [{icon}] {challenge.id}: Selected {score.selected_letter}, Correct: {score.correct_answer_letter}")
        else:
            print(f"  [!] {challenge.id}: {score.error}")

    # Calculate results
    successful = [s for s in scores if s.success]
    correct = sum(s.score for s in successful)
    print(f"\nRetry Results: {correct}/{len(successful)} correct")

    # Load existing results and update
    results_path = Path(RESULTS_FILE)
    if results_path.exists():
        with open(results_path, "r") as f:
            results = json.load(f)

        # Update Gemini responses
        gemini_responses = results["responses"]["gemini-3-pro"]
        gemini_scores = results["scores"]["gemini-3-pro"]

        for response, score in zip(responses, scores):
            if not response.success:
                continue

            # Find and update the response
            for i, r in enumerate(gemini_responses):
                if r["challenge_id"] == response.challenge_id:
                    gemini_responses[i] = {
                        "challenge_id": response.challenge_id,
                        "response_text": response.response_text,
                        "success": response.success,
                        "error": response.error,
                    }
                    break

            # Find and update the score
            for i, s in enumerate(gemini_scores["individual_scores"]):
                if s["challenge_id"] == score.challenge_id:
                    gemini_scores["individual_scores"][i] = {
                        "challenge_id": score.challenge_id,
                        "score": score.score,
                        "category": score.category,
                        "selected_letter": score.selected_letter,
                        "correct_answer_letter": score.correct_answer_letter,
                        "correct_answer_text": score.correct_answer_text,
                        "model_response": score.model_response,
                        "success": score.success,
                        "error": score.error,
                    }
                    break

        # Recalculate Gemini totals
        valid_scores = [s for s in gemini_scores["individual_scores"] if s["success"]]
        correct_count = sum(s["score"] for s in valid_scores)
        total_count = len(valid_scores)
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0

        gemini_scores["accuracy"] = accuracy
        gemini_scores["correct_count"] = correct_count
        gemini_scores["total_count"] = total_count
        gemini_scores["category_breakdown"] = {
            "Correct": sum(1 for s in valid_scores if s["category"] == "Correct"),
            "Incorrect": sum(1 for s in valid_scores if s["category"] == "Incorrect"),
            "Invalid": sum(1 for s in valid_scores if s["category"] == "Invalid"),
        }

        # Save updated results
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nUpdated {results_path}")

        # Print final summary
        print("\n" + "=" * 60)
        print("UPDATED RESULTS")
        print("=" * 60)
        for model_name, model_scores in results["scores"].items():
            acc = model_scores["accuracy"]
            correct = model_scores["correct_count"]
            total = model_scores["total_count"]
            print(f"{model_name}: {acc:.1f}% ({correct}/{total})")
    else:
        print(f"Results file not found: {results_path}")


if __name__ == "__main__":
    asyncio.run(retry_gemini())
