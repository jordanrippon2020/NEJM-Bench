"""Configuration for NEJM Image Challenge Benchmark."""

import os
from dotenv import load_dotenv

load_dotenv()

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Models to benchmark (correct OpenRouter model IDs)
MODELS = {
    "gpt-5.1": "openai/gpt-5.1",
    "gemini-3-pro": "google/gemini-3-pro-preview",
    "claude-opus-4.5": "anthropic/claude-opus-4.5",
}

# Judge model (separate from benchmarked models to avoid bias)
JUDGE_MODEL = "anthropic/claude-sonnet-4"

# Benchmark settings
SAMPLE_SIZE = 50
RANDOM_SEED = 42  # For reproducibility

# Difficulty settings
# - "all": No filtering, use all cases (easy benchmark)
# - "hard": Brier >= 0.087, physician accuracy ~30% (challenging benchmark)
# - "expert": Brier >= 0.123, physician accuracy ~27% (very hard benchmark)
DIFFICULTY_MODE = "hard"

BRIER_THRESHOLDS = {
    "easy": 0.040,      # Cases where physicians get ~74% correct
    "medium": 0.087,    # Cases where physicians get ~50% correct
    "hard": 0.087,      # >= this threshold for hard mode (~30% physician accuracy)
    "expert": 0.123,    # >= this threshold for expert mode (~27% physician accuracy)
}

# Dataset URLs
DATASET_URL = "https://raw.githubusercontent.com/cx0/nejm-image-challenge/main/image_challenge_dataset_20231223.json"

# Paths
CACHE_DIR = "cache"
RESULTS_DIR = "results"

# API settings
MAX_RETRIES = 3
RETRY_DELAY = 1.0  # seconds
REQUEST_TIMEOUT = 300.0  # seconds (5 min for extended thinking with max tokens)

# Prompts
SYSTEM_PROMPT = """You are an expert physician taking a diagnostic image challenge. Given a clinical image and patient history, select the most likely diagnosis from the options provided. Respond with only the letter (A, B, C, D, or E)."""

USER_PROMPT_TEMPLATE = """Clinical Context: {clinical_description}

Based on the clinical image, select the most likely diagnosis:
A) {option_A}
B) {option_B}
C) {option_C}
D) {option_D}
E) {option_E}

Respond with only the letter (A, B, C, D, or E)."""

# Legacy judge prompt - kept for reference but no longer used with binary scoring
JUDGE_PROMPT_TEMPLATE_LEGACY = """You are an expert medical educator evaluating AI diagnostic responses.

Correct Diagnosis: {correct_answer}
Model's Response: {model_response}

Score this response on a 0-10 scale based on diagnostic accuracy:

10 = Exact match (correct diagnosis, proper terminology)
9  = Synonymous (correct diagnosis, alternative valid medical term)
8  = More specific subtype that is still correct
7  = Correct broader category (e.g., "carcinoma" vs "squamous cell carcinoma")
5-6 = Related condition (same organ system/disease family, wrong specific diagnosis)
3-4 = Partially related (correct anatomical area or general pathology type)
1-2 = Wrong but medically plausible given the presentation
0  = Completely wrong or nonsensical

Respond with JSON only, no other text:
{{"score": <0-10>, "category": "<category name>", "reasoning": "<brief explanation>"}}"""

# New binary scoring - no judge needed, just letter matching
JUDGE_PROMPT_TEMPLATE = None  # Binary scoring doesn't need an LLM judge
