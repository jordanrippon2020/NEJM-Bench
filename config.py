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
SYSTEM_PROMPT = """You are an expert physician. Given a clinical image and patient history, provide your most likely diagnosis. Be specific and concise. Do not hedge or list multiple possibilities - give your single best answer."""

USER_PROMPT_TEMPLATE = """Clinical Context: {clinical_description}

What is the most likely diagnosis?"""

JUDGE_PROMPT_TEMPLATE = """You are an expert medical educator evaluating AI diagnostic responses.

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
