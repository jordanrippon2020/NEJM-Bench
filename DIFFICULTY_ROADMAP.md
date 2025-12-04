# NEJM Benchmark Difficulty Roadmap

## Current Status (Run 5888bf48)

**Hard Mode Results (Brier >= 0.087, ~50 cases):**
| Model | Accuracy | vs Physician Baseline |
|-------|----------|----------------------|
| GPT-5.1 | 68.0% | +38% above baseline |
| Claude Opus 4.5 | 68.0% | +38% above baseline |
| Gemini 3 Pro | 38.0% | +8% above baseline |

**Physician Baseline:** ~30% accuracy on hard cases (Brier >= 0.087)

**Problem:** GPT-5.1 and Claude are still performing well above physician baseline. Target is 22-45% to create meaningful differentiation.

---

## Strategies to Increase Difficulty

### 1. Expert Tier (Brier >= 0.123)
**Status:** Ready to implement
**Expected Impact:** High

Switch to expert-only cases where physicians average ~27% accuracy.

```python
# config.py
DIFFICULTY_MODE = "expert"  # Instead of "hard"
```

**Pros:**
- Straightforward change
- Cases where even specialists struggle
- ~150 cases available

**Cons:**
- Smaller sample pool
- May hit floor effects for weaker models

---

### 2. Ablate Extended Thinking
**Status:** Easy to test
**Expected Impact:** Medium-High

Current config gives models maximum reasoning:
- GPT-5.1: `reasoning.effort = "high"`, 64K completion tokens
- Claude: `reasoning.effort = "high"`, 64K tokens
- Gemini: `temperature = 1.0`, 64K tokens

**Test Configuration:**
```python
# models.py - disable extended thinking
if "gpt-5" in model_id:
    payload["max_tokens"] = 500  # Force short response
    # Remove reasoning parameter

if "claude" in model_id:
    payload["max_tokens"] = 500
    # Remove reasoning parameter
```

**Expected:** 10-20% accuracy drop without deep reasoning

---

### 3. Time Pressure Simulation
**Status:** Medium effort
**Expected Impact:** Medium

Add instruction that simulates clinical time pressure:

```python
SYSTEM_PROMPT = """You are an emergency physician with 30 seconds to make a diagnosis.
Given the image and brief history, select the most likely diagnosis immediately.
Do not deliberate - trust your first instinct. Respond with only the letter."""
```

---

### 4. Incomplete Information
**Status:** Medium effort
**Expected Impact:** Medium-High

Remove or truncate clinical context to simulate real diagnostic uncertainty:

```python
# Option A: Remove clinical description entirely
USER_PROMPT = """Based on this clinical image alone, select the most likely diagnosis:
A) {option_A}
...
"""

# Option B: Truncate to first sentence only
clinical_brief = challenge.clinical_description.split('.')[0] + '.'
```

---

### 5. Adversarial Distractors
**Status:** High effort (requires dataset modification)
**Expected Impact:** High

Replace easy distractor options with plausible alternatives:
- Current: Options often include obviously wrong choices
- Improved: All 5 options should be diagnostically plausible given the image

**Implementation:**
1. Use GPT-4 to generate 4 plausible alternatives per case
2. Validate that alternatives are medically reasonable
3. Create new dataset variant

---

### 6. Multi-Image Cases
**Status:** High effort
**Expected Impact:** Medium

Some NEJM challenges have multiple images. Currently we only use the first.
- Require models to synthesize across 2-3 images
- Tests visual integration, not just pattern recognition

---

### 7. Differential Diagnosis Format
**Status:** Medium effort
**Expected Impact:** Medium

Instead of "select one answer," require ranked differentials:

```python
USER_PROMPT = """Rank all 5 diagnoses from most to least likely.
Format: 1. [letter] 2. [letter] 3. [letter] 4. [letter] 5. [letter]"""

# Scoring:
# 1st place correct = 1.0
# 2nd place correct = 0.5
# 3rd place correct = 0.25
# etc.
```

---

### 8. Remove Answer Letter Hints
**Status:** Low effort
**Expected Impact:** Low-Medium

Some models may have memorized NEJM challenges with answers. Test by:
1. Shuffling option order (A becomes C, etc.)
2. Using numeric options (1-5) instead of letters

---

## Recommended Implementation Order

### Phase 1: Quick Wins
1. **Expert tier** - Just change config, immediate 10-15% drop expected
2. **Ablate reasoning** - Test without extended thinking

### Phase 2: Prompt Engineering
3. **Time pressure prompt** - Simple prompt change
4. **Incomplete information** - Remove clinical context

### Phase 3: Dataset Modifications
5. **Shuffle options** - Detect memorization
6. **Adversarial distractors** - Requires new dataset version

---

## Tracking Experiments

| Experiment | Config Change | GPT-5.1 | Claude | Gemini | Notes |
|------------|--------------|---------|--------|--------|-------|
| Baseline (hard) | Brier >= 0.087 | 68.0% | 68.0% | 38.0% | Current |
| Expert tier | Brier >= 0.123 | ? | ? | ? | TODO |
| No reasoning | Remove thinking | ? | ? | ? | TODO |
| No context | Image only | ? | ? | ? | TODO |
| Shuffled options | Randomize A-E | ? | ? | ? | TODO |

---

## Success Criteria

The benchmark achieves its goal when:
1. **Top models score 35-50%** (above physician baseline but not dominant)
2. **Clear differentiation** between models (>10% spread)
3. **Meaningful failures** - errors are diagnostically plausible, not random

---

*Last updated: 2024-12-04*
