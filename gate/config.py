"""Fixed configuration: seeds, knobs, models, prices.

Models are configurable through environment variables (plan section 5).
Prices are USD per 1M tokens, from the official z.ai price list, checked
2026-10-13 (sources: z.ai announcement via openrouter.ai/z-ai/glm-5.3-flash
and press coverage). Raw token counts are always recorded alongside USD, so
prices can be recomputed if they change.
"""

import os

# Partition (plan section 5): split train by task index with a fixed seed.
SPLIT_SEED = 0
FIT_SIZE = 250  # fit = first 250 shuffled train indices; cal = rest + dev

# Models and provider. Agent = flash: the 20-task pilots (2026-10-13,
# results/collect/*pilot*) showed glm-5.3 errs on only 4% of gated actions
# on train (too few positives to calibrate) while glm-5.3-flash errs on
# ~12%, close to the historical sonnet-3.5 base rate (21%), at a tenth of
# the price. Both pilots also showed verbalized and agreement saturated
# near 1.0 (see notes/progress.md, Phase 2).
AGENT_MODEL = os.environ.get("GATE_AGENT_MODEL", "glm-5.3-flash")
USER_MODEL = os.environ.get("GATE_USER_MODEL", "glm-5.3-flash")
PROVIDER = os.environ.get("GATE_MODEL_PROVIDER", "anthropic")
# Endpoint actually used, recorded in every Phase 6 result row. Phase 2 used
# z.ai's Anthropic-compatible API; Phase 6 used z.ai's OpenAI-compatible API.
ENDPOINT = (
    os.environ.get("OPENAI_API_BASE") if PROVIDER == "openai"
    else os.environ.get("ANTHROPIC_BASE_URL")
)

# Signals (plan 4.4).
AGREEMENT_K = 5
AGREEMENT_TEMPERATURE = 0.7

# Runaway guard, not a budget (Youssef, 2026-10-13: no spending cap).
MAX_USD = float(os.environ.get("GATE_MAX_USD", "1000"))

PRICES = {
    "glm-5.3": {"input": 1.40, "output": 4.40, "cache_read": 0.26},
    "glm-5.3-flash": {"input": 0.15, "output": 0.50, "cache_read": 0.03},
}

# Disk cache for deterministic (temperature-0) LLM calls.
CACHE_DIR = os.environ.get("GATE_CACHE_DIR", ".cache/llm")


def fit_cal_ids():
    """The fit/cal partition of plan section 5: shuffle train task indices
    with the fixed seed; fit = first FIT_SIZE, cal = the rest plus dev.

    Returns (fit_ids, cal_train_ids); dev (indices 0..19 of the dev split)
    always belongs to cal and is added by the callers.
    """
    import random

    idx = list(range(500))
    random.Random(SPLIT_SEED).shuffle(idx)
    return idx[:FIT_SIZE], idx[FIT_SIZE:]
