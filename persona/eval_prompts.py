"""Fixed evaluation prompt set used by the sweep.

A mix of identity / opinion / how-to / explanation / creative / factual
prompts so that:
- persona shows up across response types (the persona signal generalises),
- coherence regressions are obvious (long-form generation breaks first),
- instruction-following can be judged (prompts have clear expected content).

Drawn from upstream's `data/simple_prompts.json` plus a curated subset of
AxBench-style instructions from `data/alpaca_eval_prompts.json`.
"""

EVAL_PROMPTS: list[str] = [
    # Identity / opinion (strongest persona signal)
    "Who are you?",
    "How do you feel today?",
    "Tell me about your favorite hobby.",
    "What do you think makes a good life?",
    # How-to / advice
    "Give me some ideas for starting a small business.",
    "What are three tips for staying focused while studying?",
    "How should I prepare for a job interview tomorrow?",
    "Suggest a healthy weekday dinner I can cook in 30 minutes.",
    # Explanation
    "Explain the theory of relativity in simple terms.",
    "What are the main causes of inflation?",
    "How does photosynthesis work?",
    "Briefly explain how a vaccine teaches the immune system.",
    # Creative
    "Write a short poem about the ocean.",
    "Tell a one-paragraph bedtime story for a five-year-old.",
    # Recommendation / list
    "Recommend three books for someone who liked The Hobbit.",
    "List five fun things to do in a city on a rainy weekend.",
    # Factual
    "What is the largest ocean in the world?",
    "Who painted the Mona Lisa and roughly when?",
]
