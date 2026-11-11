"""All AI calls live here. To switch provider (Claude, OpenAI), only change this file."""
import json
import os
import time

CATEGORIES = ["Hardware", "Software", "Network", "Email", "Security",
              "Accounts & Access", "Cloud & Servers", "Other"]
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash")
# Used if the main model stays overloaded (503) after retries
FALLBACK_MODEL = os.getenv("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash-lite")
BUSY_MARKERS = ("503", "UNAVAILABLE", "429", "RESOURCE_EXHAUSTED", "overloaded")


def enabled() -> bool:
    return bool(os.getenv("GEMINI_API_KEY"))


def _generate(prompt, config=None):
    """Call Gemini: 3 tries on the main model, then the fallback model once."""
    from google import genai

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    last_error = None
    for model, wait in [(MODEL, 1), (MODEL, 2), (MODEL, 4), (FALLBACK_MODEL, 0)]:
        try:
            return client.models.generate_content(model=model, contents=prompt, config=config)
        except Exception as e:
            last_error = e
            if not any(m in str(e) for m in BUSY_MARKERS):
                raise  # a real error (bad key, bad request): do not retry
            time.sleep(wait)
    raise last_error


def analyze(subject: str, description: str) -> dict:
    """Return {category, priority, summary}. Raises on any failure; callers handle it."""
    from google.genai import types

    prompt = (
        "You are an IT support triage assistant. Classify this support ticket.\n"
        "Priority guide: High = outage, security risk, or many users blocked; "
        "Medium = one user blocked or degraded service; Low = request or minor issue.\n"
        "Summary: one plain sentence, max 25 words.\n\n"
        f"Subject: {subject}\nDescription: {description or '(none)'}"
    )
    config = types.GenerateContentConfig(
        response_mime_type="application/json",
        response_schema={
            "type": "object",
            "properties": {
                "category": {"type": "string", "enum": CATEGORIES},
                "priority": {"type": "string", "enum": ["Low", "Medium", "High"]},
                "summary": {"type": "string"},
            },
            "required": ["category", "priority", "summary"],
        },
    )
    data = json.loads(_generate(prompt, config).text)
    if data.get("category") not in CATEGORIES:
        data["category"] = "Other"
    if data.get("priority") not in ("Low", "Medium", "High"):
        data["priority"] = "Medium"
    data["summary"] = str(data.get("summary", ""))[:300]
    return data


def weekly_report(stats: str) -> str:
    """Turn raw numbers into a short manager-friendly report."""
    prompt = (
        "Write a short weekly operations report for an IT support manager, based only on "
        "these numbers. Plain text, no markdown symbols, under 150 words, with four short "
        "parts: Overview, Trends, Risks, Recommended actions. Do not invent numbers.\n\n" + stats
    )
    return _generate(prompt).text.strip()
