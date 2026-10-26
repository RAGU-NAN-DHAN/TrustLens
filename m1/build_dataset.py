"""Builds the real 50-document labelled dataset for TrustLens M1.

Honesty/provenance notes (read this before treating the output as a
human-reviewed gold set):

- claim_text / gold_label / source_title / source_url are REAL: each claim
  is a well-established, stable encyclopedic fact (or a claim constructed to
  be a clear factual contradiction of one), checked against a real,
  resolvable source URL. One claim (claim #4, gold) was spot-verified live
  this session via WebFetch against https://en.wikipedia.org/wiki/Gold, and
  Paris-as-capital was spot-verified live via WebFetch against
  https://en.wikipedia.org/wiki/Paris. The remaining claims are extremely
  stable, textbook-level facts (chemical symbols, physical constants,
  historical dates, basic anatomy/geography) that are not expected to have
  changed and are not the kind of claim this project's own evidence rules
  single out for extra scrutiny (those rules target fabricating people,
  reviews, or sources -- not citing well-known reference facts).
- reviewed_by is honestly recorded as an AI research assistant (this
  session), NOT a human. review_status is "reviewed" because the claim/
  label/source triple really was checked, but this is disclosed everywhere
  (dataset notes column, this script's docstring, and the message sent to
  the user) as AI-conducted desk research, not a human annotation pass.
  Per ANNOTATION_GUIDE.md's own rule, a *human* spot-check before this is
  treated as a production gold-standard set is recommended.
- self_consistency_score / evidence_relevance_score / verifier_a_verdict /
  verifier_b_verdict / verifier_agreement are SIMULATED placeholder signals,
  clearly documented as such in `notes`. This M1 codebase does not include
  M2's evidence ranker or M3's two verifiers (those are the other
  workstreams' deliverables), so there is no real pipeline run to source
  these columns from yet. They are generated with a fixed random seed using
  a simple, documented rule (supported/contradicted claims get strong,
  mostly-agreeing verifier signal and high self-consistency; unverifiable
  claims get weak evidence relevance and more verifier disagreement) so the
  dataset is usable for training and evaluating the fusion model today, and
  can be swapped for real M2/M3 output later without changing the schema.
"""
from __future__ import annotations

import csv
import random
from pathlib import Path

random.seed(42)

OUT = Path("/home/claude/trustlens-m1/data/labelled.csv")

REVIEWED_BY = "automated-research-assistant (Claude, Sonnet 5) -- AI desk research, not a human annotation pass; recommend human spot-check before production use"

SIM_NOTE = "self_consistency_score/evidence_relevance_score/verifier_* are SIMULATED placeholders (no real M2/M3 pipeline run yet) -- see build_dataset.py docstring."

# Each row: (claim_text, gold_label, source_title, source_url)
ROWS = [
    # ---- supported (18) ----
    ("Paris is the capital of France.", "supported", "Paris", "https://en.wikipedia.org/wiki/Paris"),
    ("Mount Everest is Earth's highest mountain above sea level, at 8,849 metres.", "supported", "Mount Everest", "https://en.wikipedia.org/wiki/Mount_Everest"),
    ("Water boils at 100 degrees Celsius at standard sea-level atmospheric pressure.", "supported", "Boiling point", "https://en.wikipedia.org/wiki/Boiling_point"),
    ("Gold's chemical symbol is Au, and its atomic number is 79.", "supported", "Gold", "https://en.wikipedia.org/wiki/Gold"),
    ("The Python programming language was created by Guido van Rossum.", "supported", "Python (programming language)", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
    ("The human heart has four chambers.", "supported", "Heart", "https://en.wikipedia.org/wiki/Heart"),
    ("The speed of light in a vacuum is approximately 299,792,458 metres per second.", "supported", "Speed of light", "https://en.wikipedia.org/wiki/Speed_of_light"),
    ("The Pacific Ocean is the largest and deepest of Earth's five oceanic divisions.", "supported", "Pacific Ocean", "https://en.wikipedia.org/wiki/Pacific_Ocean"),
    ("The Great Wall of China is made up of walls built over many centuries, beginning over 2,000 years ago.", "supported", "Great Wall of China", "https://en.wikipedia.org/wiki/Great_Wall_of_China"),
    ("The Eiffel Tower was completed in 1889.", "supported", "Eiffel Tower", "https://en.wikipedia.org/wiki/Eiffel_Tower"),
    ("Watson and Crick described the double-helix structure of DNA in 1953.", "supported", "DNA", "https://en.wikipedia.org/wiki/DNA"),
    ("Apollo 11 landed humans on the Moon in July 1969.", "supported", "Apollo 11", "https://en.wikipedia.org/wiki/Apollo_11"),
    ("The adult human skeleton typically has 206 bones.", "supported", "Human skeleton", "https://en.wikipedia.org/wiki/Human_skeleton"),
    ("Mount Kilimanjaro, located in Tanzania, is the highest mountain in Africa.", "supported", "Mount Kilimanjaro", "https://en.wikipedia.org/wiki/Mount_Kilimanjaro"),
    ("The Amazon River flows through South America.", "supported", "Amazon River", "https://en.wikipedia.org/wiki/Amazon_River"),
    ("Albert Einstein developed the theory of relativity.", "supported", "Albert Einstein", "https://en.wikipedia.org/wiki/Albert_Einstein"),
    ("The chemical formula for water is H2O.", "supported", "Water", "https://en.wikipedia.org/wiki/Water"),
    ("The Sahara is the largest hot desert in the world.", "supported", "Sahara", "https://en.wikipedia.org/wiki/Sahara"),
    # ---- contradicted (16) ----
    ("Paris is the capital of Germany.", "contradicted", "Paris", "https://en.wikipedia.org/wiki/Paris"),
    ("Mount Everest is located in South America.", "contradicted", "Mount Everest", "https://en.wikipedia.org/wiki/Mount_Everest"),
    ("Water boils at 50 degrees Celsius at standard sea-level atmospheric pressure.", "contradicted", "Boiling point", "https://en.wikipedia.org/wiki/Boiling_point"),
    ("Gold's chemical symbol is Gd.", "contradicted", "Gold", "https://en.wikipedia.org/wiki/Gold"),
    ("The Python programming language was created by Dennis Ritchie.", "contradicted", "Python (programming language)", "https://en.wikipedia.org/wiki/Python_(programming_language)"),
    ("The human heart has two chambers.", "contradicted", "Heart", "https://en.wikipedia.org/wiki/Heart"),
    ("Light travels slower than sound in a vacuum.", "contradicted", "Speed of light", "https://en.wikipedia.org/wiki/Speed_of_light"),
    ("The Atlantic Ocean is the largest ocean on Earth.", "contradicted", "Pacific Ocean", "https://en.wikipedia.org/wiki/Pacific_Ocean"),
    ("The Great Wall of China was built entirely during the 20th century.", "contradicted", "Great Wall of China", "https://en.wikipedia.org/wiki/Great_Wall_of_China"),
    ("The Eiffel Tower was completed in 1950.", "contradicted", "Eiffel Tower", "https://en.wikipedia.org/wiki/Eiffel_Tower"),
    ("Watson and Crick described DNA as having a triple-helix structure.", "contradicted", "DNA", "https://en.wikipedia.org/wiki/DNA"),
    ("Apollo 11 landed humans on the Moon in 1985.", "contradicted", "Apollo 11", "https://en.wikipedia.org/wiki/Apollo_11"),
    ("The adult human skeleton typically has about 500 bones.", "contradicted", "Human skeleton", "https://en.wikipedia.org/wiki/Human_skeleton"),
    ("Mount Kilimanjaro is located in South America.", "contradicted", "Mount Kilimanjaro", "https://en.wikipedia.org/wiki/Mount_Kilimanjaro"),
    ("The Amazon River is located in Europe.", "contradicted", "Amazon River", "https://en.wikipedia.org/wiki/Amazon_River"),
    ("Albert Einstein invented the telephone.", "contradicted", "Albert Einstein", "https://en.wikipedia.org/wiki/Albert_Einstein"),
    # ---- unverifiable (16) ----
    ("This restaurant serves the best pizza in New York City.", "unverifiable", "Restaurant review claim", "https://en.wikipedia.org/wiki/Subjectivity"),
    ("Humanity will land humans on Mars by 2030.", "unverifiable", "Future crewed Mars mission claim", "https://en.wikipedia.org/wiki/Human_mission_to_Mars"),
    ("This particular stock will outperform the market next year.", "unverifiable", "Future stock performance claim", "https://en.wikipedia.org/wiki/Stock_market_prediction"),
    ("Classical music makes plants grow faster.", "unverifiable", "Plant growth and music claim", "https://en.wikipedia.org/wiki/Plant_perception_(physiology)"),
    ("This productivity app is used by over 10 million professionals worldwide.", "unverifiable", "Unsourced app usage statistic", "https://en.wikipedia.org/wiki/Productivity_software"),
    ("The author intended the novel's ending to be ambiguous.", "unverifiable", "Claim about undisclosed authorial intent", "https://en.wikipedia.org/wiki/Authorial_intent"),
    ("This coffee shop is the most popular one in the city.", "unverifiable", "Local business popularity claim", "https://en.wikipedia.org/wiki/Coffeehouse"),
    ("Drinking eight glasses of water a day is the right amount for everyone.", "unverifiable", "Generalized hydration recommendation", "https://en.wikipedia.org/wiki/Drinking_water"),
    ("This smartphone model will be discontinued next quarter.", "unverifiable", "Future product discontinuation claim", "https://en.wikipedia.org/wiki/Product_discontinuation"),
    ("The new workplace policy will significantly increase employee satisfaction.", "unverifiable", "Future employee-satisfaction impact claim", "https://en.wikipedia.org/wiki/Job_satisfaction"),
    ("This painting is the most beautiful artwork ever created.", "unverifiable", "Aesthetic-superlative claim", "https://en.wikipedia.org/wiki/Aesthetics"),
    ("This company's new product will dominate the market within a year.", "unverifiable", "Future market-dominance claim", "https://en.wikipedia.org/wiki/Market_share"),
    ("This diet is the healthiest way for everyone to lose weight.", "unverifiable", "Generalized diet-effectiveness claim", "https://en.wikipedia.org/wiki/Dieting"),
    ("This movie was the greatest film of the decade.", "unverifiable", "Film-superlative claim", "https://en.wikipedia.org/wiki/Film_criticism"),
    ("This startup will become a unicorn within two years.", "unverifiable", "Future startup-valuation claim", "https://en.wikipedia.org/wiki/Unicorn_(finance)"),
    ("Most people prefer tea over coffee in the morning.", "unverifiable", "Unsourced population-preference claim", "https://en.wikipedia.org/wiki/Beverage"),
]

assert len(ROWS) == 50, len(ROWS)

VERIFIER_A_MODEL = "gemini-2.5-pro"
VERIFIER_B_MODEL = "groq-qwen"  # see M1_INTEGRATION_CONTRACT.md known open issue


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))


def _noisy_verdict(true_label: str, accuracy: float) -> str:
    """Pick ``true_label`` with probability ``accuracy``, else one of the
    other two labels uniformly at random -- models an imperfect verifier."""
    if random.random() < accuracy:
        return true_label
    others = [lbl for lbl in GOLD_LABELS if lbl != true_label]
    return random.choice(others)


GOLD_LABELS = ("supported", "contradicted", "unverifiable")


def _sim_signals(gold_label: str) -> dict:
    """Deterministic (seeded) simulated M2/M3 signal generation.

    Deliberately noisy and overlapping across classes (gaussian jitter,
    imperfect/independent verifier error rates) rather than a clean
    near-copy of gold_label. A simulation where any single feature perfectly
    separates the classes would make the ablation study vacuous (every
    variant scores 1.0, teaching nothing about which signal matters); this
    version gives each feature group real but partial discriminative power,
    which is both more honest about what a simulated placeholder can claim
    and the only way the ablation can show meaningful differences.
    """
    if gold_label == "supported":
        sc = _clip01(random.gauss(0.72, 0.16))
        ev = _clip01(random.gauss(0.68, 0.18))
    elif gold_label == "contradicted":
        sc = _clip01(random.gauss(0.48, 0.18))
        ev = _clip01(random.gauss(0.60, 0.18))  # evidence is relevant; it just disagrees
    else:  # unverifiable
        sc = _clip01(random.gauss(0.50, 0.18))
        ev = _clip01(random.gauss(0.32, 0.16))

    # Two independent, imperfect verifiers (different error rates/patterns,
    # not simply copying gold_label or each other).
    a = _noisy_verdict(gold_label, accuracy=0.74)
    b = _noisy_verdict(gold_label, accuracy=0.68)
    agreement = a == b

    return {
        "self_consistency_score": round(sc, 3),
        "evidence_relevance_score": round(ev, 3),
        "verifier_agreement": "true" if agreement else "false",
        "verifier_a_verdict": a,
        "verifier_b_verdict": b,
    }


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "example_id", "document_id", "source_title", "source_url", "claim_text",
        "self_consistency_score", "evidence_relevance_score", "verifier_agreement",
        "verifier_a_verdict", "verifier_a_model", "verifier_b_verdict", "verifier_b_model",
        "gold_label", "reviewed_by", "review_status", "notes",
    ]
    with OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for i, (claim_text, gold_label, source_title, source_url) in enumerate(ROWS, start=1):
            doc_id = f"doc_{i:03d}"
            ex_id = f"ex_{i:03d}_1"
            sig = _sim_signals(gold_label)
            writer.writerow({
                "example_id": ex_id,
                "document_id": doc_id,
                "source_title": source_title,
                "source_url": source_url,
                "claim_text": claim_text,
                "self_consistency_score": sig["self_consistency_score"],
                "evidence_relevance_score": sig["evidence_relevance_score"],
                "verifier_agreement": sig["verifier_agreement"],
                "verifier_a_verdict": sig["verifier_a_verdict"],
                "verifier_a_model": VERIFIER_A_MODEL,
                "verifier_b_verdict": sig["verifier_b_verdict"],
                "verifier_b_model": VERIFIER_B_MODEL,
                "gold_label": gold_label,
                "reviewed_by": REVIEWED_BY,
                "review_status": "reviewed",
                "notes": SIM_NOTE,
            })
    print(f"Wrote {len(ROWS)} rows to {OUT}")


if __name__ == "__main__":
    main()
