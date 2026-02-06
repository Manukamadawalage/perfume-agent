# fuzzy_select.py
"""
Fuzzy Logic Oil Selection and Quality Assessment
Replaces logic_select.py with fuzzy quality evaluation

Uses EXISTING data:
- oils.json for oil properties (role_weights, features, max_pct)
- No new data files needed
"""
import numpy as np
from models import FAMILIES
from fuzzy_formulation import get_fuzzy_engine


def name_matches(name, words):
    """Check if oil name contains any of the words (from original)"""
    n = name.lower()
    return any(w.lower() in n for w in words)


def rank_oils_for_note(catalog, user_vec, note_idx, loves, dislikes, cap_factor, use_fuzzy=True):
    """
    Enhanced oil ranking using fuzzy logic.

    Args:
        catalog: Oil catalog
        user_vec: User preference vector (8D)
        note_idx: 0=top, 1=middle, 2=base
        loves: List of loved oil names
        dislikes: List of disliked oil names
        cap_factor: Safety cap multiplier
        use_fuzzy: Use fuzzy logic if True, else crisp

    Returns:
        List of (score, oil, eff_cap, fuzzy_quality) tuples, sorted by score
    """
    if not use_fuzzy:
        raise RuntimeError("Crisp logic disabled: fuzzy-only configuration in effect")

    fuzzy_engine = get_fuzzy_engine()
    ranked = []

    for oil in catalog.oils:
        # Exclude carriers (same as original)
        if getattr(oil, 'is_carrier', False):
            continue

        # Exclude dislikes (same as original)
        if name_matches(oil.name, dislikes):
            continue

        # Calculate effective cap
        eff_cap = oil.max_pct * cap_factor
        if eff_cap <= 0.05:
            continue

        # Calculate family similarity score
        f = np.array(oil.features, dtype=float)
        f_norm = f / (np.linalg.norm(f) + 1e-9)
        fam_score = float(np.dot(f_norm, user_vec))

        # Get role fit for this note
        role_bonus = oil.role_weights[note_idx]

        # Check if user loves this oil
        is_loved = name_matches(oil.name, loves)

        if use_fuzzy:
            # FUZZY QUALITY ASSESSMENT
            fuzzy_quality = fuzzy_engine.fuzzy_quality_assessment(
                family_match=fam_score,
                role_fit=role_bonus,
                user_loves=is_loved
            )

            # Defuzzify to get crisp score for ranking
            score = fuzzy_engine.defuzzify_quality(fuzzy_quality)

            # Store fuzzy quality for later use
            ranked.append((score, oil, eff_cap, fuzzy_quality))

        else:
            # CRISP SCORING (original algorithm)
            love_bonus = 0.1 if is_loved else 0.0
            score = 0.6 * fam_score + 0.3 * role_bonus + 0.1 * love_bonus

            # No fuzzy quality in crisp mode
            ranked.append((score, oil, eff_cap, None))

    # Sort by score (descending)
    ranked.sort(key=lambda x: x[0], reverse=True)

    # Print top oils for this note
    note_names = ["TOP", "MIDDLE", "BASE"]
    if ranked:
        print(f"\n  {note_names[note_idx]} Note - Top 5 oils:")
        for i, (score, oil, cap, fuzzy_q) in enumerate(ranked[:5]):
            if use_fuzzy and fuzzy_q:
                # Show fuzzy quality breakdown
                dominant = max(fuzzy_q.items(), key=lambda x: x[1])
                print(f"    {i+1}. {oil.name:30s} score={score:.3f} [{dominant[0]}:{dominant[1]:.2f}]")
            else:
                print(f"    {i+1}. {oil.name:30s} score={score:.3f}")

    return ranked


def filter_by_fuzzy_quality(ranked_oils, min_acceptable_membership=0.5):
    """
    Filter oils using fuzzy quality memberships.
    Replaces crisp threshold: if score >= 0.10

    Args:
        ranked_oils: List from rank_oils_for_note()
                    Can be 3-tuple (score, oil, eff_cap) or 4-tuple with fuzzy_quality
        min_acceptable_membership: Minimum membership in acceptable+ categories

    Returns:
        Filtered list of (score, oil, eff_cap, fuzzy_quality)
    """
    fuzzy_engine = get_fuzzy_engine()
    filtered = []

    for item in ranked_oils:
        # Handle both 3-tuple (from AI ranking) and 4-tuple (from fuzzy ranking)
        if len(item) == 3:
            score, oil, eff_cap = item
            fuzzy_quality = None
        elif len(item) == 4:
            score, oil, eff_cap, fuzzy_quality = item
        else:
            continue  # Skip malformed items
        if fuzzy_quality is None:
            # Crisp mode - use score threshold
            if score >= 0.10:
                filtered.append((score, oil, eff_cap, fuzzy_quality))
        else:
            # Fuzzy mode - check if oil should be included
            if fuzzy_engine.should_include_oil(fuzzy_quality, min_quality_threshold=0.25):
                filtered.append((score, oil, eff_cap, fuzzy_quality))

    return filtered


def get_oil_quality_explanation(fuzzy_quality: dict) -> str:
    """
    Generate human-readable explanation of fuzzy quality assessment.
    For better interpretability.

    Args:
        fuzzy_quality: Dict with memberships {poor, acceptable, good, excellent}

    Returns:
        Human-readable string
    """
    if fuzzy_quality is None:
        return "N/A (crisp mode)"

    # Find dominant quality level
    sorted_qualities = sorted(fuzzy_quality.items(), key=lambda x: x[1], reverse=True)

    parts = []
    for quality, membership in sorted_qualities:
        if membership > 0.1:  # Only show significant memberships
            percentage = int(membership * 100)
            parts.append(f"{percentage}% {quality}")

    return ", ".join(parts) if parts else "undefined quality"


if __name__ == "__main__":
    # Test fuzzy oil selection
    import json
    from models import Catalog, Oil

    # Load catalog
    with open("oils.json") as f:
        data = json.load(f)
        catalog = Catalog(oils=[Oil(**o) for o in data["oils"]])

    # Test user preference vector (prefer citrus and fresh)
    user_vec = np.array([0.5, 0.1, 0.2, 0.3, 0.05, 0.05, 0.0, 0.0])  # citrus dominant
    user_vec = user_vec / user_vec.sum()

    print("=== Testing Fuzzy Oil Selection ===")
    print(f"User preferences: Citrus (50%), Green (30%), Woody (20%)\n")

    # Test for top notes
    ranked = rank_oils_for_note(
        catalog=catalog,
        user_vec=user_vec,
        note_idx=0,  # Top note
        loves=["Bergamot"],
        dislikes=["Jasmine"],
        cap_factor=0.8,
        use_fuzzy=True
    )

    print(f"\nTotal ranked: {len(ranked)} oils")

    # Filter using fuzzy quality
    filtered = filter_by_fuzzy_quality(ranked, min_acceptable_membership=0.5)
    print(f"After fuzzy filtering: {len(filtered)} oils")

    # Show quality explanations for top 3
    print("\nQuality Explanations:")
    for i, (score, oil, cap, fuzzy_q) in enumerate(filtered[:3]):
        explanation = get_oil_quality_explanation(fuzzy_q)
        print(f"  {i+1}. {oil.name}: {explanation}")
