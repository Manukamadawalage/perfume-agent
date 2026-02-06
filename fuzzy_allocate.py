# fuzzy_allocate.py
"""
Fuzzy Logic Oil Allocation Strategy
Pure fuzzy allocation for AI-powered perfume formulation

Uses EXISTING data:
- oils.json for oil properties
- Fuzzy quality assessments from fuzzy_select.py
"""
from collections import defaultdict
from fuzzy_formulation import get_fuzzy_engine

# Essential oils that should be included even at lower fuzzy quality scores
ESSENTIAL_OILS = [
    "Frankincense",
    "Sandalwood",
    "Cedarwood",
    "Lavender"
]

# Precious oils that can be used in smaller amounts
PRECIOUS_OILS = [
    "Jasmine",
    "Rose",
    "Ylang-Ylang",
    "Clove",
    "Vanilla"
]


def fuzzy_recipe_complexity(occasion: str, concentration: str, user_experience: str = "beginner") -> int:
    """
    Fuzzy determination of recipe complexity (max oils).
    Replaces fixed: max_oils = 12

    Args:
        occasion: daily, special, summer, winter
        concentration: EdC, EDT, EDP, Parfum
        user_experience: beginner, intermediate, advanced

    Returns:
        Max oils for recipe (8-16)
    """
    # Fuzzy rules for complexity
    complexity_score = 0.0

    # Rule 1: Special occasions → more complex
    if occasion == "special":
        complexity_score += 0.4
    elif occasion == "daily":
        complexity_score += 0.1

    # Rule 2: Higher concentration → more complex
    if concentration == "Parfum":
        complexity_score += 0.4
    elif concentration == "EDP":
        complexity_score += 0.3
    elif concentration == "EDT":
        complexity_score += 0.2
    else:  # EdC
        complexity_score += 0.1

    # Rule 3: User experience
    if user_experience == "advanced":
        complexity_score += 0.3
    elif user_experience == "intermediate":
        complexity_score += 0.2
    else:  # beginner
        complexity_score += 0.0

    # Normalize to 0-1 range
    complexity_score = min(1.0, complexity_score)

    # Fuzzy sets for complexity
    if complexity_score < 0.3:
        # Simple recipe
        membership_simple = 1.0 - (complexity_score / 0.3)
        membership_moderate = complexity_score / 0.3
        membership_complex = 0.0
    elif complexity_score < 0.7:
        # Moderate recipe
        membership_simple = 0.0
        membership_moderate = 1.0 - abs(complexity_score - 0.5) / 0.2
        membership_complex = (complexity_score - 0.3) / 0.4
    else:
        # Complex recipe
        membership_simple = 0.0
        membership_moderate = 1.0 - (complexity_score - 0.7) / 0.3
        membership_complex = (complexity_score - 0.7) / 0.3

    # Defuzzify to max oils
    max_oils = int(
        membership_simple * 8 +
        membership_moderate * 12 +
        membership_complex * 16
    )

    return max(8, min(16, max_oils))


def fuzzy_min_percent(oil, fuzzy_strength: dict = None) -> float:
    """
    Fuzzy determination of minimum percentage for oil.
    Replaces fixed: min_percent = 0.8

    Args:
        oil: Oil object
        fuzzy_strength: Optional fuzzy strength assessment

    Returns:
        Minimum percentage for this oil
    """
    fuzzy_engine = get_fuzzy_engine()

    # Get oil strength (either from parameter or calculate)
    if fuzzy_strength is None:
        fuzzy_strength = fuzzy_engine.fuzzy_linguistic_strength(oil)

    # Fuzzy rules for minimum amount
    # Very strong oils (jasmine, rose) can be used sparingly
    # Weak oils need higher minimums to be noticeable

    min_pct = (
        fuzzy_strength.get("very_strong", 0) * 0.4 +
        fuzzy_strength.get("strong", 0) * 0.6 +
        fuzzy_strength.get("moderate", 0) * 0.8 +
        fuzzy_strength.get("weak", 0) * 1.2
    )

    # Additional rule: Precious oils can be used in smaller amounts
    if oil.name in PRECIOUS_OILS:
        min_pct = min(min_pct, 0.6)

    return min_pct


def allocate(note_targets, ranked_by_note, catalog, cap_factor: float,
             min_score=0.10, max_oils=None, min_percent=0.8,
             allow_essential_oils=True, use_fuzzy=True,
             occasion="daily", concentration="EDP"):
    """
    Enhanced allocation using fuzzy logic.

    Args:
        note_targets: {"top":..., "middle":..., "base":...} in %
        ranked_by_note: [ranked_top, ranked_mid, ranked_base] from fuzzy_select
        catalog: Catalog (for oil metadata)
        cap_factor: Fuzzy safety factor
        min_score: Minimum score (used if not fuzzy)
        max_oils: Maximum oils (None = auto-determine from fuzzy rules)
        min_percent: Minimum % (used if not fuzzy)
        allow_essential_oils: Allow essential oils below threshold
        use_fuzzy: Use fuzzy allocation strategy
        occasion: For fuzzy complexity calculation
        concentration: For fuzzy complexity calculation

    Returns:
        List of {"oil_name": str, "percent": float, "note": str, "fuzzy_quality": dict}
    """
    if not use_fuzzy:
        raise RuntimeError("Crisp allocation is disabled: fuzzy-only configuration in effect")

    fuzzy_engine = get_fuzzy_engine()

    # Fuzzy determination of max oils if not specified
    if max_oils is None and use_fuzzy:
        max_oils = fuzzy_recipe_complexity(occasion, concentration)
        print(f"Fuzzy recipe complexity -> max {max_oils} oils")
    elif max_oils is None:
        max_oils = 12  # Default

    # Track usage
    used_by_oil = defaultdict(float)
    result = []
    total_oils_used = set()

    # Filter ranked lists
    filtered_ranks = []
    for note_idx, ranked in enumerate(ranked_by_note):
        if use_fuzzy:
            # Use fuzzy quality filtering
            from fuzzy_select import filter_by_fuzzy_quality
            filtered = filter_by_fuzzy_quality(ranked, min_acceptable_membership=0.4)

            # Add essential oils even if quality is low
            if allow_essential_oils:
                for item in ranked:
                    # Handle both 3-tuple and 4-tuple
                    if len(item) == 3:
                        score, oil, cap = item
                        fuzzy_q = None
                    else:
                        score, oil, cap, fuzzy_q = item

                    if oil.name in ESSENTIAL_OILS:
                        # Check if not already in filtered
                        already_included = any(
                            f_oil.name == oil.name
                            for _, f_oil, _, _ in filtered
                        )
                        if not already_included:
                            filtered.append((score, oil, cap, fuzzy_q))
        else:
            # Crisp filtering - handle both tuple formats
            if allow_essential_oils:
                filtered = []
                for item in ranked:
                    if len(item) == 3:
                        score, oil, cap = item
                        fq = None
                    else:
                        score, oil, cap, fq = item
                    if score >= min_score or oil.name in ESSENTIAL_OILS:
                        filtered.append((score, oil, cap, fq))
            else:
                filtered = []
                for item in ranked:
                    if len(item) == 3:
                        score, oil, cap = item
                        fq = None
                    else:
                        score, oil, cap, fq = item
                    if score >= min_score:
                        filtered.append((score, oil, cap, fq))

        filtered_ranks.append(filtered)

        essential_count = len([o for s, o, c, fq in filtered if o.name in ESSENTIAL_OILS])
        print(f"  Note {note_idx}: {len(ranked)} oils ranked, {len(filtered)} passed fuzzy filter (+{essential_count} essential)")

    # Helper to add oil allocation
    def try_add(oil, note_name, remaining, is_topup=False):
        # Check oil limit
        if len(total_oils_used) >= max_oils and oil.name not in total_oils_used:
            return 0.0

        headroom = (oil.max_pct * cap_factor) - used_by_oil[oil.name]
        if headroom <= 0.05:
            return 0.0

        # Fuzzy minimum amount determination
        if use_fuzzy:
            effective_min = fuzzy_min_percent(oil)
        else:
            effective_min = min_percent
            if oil.name in PRECIOUS_OILS:
                effective_min = min(0.6, min_percent)

        # Do not demand more than the available headroom (important for low-cap oils)
        cap_limited_min = min(effective_min, headroom)
        if cap_limited_min <= 0:
            return 0.0

        # Propose allocation
        if is_topup:
            step = min(max(round(remaining * 0.3, 1), cap_limited_min), headroom, remaining)
        else:
            step = min(max(round(remaining * 0.5, 1), cap_limited_min), headroom, 20.0)

        if step <= 0 or step < cap_limited_min:
            return 0.0

        result.append({
            "oil_name": oil.name,
            "percent": float(step),
            "note": note_name
        })
        used_by_oil[oil.name] += step
        total_oils_used.add(oil.name)
        return step

    # Main allocation loop
    print(f"\nFuzzy Allocation (max {max_oils} oils):")

    for (note_name, note_idx) in [("top", 0), ("middle", 1), ("base", 2)]:
        remaining = note_targets[note_name]
        oils_in_note = 0
        max_oils_per_note = 6

        print(f"\n  {note_name.upper()} note (target: {remaining}%):")

        for score, oil, eff_cap, fuzzy_quality in filtered_ranks[note_idx]:
            if remaining <= 0.5:
                break
            if oils_in_note >= max_oils_per_note:
                break
            if len(total_oils_used) >= max_oils:
                break

            added = try_add(oil, note_name, remaining, is_topup=False)
            if added > 0:
                # Show fuzzy quality if available
                if use_fuzzy and fuzzy_quality:
                    dominant = max(fuzzy_quality.items(), key=lambda x: x[1])
                    quality_str = f"[{dominant[0]}:{dominant[1]:.2f}]"
                else:
                    quality_str = f"[score:{score:.3f}]"

                print(f"    + {oil.name:30s} {added:5.1f}% {quality_str}")
                remaining = round(remaining - added, 1)
                oils_in_note += 1

        # Selective top-up
        if remaining > 1.0 and len(total_oils_used) < max_oils:
            print(f"    Remaining: {remaining:.1f}% - selective top-up...")
            for score, oil, eff_cap, fuzzy_quality in filtered_ranks[note_idx][:3]:
                if remaining <= 0.5:
                    break
                if len(total_oils_used) >= max_oils:
                    break

                # Only top-up high quality oils
                if use_fuzzy and fuzzy_quality:
                    # Check if "good" or "excellent" membership is significant
                    if fuzzy_quality.get("good", 0) < 0.3 and fuzzy_quality.get("excellent", 0) < 0.1:
                        continue
                else:
                    if score < min_score * 2:
                        continue

                added = try_add(oil, note_name, remaining, is_topup=True)
                if added > 0:
                    print(f"    + {oil.name:30s} {added:5.1f}% (topup)")
                    remaining = round(remaining - added, 1)

    print(f"\n  Total oils used: {len(total_oils_used)}/{max_oils}")

    return result


if __name__ == "__main__":
    # Test fuzzy allocation
    import json
    from models import Catalog, Oil
    from fuzzy_select import rank_oils_for_note
    import numpy as np

    # Load catalog
    with open("oils.json") as f:
        data = json.load(f)
        catalog = Catalog(oils=[Oil(**o) for o in data["oils"]])

    # Test preferences
    user_vec = np.array([0.5, 0.1, 0.2, 0.3, 0.05, 0.05, 0.0, 0.0])
    user_vec = user_vec / user_vec.sum()

    # Rank oils for each note
    print("=== Testing Fuzzy Allocation ===\n")

    ranked_top = rank_oils_for_note(catalog, user_vec, 0, ["Bergamot"], [], 0.85, use_fuzzy=True)
    ranked_mid = rank_oils_for_note(catalog, user_vec, 1, [], [], 0.85, use_fuzzy=True)
    ranked_base = rank_oils_for_note(catalog, user_vec, 2, [], [], 0.85, use_fuzzy=True)

    note_targets = {"top": 30.0, "middle": 45.0, "base": 25.0}

    # Test fuzzy allocation
    recipe = allocate(
        note_targets=note_targets,
        ranked_by_note=[ranked_top, ranked_mid, ranked_base],
        catalog=catalog,
        cap_factor=0.85,
        use_fuzzy=True,
        occasion="special",
        concentration="EDP"
    )

    print("\n=== Final Recipe ===")
    for item in recipe:
        print(f"  {item['note']:6s} {item['oil_name']:30s} {item['percent']:5.1f}%")

    total = sum(item["percent"] for item in recipe)
    print(f"\nTotal: {total:.1f}%")
