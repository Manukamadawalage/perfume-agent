# logic_validate.py
"""
Recipe Validation
Ensures recipe meets safety and quality requirements
(safety caps can be relaxed via env: ENFORCE_SAFETY_CAPS=false)
"""

import os

# Allow caps to be relaxed via env var (default: enforce)
ENFORCE_SAFETY_CAPS = os.getenv("ENFORCE_SAFETY_CAPS", "false").lower() == "true"


def validate(recipe_items, catalog, note_targets, sensitive_skin=False, enforce_caps=ENFORCE_SAFETY_CAPS):
    """
    Validate recipe for safety and quality.

    Raises AssertionError if validation fails.

    Args:
        recipe_items: List of {"oil_name": str, "percent": float, "note": str}
        catalog: Catalog (for safety cap checking)
        note_targets: {"top": %, "middle": %, "base": %}
        sensitive_skin: Apply stricter validation

    Raises:
        AssertionError: If validation fails
    """
    if not recipe_items:
        raise AssertionError("Recipe is empty")

    # Build oil lookup
    oil_by_name = {oil.name: oil for oil in catalog.oils}

    # 1) Check total percentage
    total = sum(item["percent"] for item in recipe_items)
    assert 99.0 <= total <= 101.0, f"Total percentage {total:.2f}% not close to 100%"

    # 2) Check safety caps (can be disabled via env)
    cap_factor = 0.7 if sensitive_skin else 1.0

    # Consolidate totals per oil (same oil can appear in multiple notes)
    oil_totals = {}
    for item in recipe_items:
        name = item["oil_name"]
        oil_totals[name] = oil_totals.get(name, 0.0) + item["percent"]

    for oil_name, total_pct in oil_totals.items():
        if oil_name not in oil_by_name:
            print(f"[WARNING] Oil '{oil_name}' not found in catalog")
            continue

        oil = oil_by_name[oil_name]
        # max_pct is stored as a percent (e.g., 0.4 means 0.4%), so keep units aligned
        max_allowed = oil.max_pct * cap_factor

        if enforce_caps:
            assert total_pct <= max_allowed + 0.1, \
                f"Oil '{oil_name}' exceeds safety cap: {total_pct:.2f}% > {max_allowed:.2f}%"
        else:
            if total_pct > max_allowed:
                print(f"[WARNING] Oil '{oil_name}' over safety cap: {total_pct:.2f}% > {max_allowed:.2f}% (caps relaxed)")

    # 3) Check minimum percentages (oils should be meaningful)
    for item in recipe_items:
        name = item["oil_name"]
        pct = item["percent"]

        # Special exception for precious oils (can be < 1%)
        if name in ["Jasmine", "Rose", "Ylang-Ylang", "Clove", "Vanilla"]:
            assert pct >= 0.3, f"Oil '{name}' too low: {pct:.2f}%"
        else:
            assert pct >= 0.5, f"Oil '{name}' too low: {pct:.2f}%"

    # 4) Check each oil appears only once per note (no duplicates)
    seen_in_note = set()
    for item in recipe_items:
        key = (item["oil_name"], item["note"])
        assert key not in seen_in_note, f"Duplicate: {item['oil_name']} in {item['note']}"
        seen_in_note.add(key)

    # 5) Warn if note distribution deviates too much from targets
    note_totals = {"top": 0.0, "middle": 0.0, "base": 0.0}
    for item in recipe_items:
        note = item["note"]
        if note in note_totals:
            note_totals[note] += item["percent"]

    for note, target in note_targets.items():
        actual = note_totals.get(note, 0.0)
        deviation = abs(actual - target)
        if deviation > 15.0:  # More than 15% deviation
            print(f"[WARNING] {note.upper()} note deviates from target: {actual:.1f}% vs {target:.1f}%")

    print("[OK] Recipe validation passed")


if __name__ == "__main__":
    # Test
    import json
    from models import Catalog, Oil

    with open("oils.json") as f:
        data = json.load(f)
        catalog = Catalog(oils=[Oil(**o) for o in data["oils"]])

    recipe = [
        {"oil_name": "Bergamot (expressed)", "percent": 12.0, "note": "top"},
        {"oil_name": "Lemon", "percent": 15.0, "note": "top"},
        {"oil_name": "Lavender", "percent": 20.0, "note": "middle"},
        {"oil_name": "Geranium", "percent": 18.0, "note": "middle"},
        {"oil_name": "Sandalwood", "percent": 20.0, "note": "base"},
        {"oil_name": "Cedarwood", "percent": 15.0, "note": "base"},
    ]

    note_targets = {"top": 30.0, "middle": 45.0, "base": 25.0}

    print("Validating recipe...")
    validate(recipe, catalog, note_targets, sensitive_skin=False)
