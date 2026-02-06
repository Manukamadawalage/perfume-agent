# logic_consolidate.py
"""
Recipe Consolidation
Merges duplicate oils from different notes into single entries
"""
from collections import defaultdict


def consolidate_ingredients(recipe_items):
    """
    Consolidate duplicate oils across notes.

    Some oils (like Lavender) can score well in multiple notes.
    This function merges them into a single entry for efficient dispensing.

    Args:
        recipe_items: List of {"oil_name": str, "percent": float, "note": str}

    Returns:
        List of consolidated items with:
        - "oil_name": str
        - "percent": float (sum of all occurrences)
        - "note": str (primary note - where it appears most)
        - "notes_list": list of notes where it appears
    """
    if not recipe_items:
        return []

    # Group by oil name
    oil_data = defaultdict(lambda: {"total_percent": 0.0, "note_percentages": {}})

    for item in recipe_items:
        oil_name = item["oil_name"]
        percent = item["percent"]
        note = item["note"]

        oil_data[oil_name]["total_percent"] += percent
        oil_data[oil_name]["note_percentages"][note] = \
            oil_data[oil_name]["note_percentages"].get(note, 0.0) + percent

    # Build consolidated list
    consolidated = []

    for oil_name, data in oil_data.items():
        # Determine primary note (where it appears most)
        primary_note = max(data["note_percentages"].items(), key=lambda x: x[1])[0]

        # Get all notes where it appears
        notes_list = list(data["note_percentages"].keys())

        consolidated.append({
            "oil_name": oil_name,
            "percent": round(data["total_percent"], 2),
            "note": primary_note,
            "notes_list": notes_list,
            "reason": f"Consolidated from {len(notes_list)} notes" if len(notes_list) > 1 else "Single note"
        })

    # Sort by percentage (descending) for better readability
    consolidated.sort(key=lambda x: x["percent"], reverse=True)

    # Print consolidation summary
    original_count = len(recipe_items)
    consolidated_count = len(consolidated)
    if original_count > consolidated_count:
        print(f"\n[INFO] Consolidated {original_count} entries -> {consolidated_count} unique oils")
        for item in consolidated:
            if len(item["notes_list"]) > 1:
                notes_str = ", ".join(item["notes_list"])
                print(f"  - {item['oil_name']:30s} {item['percent']:5.1f}% (from {notes_str})")

    return consolidated


if __name__ == "__main__":
    # Test
    recipe = [
        {"oil_name": "Bergamot (expressed)", "percent": 12.0, "note": "top"},
        {"oil_name": "Lavender", "percent": 8.0, "note": "top"},
        {"oil_name": "Lavender", "percent": 12.0, "note": "middle"},  # Duplicate!
        {"oil_name": "Geranium", "percent": 15.0, "note": "middle"},
        {"oil_name": "Sandalwood", "percent": 18.0, "note": "base"},
        {"oil_name": "Cedarwood", "percent": 10.0, "note": "base"},
        {"oil_name": "Cedarwood", "percent": 5.0, "note": "middle"},  # Duplicate!
    ]

    print("Before consolidation:")
    for item in recipe:
        print(f"  {item['note']:6s} | {item['oil_name']:30s} | {item['percent']:5.1f}%")
    print(f"Total entries: {len(recipe)}")

    consolidated = consolidate_ingredients(recipe)

    print("\nAfter consolidation:")
    for item in consolidated:
        notes_str = ", ".join(item["notes_list"])
        print(f"  {item['note']:6s} | {item['oil_name']:30s} | {item['percent']:5.1f}% | [{notes_str}]")
    print(f"Total unique oils: {len(consolidated)}")
    print(f"Total percentage: {sum(item['percent'] for item in consolidated):.1f}%")
