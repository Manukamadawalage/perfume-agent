# logic_dispense.py
"""
Dispense Plan Generation
Converts recipe percentages to physical amounts (ml, grams, time)
"""
import json
import os

# Concentration ratios: percentage of bottle that is fragrance concentrate
CONCENTRATION_RATIOS = {
    "EdC": 0.05,     # Eau de Cologne: 5% concentrate
    "EDT": 0.10,     # Eau de Toilette: 10%
    "EDP": 0.20,     # Eau de Parfum: 20%
    "Parfum": 0.30   # Parfum/Extrait: 30%
}


def to_dispense_plan(recipe_items, bottle_ml, concentration, catalog, valve_map):
    """
    Convert recipe percentages to physical dispensing plan.

    Args:
        recipe_items: List of {"oil_name": str, "percent": float, "note": str}
        bottle_ml: Total bottle volume (ml)
        concentration: "EdC", "EDT", "EDP", or "Parfum"
        catalog: Catalog (for oil densities)
        valve_map: Dict mapping oil names to valve IDs

    Returns:
        List of {
            "oil_name": str,
            "valve": int,
            "percent": float,
            "ml": float,
            "grams": float,
            "time_ms": int,
            "note": str
        }
    """
    # Build oil lookup
    oil_by_name = {oil.name: oil for oil in catalog.oils}

    # Calculate concentrate volume
    ratio = CONCENTRATION_RATIOS.get(concentration, 0.15)
    concentrate_ml = bottle_ml * ratio

    # Load calibration data if available
    calibration = {}
    if os.path.exists("valve_calibration.json"):
        with open("valve_calibration.json") as f:
            calibration = json.load(f)

    def _sanitize_key(valve_id, oil_name):
        return f"valve_{valve_id}_{oil_name.replace(' ', '_').replace('(', '').replace(')', '').replace('%', '')}"

    def _flow_rate_for(valve_id, oil_name):
        """
        Support both schemas:
        - New: {"<valve_id>": {"flow_rate_g_s": 0.85}}
        - Existing: {"valve_<id>_<oil>": 0.85}
        """
        if not calibration:
            return None

        # Schema 1: per-valve dict
        valve_entry = calibration.get(str(valve_id))
        if isinstance(valve_entry, dict) and "flow_rate_g_s" in valve_entry:
            return valve_entry.get("flow_rate_g_s")
        if isinstance(valve_entry, (int, float)):
            return float(valve_entry)

        # Schema 2: per-valve+oil key
        legacy_key = _sanitize_key(valve_id, oil_name)
        legacy_val = calibration.get(legacy_key)
        if isinstance(legacy_val, (int, float)):
            return float(legacy_val)

        return None

    dispense_plan = []

    for item in recipe_items:
        oil_name = item["oil_name"]
        percent = item["percent"]
        note = item["note"]

        # Get oil from catalog
        if oil_name not in oil_by_name:
            print(f"[WARNING] Oil '{oil_name}' not in catalog")
            continue

        oil = oil_by_name[oil_name]

        # Calculate ml for this oil
        ml = (percent / 100.0) * concentrate_ml

        # Calculate grams using density
        grams = ml * oil.density_g_ml

        # Get valve ID
        valve_id = valve_map.get(oil_name, -1)

        # Calculate dispense time
        flow_rate_g_s = _flow_rate_for(valve_id, oil_name)
        if flow_rate_g_s and flow_rate_g_s > 0:
            time_s = grams / flow_rate_g_s
            time_ms = int(time_s * 1000)
        else:
            # Default: assume 0.1 g/s flow rate
            time_ms = int((grams / 0.1) * 1000)

        dispense_plan.append({
            "oil_name": oil_name,
            "valve": valve_id,
            "percent": round(percent, 2),
            "ml": round(ml, 3),
            "grams": round(grams, 3),
            "time_ms": time_ms,
            "note": note
        })

    return dispense_plan


if __name__ == "__main__":
    # Test
    import json
    from models import Catalog, Oil

    with open("oils.json") as f:
        data = json.load(f)
        catalog = Catalog(oils=[Oil(**o) for o in data["oils"]])

    valve_map = {}
    if os.path.exists("valve_map.json"):
        with open("valve_map.json") as f:
            valve_map = json.load(f)

    recipe = [
        {"oil_name": "Bergamot (expressed)", "percent": 12.0, "note": "top"},
        {"oil_name": "Lavender", "percent": 20.0, "note": "middle"},
        {"oil_name": "Sandalwood", "percent": 18.0, "note": "base"},
    ]

    plan = to_dispense_plan(
        recipe_items=recipe,
        bottle_ml=30,
        concentration="EDP",
        catalog=catalog,
        valve_map=valve_map
    )

    print("=== Dispense Plan ===")
    print(f"Bottle: 30ml EDP (20% concentrate = 6ml)")
    print()
    for item in plan:
        print(f"{item['oil_name']:30s} | {item['percent']:5.1f}% | {item['ml']:6.3f}ml | {item['grams']:6.3f}g | Valve {item['valve']:2d} | {item['time_ms']:5d}ms")
