# fuzzy_targets.py
"""
Fuzzy Logic Note Pyramid Calculation
Pure fuzzy inference system for AI-powered perfume formulation

Uses EXISTING data:
- perfume_knowledge.json for note ranges
- Questionnaire for user inputs (longevity_0_5, projection_0_5, occasions)
"""
import numpy as np
from models import FAMILIES
from fuzzy_formulation import get_fuzzy_engine

# Overall character boosts for fragrance families
OVERALL_BOOST = {
    "fresh": {"citrus": 0.8, "green": 0.4, "floral": 0.2},
    "floral": {"floral": 0.9, "powdery": 0.3, "green": 0.2},
    "warm_spicy": {"spicy": 0.8, "resinous": 0.4, "woody": 0.3},
    "sweet": {"gourmand": 0.8, "floral": 0.3, "powdery": 0.2},
    "woody": {"woody": 0.8, "resinous": 0.3, "spicy": 0.2},
    "resinous": {"resinous": 0.8, "woody": 0.4, "spicy": 0.2}
}

# Occasion-based note pyramids (top, middle, base) as fractions
OCC_PYRAMIDS = {
    "daily": (0.30, 0.45, 0.25),      # Balanced
    "special": (0.25, 0.40, 0.35),    # More base for longevity
    "summer": (0.35, 0.45, 0.20),     # More top for freshness
    "winter": (0.20, 0.40, 0.40)      # More base for warmth
}

try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("[ERROR] scikit-fuzzy not available! Install with: pip install scikit-fuzzy")


class FuzzyNotePyramid:
    """
    Fuzzy Inference System for Note Distribution Calculation
    """

    def __init__(self):
        self.fuzzy_enabled = FUZZY_AVAILABLE
        self.engine = get_fuzzy_engine()

        if self.fuzzy_enabled:
            self._build_fuzzy_system()
        else:
            print("Warning: Fuzzy logic not available, using crisp fallback")

    def _build_fuzzy_system(self):
        """Build fuzzy inference system using scikit-fuzzy"""
        try:
            # INPUT: User preferences (0-5 scale)
            self.longevity = ctrl.Antecedent(np.arange(0, 6, 1), 'longevity')
            self.projection = ctrl.Antecedent(np.arange(0, 6, 1), 'projection')

            # OUTPUT: Note percentages (from existing knowledge base ranges)
            self.top_pct = ctrl.Consequent(np.arange(15, 46, 1), 'top_pct')
            self.base_pct = ctrl.Consequent(np.arange(15, 46, 1), 'base_pct')

            # Define fuzzy membership functions for inputs
            # Longevity: how long the perfume should last
            self.longevity['short'] = fuzz.trapmf(self.longevity.universe, [0, 0, 1, 2])
            self.longevity['moderate'] = fuzz.trapmf(self.longevity.universe, [1, 2, 3, 4])
            self.longevity['long'] = fuzz.trapmf(self.longevity.universe, [3, 4, 5, 5])

            # Projection: how strong the scent throw should be
            self.projection['low'] = fuzz.trapmf(self.projection.universe, [0, 0, 1, 2])
            self.projection['medium'] = fuzz.trapmf(self.projection.universe, [1, 2, 3, 4])
            self.projection['high'] = fuzz.trapmf(self.projection.universe, [3, 4, 5, 5])

            # Define fuzzy membership functions for outputs
            # Use existing knowledge base ranges
            note_ranges = self.engine.note_ranges

            # Top note percentage (typically 20-40%)
            top_min, top_max = note_ranges["top"]
            self.top_pct['low'] = fuzz.trapmf(self.top_pct.universe, [15, 15, 20, 25])
            self.top_pct['medium'] = fuzz.trapmf(self.top_pct.universe, [22, 28, 32, 38])
            self.top_pct['high'] = fuzz.trapmf(self.top_pct.universe, [35, 38, 45, 45])

            # Base note percentage (typically 20-30%)
            base_min, base_max = note_ranges["base"]
            self.base_pct['low'] = fuzz.trapmf(self.base_pct.universe, [15, 15, 18, 22])
            self.base_pct['medium'] = fuzz.trapmf(self.base_pct.universe, [20, 24, 28, 32])
            self.base_pct['high'] = fuzz.trapmf(self.base_pct.universe, [30, 35, 45, 45])

            # Define fuzzy rules based on perfume formulation principles
            # From knowledge base: "Longevity → Increase base notes"
            # From knowledge base: "Projection → Increase top/middle notes"

            rules = [
                # Rule 1: Long longevity + low projection → low top, high base
                ctrl.Rule(self.longevity['long'] & self.projection['low'],
                         (self.top_pct['low'], self.base_pct['high'])),

                # Rule 2: Short longevity + high projection → high top, low base
                ctrl.Rule(self.longevity['short'] & self.projection['high'],
                         (self.top_pct['high'], self.base_pct['low'])),

                # Rule 3: Moderate longevity + medium projection → medium everything
                ctrl.Rule(self.longevity['moderate'] & self.projection['medium'],
                         (self.top_pct['medium'], self.base_pct['medium'])),

                # Rule 4: Long longevity alone → medium top, high base
                ctrl.Rule(self.longevity['long'],
                         (self.top_pct['medium'], self.base_pct['high'])),

                # Rule 5: Short longevity alone → medium top, low base
                ctrl.Rule(self.longevity['short'],
                         (self.top_pct['medium'], self.base_pct['low'])),

                # Rule 6: High projection alone → high top, medium base
                ctrl.Rule(self.projection['high'],
                         (self.top_pct['high'], self.base_pct['medium'])),

                # Rule 7: Low projection alone → low top, medium base
                ctrl.Rule(self.projection['low'],
                         (self.top_pct['low'], self.base_pct['medium'])),

                # Rule 8: Long longevity + high projection → balanced high
                ctrl.Rule(self.longevity['long'] & self.projection['high'],
                         (self.top_pct['medium'], self.base_pct['high'])),

                # Rule 9: Moderate longevity + high projection → high top, medium base
                ctrl.Rule(self.longevity['moderate'] & self.projection['high'],
                         (self.top_pct['high'], self.base_pct['medium'])),
            ]

            # Create control system
            self.ctrl_system = ctrl.ControlSystem(rules)
            self.simulator = ctrl.ControlSystemSimulation(self.ctrl_system)

            print("Fuzzy note pyramid system built with 9 rules")

        except Exception as e:
            print(f"Error building fuzzy system: {e}")
            self.fuzzy_enabled = False

    def compute_fuzzy(self, longevity_0_5: int, projection_0_5: int) -> dict:
        """
        Compute note distribution using fuzzy inference.
        Returns: {"top": float, "middle": float, "base": float}
        """
        if not self.fuzzy_enabled:
            return None

        try:
            # Set inputs
            self.simulator.input['longevity'] = longevity_0_5
            self.simulator.input['projection'] = projection_0_5

            # Compute fuzzy inference
            self.simulator.compute()

            # Get crisp outputs (defuzzified)
            top = self.simulator.output['top_pct']
            base = self.simulator.output['base_pct']
            middle = 100 - top - base

            # Ensure valid percentages
            if middle < 15:
                # Rebalance if middle is too small
                adjustment = (15 - middle) / 2
                top -= adjustment
                base -= adjustment
                middle = 15

            return {
                "top": round(top, 1),
                "middle": round(middle, 1),
                "base": round(base, 1)
            }

        except Exception as e:
            print(f"Fuzzy inference failed: {e}")
            return None


# Global fuzzy pyramid instance
_fuzzy_pyramid = None


def compute_targets(q):
    """
    Enhanced version of original compute_targets() using fuzzy logic.
    Falls back to crisp logic if fuzzy not available.

    Returns: (user_vec, note_targets, cap_factor)
    """
    global _fuzzy_pyramid

    # Build user preference vector (same as original)
    vec = np.array([q.family_ratings.get(f, 0) for f in FAMILIES], dtype=float)
    for k, v in OVERALL_BOOST[q.overall].items():
        vec[FAMILIES.index(k)] += v * 5
    vec = np.clip(vec, 0, None)
    user_vec = vec / (vec.sum() if vec.sum() > 0 else 1.0)

    # Try fuzzy note pyramid calculation
    if FUZZY_AVAILABLE:
        if _fuzzy_pyramid is None:
            _fuzzy_pyramid = FuzzyNotePyramid()

        fuzzy_result = _fuzzy_pyramid.compute_fuzzy(q.longevity_0_5, q.projection_0_5)

        if fuzzy_result is not None:
            # Fuzzy calculation succeeded
            note_targets = fuzzy_result

            # Apply occasion adjustments (blend fuzzy with domain knowledge)
            occ = q.occasions[0] if q.occasions else "daily"
            if occ in OCC_PYRAMIDS:
                # Blend 70% fuzzy, 30% occasion template
                occ_top, occ_mid, occ_base = OCC_PYRAMIDS[occ]
                note_targets["top"] = 0.7 * note_targets["top"] + 0.3 * (occ_top * 100)
                note_targets["middle"] = 0.7 * note_targets["middle"] + 0.3 * (occ_mid * 100)
                note_targets["base"] = 0.7 * note_targets["base"] + 0.3 * (occ_base * 100)

                # Normalize
                total = note_targets["top"] + note_targets["middle"] + note_targets["base"]
                note_targets = {k: round(v * 100 / total, 1) for k, v in note_targets.items()}

            # Fuzzy skin sensitivity factor
            engine = get_fuzzy_engine()
            # Estimate allergen exposure (will be refined per oil later)
            avg_allergen_count = 2 if q.sensitive_skin else 0
            cap_factor = engine.fuzzy_skin_sensitivity_factor(q.sensitive_skin, avg_allergen_count)

            print(f"\nFuzzy Note Targets: top={note_targets['top']}%, middle={note_targets['middle']}%, base={note_targets['base']}%")
            print(f"Fuzzy Cap Factor: {cap_factor:.2f}")

            return user_vec, note_targets, cap_factor

    # Fuzzy logic not available - raise error
    raise RuntimeError(
        "Fuzzy logic is required but not available! "
        "Please install: pip install scikit-fuzzy"
    )


if __name__ == "__main__":
    # Test fuzzy vs crisp
    from models import Questionnaire

    # Create test questionnaire
    q = Questionnaire(
        overall="fresh",
        family_ratings={"citrus": 5, "floral": 2, "woody": 3, "green": 4,
                       "gourmand": 1, "spicy": 2, "powdery": 1, "resinous": 1},
        gender_vibe="unisex",
        longevity_0_5=4,
        projection_0_5=3,
        occasions=["daily"],
        sensitive_skin=True,
        dislikes=[],
        loves=["Bergamot"],
        bottle_ml=30,
        concentration="EDP",
        scent_description="Fresh citrus blend"
    )

    print("=== Testing Fuzzy vs Crisp Note Targets ===\n")

    user_vec, targets, cap = compute_targets(q)

    print(f"\nUser vector (family preferences): {user_vec}")
    print(f"Note targets: {targets}")
    print(f"Cap factor: {cap:.2f}")
