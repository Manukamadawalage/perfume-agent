# fuzzy_formulation.py
"""
Core Fuzzy Logic Engine for Perfume Formulation
Uses ONLY existing data from oils.json and perfume_knowledge.json

This module provides the foundation for fuzzy inference without needing new data files.
"""
import numpy as np
import json
import os
from typing import Dict, List, Tuple

# Try to import fuzzy logic library
try:
    import skfuzzy as fuzz
    from skfuzzy import control as ctrl
    FUZZY_AVAILABLE = True
except ImportError:
    FUZZY_AVAILABLE = False
    print("Warning: scikit-fuzzy not available. Install with: pip install scikit-fuzzy")


class FuzzyPerfumeEngine:
    """
    Main fuzzy logic engine that builds everything from existing JSON files.
    No new data files required!
    """

    def __init__(self, oils_json_path="oils.json", knowledge_json_path="kb/perfume_knowledge.json"):
        """Initialize fuzzy engine from existing data files"""
        self.fuzzy_enabled = FUZZY_AVAILABLE

        # Load existing data
        if os.path.exists(oils_json_path):
            with open(oils_json_path) as f:
                self.oils_data = json.load(f)
        else:
            self.oils_data = {"oils": []}

        if os.path.exists(knowledge_json_path):
            with open(knowledge_json_path) as f:
                self.knowledge = json.load(f)
        else:
            self.knowledge = {}

        # Extract useful constants from existing data
        self._extract_fuzzy_boundaries()

        print(f"Fuzzy Perfume Engine initialized (fuzzy_enabled={self.fuzzy_enabled})")

    def _extract_fuzzy_boundaries(self):
        """Extract fuzzy set boundaries from existing knowledge base"""
        try:
            # Parse existing percentage ranges from perfume_knowledge.json
            pyramid = self.knowledge.get("perfume_pyramid", {})

            top_range = pyramid.get("top_notes", {}).get("typical_percentage", "20-40%")
            mid_range = pyramid.get("middle_notes", {}).get("typical_percentage", "40-50%")
            base_range = pyramid.get("base_notes", {}).get("typical_percentage", "20-30%")

            self.note_ranges = {
                "top": self._parse_percentage_range(top_range),
                "middle": self._parse_percentage_range(mid_range),
                "base": self._parse_percentage_range(base_range)
            }

            print(f"Extracted note ranges from existing data: {self.note_ranges}")

        except Exception as e:
            print(f"Warning: Could not extract fuzzy boundaries: {e}")
            # Fallback defaults
            self.note_ranges = {
                "top": (20, 40),
                "middle": (40, 50),
                "base": (20, 30)
            }

    def _parse_percentage_range(self, range_str: str) -> Tuple[int, int]:
        """Parse '20-40%' → (20, 40)"""
        range_str = range_str.replace('%', '').replace('+', '').strip()
        parts = range_str.split('-')
        try:
            return (int(parts[0]), int(parts[1]))
        except:
            return (20, 40)  # Fallback

    def get_fuzzy_role_membership(self, oil) -> Dict[str, float]:
        """
        Extract fuzzy role membership from EXISTING role_weights.
        No new data needed - role_weights ARE fuzzy memberships!
        """
        # Handle both dict and Pydantic model
        if hasattr(oil, 'role_weights'):
            role_weights = oil.role_weights
        else:
            role_weights = oil.get("role_weights", [0, 0, 0])

        return {
            "top": float(role_weights[0]),
            "middle": float(role_weights[1]),
            "base": float(role_weights[2])
        }

    def get_oil_intensity(self, oil) -> float:
        """
        Calculate oil intensity from EXISTING features vector.
        High intensity oils (jasmine) have low max_pct and vice versa.
        """
        # Method 1: Use max_pct (inverse relationship)
        # Handle both dict and Pydantic model
        if hasattr(oil, 'max_pct'):
            max_pct = oil.max_pct
        else:
            max_pct = oil.get("max_pct", 0.5)
        if max_pct <= 0.01:
            return 0.95  # Very strong (jasmine, rose)
        elif max_pct <= 0.1:
            return 0.75  # Strong
        elif max_pct <= 0.5:
            return 0.50  # Moderate
        else:
            return 0.30  # Mild

    def fuzzy_linguistic_strength(self, oil) -> Dict[str, float]:
        """
        Convert oil to fuzzy strength linguistic variables.
        Uses EXISTING max_pct data.
        """
        intensity = self.get_oil_intensity(oil)

        # Fuzzy membership in linguistic sets
        if intensity > 0.85:
            return {"weak": 0.0, "moderate": 0.1, "strong": 0.3, "very_strong": 0.6}
        elif intensity > 0.65:
            return {"weak": 0.0, "moderate": 0.2, "strong": 0.6, "very_strong": 0.2}
        elif intensity > 0.45:
            return {"weak": 0.1, "moderate": 0.7, "strong": 0.2, "very_strong": 0.0}
        else:
            return {"weak": 0.7, "moderate": 0.3, "strong": 0.0, "very_strong": 0.0}

    def fuzzy_allergen_severity(self, allergen_list) -> Dict[str, float]:
        """
        Assess allergen severity from EXISTING allergen data.
        Returns fuzzy memberships: none, low, moderate, high

        Args:
            allergen_list: List of allergen strings OR Oil object
        """
        # Handle Oil object with allergens attribute
        if hasattr(allergen_list, 'allergens'):
            allergen_list = allergen_list.allergens

        count = len(allergen_list) if allergen_list else 0

        if count == 0:
            return {"none": 1.0, "low": 0.0, "moderate": 0.0, "high": 0.0}
        elif count == 1:
            return {"none": 0.0, "low": 0.8, "moderate": 0.2, "high": 0.0}
        elif count <= 3:
            return {"none": 0.0, "low": 0.2, "moderate": 0.6, "high": 0.2}
        else:
            return {"none": 0.0, "low": 0.0, "moderate": 0.3, "high": 0.7}

    def fuzzy_skin_sensitivity_factor(self, is_sensitive: bool, allergen_count: int) -> float:
        """
        Compute fuzzy safety cap factor.
        Replaces crisp: cap_factor = 0.7 if sensitive else 1.0
        """
        if not is_sensitive:
            sensitivity = {"normal": 1.0, "mild": 0.0, "moderate": 0.0, "severe": 0.0}
        else:
            # Fuzzy sensitivity based on allergen exposure
            if allergen_count == 0:
                sensitivity = {"normal": 0.0, "mild": 0.8, "moderate": 0.2, "severe": 0.0}
            elif allergen_count <= 2:
                sensitivity = {"normal": 0.0, "mild": 0.4, "moderate": 0.5, "severe": 0.1}
            elif allergen_count <= 4:
                sensitivity = {"normal": 0.0, "mild": 0.1, "moderate": 0.6, "severe": 0.3}
            else:
                sensitivity = {"normal": 0.0, "mild": 0.0, "moderate": 0.3, "severe": 0.7}

        # Fuzzy safety reduction factors
        safety_factors = {
            "normal": 1.0,    # No reduction
            "mild": 0.90,     # 10% reduction
            "moderate": 0.75, # 25% reduction
            "severe": 0.55    # 45% reduction
        }

        # Weighted defuzzification
        total_membership = sum(sensitivity.values())
        if total_membership == 0:
            return 1.0

        cap_factor = sum(sensitivity[level] * safety_factors[level]
                        for level in sensitivity) / total_membership

        return cap_factor

    def fuzzy_quality_assessment(self, family_match: float, role_fit: float, user_loves: bool) -> Dict[str, float]:
        """
        Fuzzy oil quality assessment.
        Replaces crisp threshold: if score >= 0.10

        Returns memberships in: poor, acceptable, good, excellent
        """
        # Fuzzy rules for quality
        if family_match > 0.7 and role_fit > 0.6:
            # Excellent match
            quality = {"poor": 0.0, "acceptable": 0.0, "good": 0.3, "excellent": 0.7}
        elif family_match > 0.5 and role_fit > 0.4:
            # Good match
            quality = {"poor": 0.0, "acceptable": 0.1, "good": 0.7, "excellent": 0.2}
        elif family_match > 0.3 or role_fit > 0.3:
            # Acceptable
            quality = {"poor": 0.0, "acceptable": 0.7, "good": 0.3, "excellent": 0.0}
        elif family_match > 0.1 or role_fit > 0.2:
            # Borderline - fuzzy helps here!
            quality = {"poor": 0.4, "acceptable": 0.6, "good": 0.0, "excellent": 0.0}
        else:
            # Poor match
            quality = {"poor": 0.9, "acceptable": 0.1, "good": 0.0, "excellent": 0.0}

        # Boost if user loves this oil
        if user_loves:
            quality["excellent"] = min(1.0, quality.get("excellent", 0) + 0.3)
            quality["good"] = min(1.0, quality.get("good", 0) + 0.2)
            # Normalize
            total = sum(quality.values())
            quality = {k: v/total for k, v in quality.items()}

        return quality

    def defuzzify_quality(self, quality: Dict[str, float]) -> float:
        """
        Convert fuzzy quality memberships to crisp score.
        Uses centroid method.
        """
        # Assign crisp values to linguistic terms
        crisp_values = {
            "poor": 0.05,
            "acceptable": 0.25,
            "good": 0.60,
            "excellent": 0.90
        }

        # Weighted average (centroid defuzzification)
        total_membership = sum(quality.values())
        if total_membership == 0:
            return 0.0

        crisp_score = sum(quality[term] * crisp_values[term]
                         for term in quality) / total_membership

        return crisp_score

    def should_include_oil(self, quality: Dict[str, float], min_quality_threshold: float = 0.3) -> bool:
        """
        Fuzzy decision: should this oil be included?
        Replaces: if score >= 0.10

        Now considers fuzzy memberships for smoother decisions.
        """
        # Include if "good" or "excellent" membership is significant
        if quality.get("excellent", 0) > 0.1:
            return True
        if quality.get("good", 0) > 0.3:
            return True
        if quality.get("acceptable", 0) > 0.5:
            return True

        # Defuzzify and compare
        crisp_score = self.defuzzify_quality(quality)
        return crisp_score >= min_quality_threshold


# Singleton instance
_fuzzy_engine = None

def get_fuzzy_engine(oils_json="oils.json", knowledge_json="kb/perfume_knowledge.json"):
    """Get or create singleton fuzzy engine instance"""
    global _fuzzy_engine
    if _fuzzy_engine is None:
        _fuzzy_engine = FuzzyPerfumeEngine(oils_json, knowledge_json)
    return _fuzzy_engine


if __name__ == "__main__":
    # Test with existing data
    engine = FuzzyPerfumeEngine()

    print("\n=== Testing Fuzzy Engine with Existing Data ===\n")

    # Test fuzzy allergen severity
    allergens = ["Limonene", "Linalool", "Citral"]
    severity = engine.fuzzy_allergen_severity(allergens)
    print(f"Allergens: {allergens}")
    print(f"Fuzzy severity: {severity}")

    # Test fuzzy skin sensitivity
    cap_factor = engine.fuzzy_skin_sensitivity_factor(is_sensitive=True, allergen_count=3)
    print(f"\nSkin sensitivity cap factor (fuzzy): {cap_factor:.2f}")
    print(f"  (Crisp would be: 0.70)")

    # Test fuzzy quality assessment
    quality = engine.fuzzy_quality_assessment(family_match=0.65, role_fit=0.55, user_loves=False)
    print(f"\nFuzzy quality assessment:")
    for level, membership in quality.items():
        if membership > 0:
            print(f"  {level}: {membership:.2f}")

    crisp = engine.defuzzify_quality(quality)
    print(f"Defuzzified score: {crisp:.2f}")
    print(f"Should include? {engine.should_include_oil(quality)}")
