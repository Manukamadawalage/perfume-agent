# nlp_dislike_detector.py
"""
Extract dislikes and allergies from natural language descriptions
Works without any AI/NLP libraries - uses pattern matching
"""

import re

# Common dislike/allergy phrases
DISLIKE_PATTERNS = [
    r"don't like (\w+)",
    r"dislike (\w+)",
    r"hate (\w+)",
    r"avoid (\w+)",
    r"not a fan of (\w+)",
    r"can't stand (\w+)",
    r"don't want (\w+)",
    r"no (\w+)",
    r"without (\w+)",
    r"allergic to (\w+)",
    r"allergy to (\w+)",
    r"sensitive to (\w+)",
    r"makes me (\w+)",  # e.g., "patchouli makes me sick"
]

# Oil name variations
OIL_ALIASES = {
    "patchouli": ["patchouli", "patch"],
    "vanilla": ["vanilla"],
    "rose": ["rose"],
    "lavender": ["lavender"],
    "bergamot": ["bergamot"],
    "lemon": ["lemon"],
    "jasmine": ["jasmine"],
    "ylang": ["ylang-ylang", "ylang"],
    "geranium": ["geranium"],
    "clove": ["clove"],
    "cedar": ["cedarwood", "cedar"],
    "sandalwood": ["sandalwood", "sandal"],
    "vetiver": ["vetiver"],
    "frankincense": ["frankincense", "frankin"],
    "pepper": ["black pepper", "pepper"],
    "musk": ["musk"],
    "amber": ["amber"],
    "oud": ["oud", "agarwood"],
    "oakmoss": ["oakmoss", "oak moss"],
}


def extract_dislikes_from_text(description: str, catalog_oils: list = None) -> list:
    """
    Extract oils to avoid from natural language description

    Args:
        description: User's text description
        catalog_oils: Optional list of Oil objects from catalog (for exact matching)

    Returns:
        List of oil names to avoid (matches catalog names)
    """
    if not description or not isinstance(description, str):
        return []

    description = description.lower()
    dislikes = []

    # Method 1: Pattern matching for explicit dislikes
    for pattern in DISLIKE_PATTERNS:
        matches = re.findall(pattern, description, re.IGNORECASE)
        for match in matches:
            dislikes.append(match.lower())

    # Method 2: Check for oil names with STRONG negative context
    # Use "but" and "except" to split positive from negative
    # Example: "I love rose but allergic to patchouli" → only patchouli is disliked

    negative_words = ["don't", "not", "no", "avoid", "without", "hate", "dislike",
                      "allergic", "allergy", "sensitive", "headache", "nausea", "sick",
                      "makes me", "gives me"]

    # Split by strong separators first
    parts = re.split(r'\bbut\b|\bexcept\b|\bhowever\b', description)

    for part in parts:
        part = part.lower().strip()

        # Check if this part has negative context
        has_negative = any(word in part for word in negative_words)

        if has_negative:
            # Look for oil names in this negative part only
            for oil_key, aliases in OIL_ALIASES.items():
                for alias in aliases:
                    if alias in part:
                        dislikes.append(oil_key)
                        break

    # Method 3: Check against actual catalog oils (if provided)
    if catalog_oils:
        for oil in catalog_oils:
            oil_name_lower = oil.name.lower()

            # Check for negative mentions
            for neg_word in negative_words:
                # Pattern: "no rose", "avoid lavender", etc.
                pattern = rf"\b{neg_word}\b.*\b{re.escape(oil_name_lower.split()[0])}\b"
                if re.search(pattern, description, re.IGNORECASE):
                    dislikes.append(oil_name_lower)

    # Remove duplicates and normalize
    dislikes = list(set(dislikes))

    # Map aliases back to full names (if catalog provided)
    if catalog_oils:
        normalized = []
        for dislike in dislikes:
            # Try to match to actual catalog oil
            for oil in catalog_oils:
                oil_name_lower = oil.name.lower()
                # Check if dislike matches any part of oil name
                if dislike in oil_name_lower or oil_name_lower.startswith(dislike):
                    normalized.append(oil.name)
                    break
            else:
                # Keep original if no match
                normalized.append(dislike)
        dislikes = normalized

    return dislikes


def extract_loves_from_text(description: str, catalog_oils: list = None) -> list:
    """
    Extract oils the user loves from natural language description

    Args:
        description: User's text description
        catalog_oils: Optional list of Oil objects from catalog

    Returns:
        List of oil names the user loves
    """
    if not description or not isinstance(description, str):
        return []

    description = description.lower()
    loves = []

    # Positive patterns
    LOVE_PATTERNS = [
        r"love (\w+)",
        r"adore (\w+)",
        r"favorite (\w+)",
        r"favourite (\w+)",
        r"prefer (\w+)",
        r"want (\w+)",
        r"like (\w+)",
        r"enjoy (\w+)",
    ]

    for pattern in LOVE_PATTERNS:
        matches = re.findall(pattern, description, re.IGNORECASE)
        for match in matches:
            loves.append(match.lower())

    # Check for oil names with positive context (but NOT in negative parts)
    positive_words = ["love", "like", "adore", "favorite", "prefer", "want", "enjoy", "beautiful", "amazing"]
    negative_words = ["don't", "not", "no", "avoid", "without", "hate", "dislike",
                      "allergic", "allergy", "sensitive"]

    # Split by "but/except" to separate positive from negative
    parts = re.split(r'\bbut\b|\bexcept\b|\bhowever\b', description)

    for part in parts:
        part = part.lower().strip()

        has_positive = any(word in part for word in positive_words)
        has_negative = any(word in part for word in negative_words)

        # Only extract loves if positive context and NO negative words
        if has_positive and not has_negative:
            for oil_key, aliases in OIL_ALIASES.items():
                for alias in aliases:
                    if alias in part:
                        loves.append(oil_key)
                        break

    # Normalize if catalog provided
    if catalog_oils:
        normalized = []
        for love in loves:
            for oil in catalog_oils:
                oil_name_lower = oil.name.lower()
                if love in oil_name_lower or oil_name_lower.startswith(love):
                    normalized.append(oil.name)
                    break
            else:
                normalized.append(love)
        loves = normalized

    return list(set(loves))


# Test examples
if __name__ == "__main__":
    test_descriptions = [
        "I love rose and jasmine but I'm allergic to patchouli",
        "Please avoid vanilla, I don't like sweet scents",
        "I hate musk and can't stand sandalwood",
        "Something fresh without lavender, makes me sick",
        "I want bergamot and lemon, no floral notes",
        "Allergic to ylang-ylang and oakmoss",
    ]

    print("="*60)
    print("TESTING DISLIKE DETECTION")
    print("="*60)

    for desc in test_descriptions:
        print(f"\nDescription: '{desc}'")
        dislikes = extract_dislikes_from_text(desc)
        loves = extract_loves_from_text(desc)
        print(f"  Dislikes detected: {dislikes}")
        print(f"  Loves detected: {loves}")
