# models.py
from pydantic import BaseModel, Field
from typing import List, Dict, Literal

FAMILIES = ["citrus","floral","woody","green","gourmand","spicy","powdery","resinous"]
NOTES = ["top","middle","base"]

class Oil(BaseModel):
    name: str
    density_g_ml: float
    role_weights: List[float]    # [top, middle, base]
    features: List[float]        # len=8 aligned with FAMILIES
    max_pct: float               # % of fragrance concentrate (not full bottle)
    allergens: List[str] = []
    is_carrier: bool = False     # True for alcohol/carrier, False for essential oils

class Catalog(BaseModel):
    oils: List[Oil]

class Questionnaire(BaseModel):
    overall: Literal["fresh","floral","warm_spicy","sweet","woody","resinous"]
    family_ratings: Dict[str, int]   # each 0..5 for each in FAMILIES
    gender_vibe: Literal["feminine","masculine","unisex"]
    longevity_0_5: int
    projection_0_5: int
    occasions: List[Literal["daily","special","summer","winter"]]
    sensitive_skin: bool
    dislikes: List[str]
    loves: List[str]
    bottle_ml: int
    concentration: Literal["EdC","EDT","EDP","Parfum"]
    scent_description: str = ""  # Optional: user's free-text description for NLP analysis

class FeedbackRequest(BaseModel):
    questionnaire: Dict
    recipe: Dict
    rating: float = Field(..., ge=0, le=5)  # 0-5 star rating
    comments: str = ""
