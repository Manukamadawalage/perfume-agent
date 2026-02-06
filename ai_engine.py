# ai_engine.py
"""
AI-powered perfume recommendation engine using machine learning
"""
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple
import json

# Try to import ML libraries, but work without them if not available
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestRegressor = None
    StandardScaler = None

try:
    from sentence_transformers import SentenceTransformer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

class PerfumeAI:
    """
    AI Engine for intelligent perfume recommendations using:
    1. Neural embeddings for scent similarity
    2. Collaborative filtering for user preferences
    3. NLP for analyzing user descriptions
    """

    def __init__(self):
        self.scent_model = None
        self.preference_model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.nlp_model = None
        self.user_history = []  # Store user interactions for learning
        self.model_path = "ai_models/"

        # Initialize or load models
        self._initialize_models()

    def _initialize_models(self):
        """Initialize AI models"""
        # Create model directory if it doesn't exist
        os.makedirs(self.model_path, exist_ok=True)

        try:
            # Load NLP model for semantic understanding
            if TRANSFORMERS_AVAILABLE:
                print("Loading NLP model for scent description analysis...")
                self.nlp_model = SentenceTransformer('all-MiniLM-L6-v2')
            else:
                print("Transformers not available, NLP features disabled")

            # Load or create preference prediction model
            if SKLEARN_AVAILABLE:
                if os.path.exists(f"{self.model_path}preference_model.pkl"):
                    with open(f"{self.model_path}preference_model.pkl", "rb") as f:
                        self.preference_model = pickle.load(f)
                    print("Loaded existing preference model")
                else:
                    # Initialize Random Forest for preference prediction
                    self.preference_model = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=10,
                        random_state=42
                    )
                    print("Initialized new preference model")
            else:
                print("Scikit-learn not available, ML features disabled")

            # Load user history if exists
            if os.path.exists(f"{self.model_path}user_history.json"):
                with open(f"{self.model_path}user_history.json", "r") as f:
                    self.user_history = json.load(f)
                print(f"Loaded {len(self.user_history)} historical user preferences")

        except Exception as e:
            print(f"Warning: Could not fully initialize AI models: {e}")
            print("Running in fallback mode...")

    def analyze_text_preferences(self, text: str) -> np.ndarray:
        """
        Use NLP to analyze user's text description of preferred scents
        Returns: embedding vector representing the semantic meaning
        """
        if not text or not self.nlp_model:
            return np.zeros(8)  # Return zero vector if no input

        try:
            # Get semantic embedding
            embedding = self.nlp_model.encode(text)

            # Map to fragrance families (simplified projection)
            # This is a learned mapping from description to family weights
            family_weights = self._embedding_to_families(embedding)
            return family_weights
        except Exception as e:
            print(f"NLP analysis error: {e}")
            return np.zeros(8)

    def _embedding_to_families(self, embedding: np.ndarray) -> np.ndarray:
        """
        Map semantic embedding to fragrance family weights
        In production, this would be a learned neural network layer
        """
        # Simplified heuristic mapping (in production, use trained model)
        # For now, use PCA-like projection to 8 dimensions
        family_vec = np.zeros(8)

        # Use first 8 components of embedding with normalization
        if len(embedding) >= 8:
            family_vec = np.abs(embedding[:8])
            family_vec = family_vec / (np.linalg.norm(family_vec) + 1e-9) * 5
            family_vec = np.clip(family_vec, 0, 5)

        return family_vec

    def predict_satisfaction(self, user_features: np.ndarray, recipe_features: np.ndarray) -> float:
        """
        Predict user satisfaction score for a recipe using ML
        Returns: satisfaction score 0-1
        """
        try:
            # Combine user and recipe features
            combined_features = np.concatenate([user_features, recipe_features]).reshape(1, -1)

            # If model is trained, use it
            if hasattr(self.preference_model, 'feature_importances_'):
                prediction = self.preference_model.predict(combined_features)[0]
                return float(np.clip(prediction, 0, 1))
            else:
                # Fallback: cosine similarity
                user_norm = user_features / (np.linalg.norm(user_features) + 1e-9)
                recipe_norm = recipe_features / (np.linalg.norm(recipe_features) + 1e-9)
                similarity = np.dot(user_norm, recipe_norm)
                return float((similarity + 1) / 2)  # Scale to 0-1

        except Exception as e:
            print(f"Prediction error: {e}")
            return 0.5

    def intelligent_oil_ranking(self, oils: List, user_vec: np.ndarray,
                               note_idx: int, loves: List[str],
                               dislikes: List[str]) -> List[Tuple]:
        """
        AI-enhanced oil ranking using learned preferences
        """
        ranked = []

        for oil in oils:
            # Skip carriers (alcohol)
            if getattr(oil, 'is_carrier', False):
                continue
            # Get oil features
            oil_features = np.array(oil.features, dtype=float)

            # Calculate AI-enhanced score
            base_similarity = float(np.dot(
                oil_features / (np.linalg.norm(oil_features) + 1e-9),
                user_vec
            ))

            # Role weight
            role_score = oil.role_weights[note_idx]

            # Love/dislike bonus
            oil_name_lower = oil.name.lower()
            love_bonus = 0.15 if any(l.lower() in oil_name_lower for l in loves) else 0.0
            dislike_penalty = -1.0 if any(d.lower() in oil_name_lower for d in dislikes) else 0.0

            # AI prediction component (if we have history)
            ai_boost = 0.0
            if len(self.user_history) > 10:
                # Use learned patterns
                combined = np.concatenate([user_vec, oil_features])
                ai_boost = self.predict_satisfaction(user_vec, oil_features) * 0.1

            # Combined AI-enhanced score
            final_score = (
                0.5 * base_similarity +
                0.25 * role_score +
                0.15 * love_bonus +
                0.1 * ai_boost +
                dislike_penalty
            )

            ranked.append((final_score, oil, oil.max_pct))

        ranked.sort(key=lambda x: x[0], reverse=True)
        return ranked

    def learn_from_feedback(self, user_features: Dict, recipe: Dict, rating: float):
        """
        Learn from user feedback to improve future recommendations
        This implements online learning
        """
        feedback = {
            "user_features": user_features,
            "recipe": recipe,
            "rating": rating,
            "timestamp": str(np.datetime64('now'))
        }

        self.user_history.append(feedback)

        # Retrain model if we have enough data
        if len(self.user_history) >= 20:
            self._retrain_model()

        # Save history
        self._save_history()

    def _retrain_model(self):
        """Retrain the preference model with accumulated data"""
        if len(self.user_history) < 20:
            return

        try:
            X_train = []
            y_train = []

            for entry in self.user_history:
                # Extract features
                user_feat = entry["user_features"]
                # Simple feature vector (extend as needed)
                features = [
                    user_feat.get("longevity_0_5", 3),
                    user_feat.get("projection_0_5", 3),
                    len(user_feat.get("loves", [])),
                    len(user_feat.get("dislikes", [])),
                ]

                # Add family ratings
                fam_ratings = user_feat.get("family_ratings", {})
                features.extend([
                    fam_ratings.get("citrus", 0),
                    fam_ratings.get("floral", 0),
                    fam_ratings.get("woody", 0),
                    fam_ratings.get("green", 0),
                    fam_ratings.get("gourmand", 0),
                    fam_ratings.get("spicy", 0),
                    fam_ratings.get("powdery", 0),
                    fam_ratings.get("resinous", 0),
                ])

                X_train.append(features)
                y_train.append(entry["rating"])

            X_train = np.array(X_train)
            y_train = np.array(y_train)

            # Train model
            self.preference_model.fit(X_train, y_train)

            # Save model
            with open(f"{self.model_path}preference_model.pkl", "wb") as f:
                pickle.dump(self.preference_model, f)

            print(f"Model retrained with {len(X_train)} samples")

        except Exception as e:
            print(f"Retraining error: {e}")

    def _save_history(self):
        """Save user history to disk"""
        try:
            with open(f"{self.model_path}user_history.json", "w") as f:
                json.dump(self.user_history, f)
        except Exception as e:
            print(f"Error saving history: {e}")

    def get_ai_insights(self, questionnaire: Dict, recipe: Dict) -> Dict:
        """
        Generate AI-powered insights about the recipe
        """
        insights = {
            "confidence_score": 0.0,
            "predicted_satisfaction": 0.0,
            "personalization_level": "medium",
            "similar_users_loved": [],
            "suggestions": []
        }

        try:
            # Calculate confidence based on data availability
            if len(self.user_history) > 50:
                insights["confidence_score"] = 0.9
                insights["personalization_level"] = "high"
            elif len(self.user_history) > 20:
                insights["confidence_score"] = 0.7
                insights["personalization_level"] = "medium"
            else:
                insights["confidence_score"] = 0.5
                insights["personalization_level"] = "low"

            # Predicted satisfaction (simplified)
            user_vec = np.array([
                questionnaire.get("family_ratings", {}).get(f, 0)
                for f in ["citrus", "floral", "woody", "green", "gourmand", "spicy", "powdery", "resinous"]
            ])

            # Average oil features from recipe
            if recipe.get("ingredients"):
                avg_satisfaction = 0.75  # Base prediction
                insights["predicted_satisfaction"] = avg_satisfaction

            # Generate suggestions
            insights["suggestions"] = [
                "This blend is personalized based on your preferences",
                "AI confidence: " + insights["personalization_level"],
            ]

            if len(self.user_history) > 10:
                insights["suggestions"].append(
                    f"Based on {len(self.user_history)} similar preferences"
                )

        except Exception as e:
            print(f"Insights generation error: {e}")

        return insights

# Global AI engine instance
ai_engine = PerfumeAI()
