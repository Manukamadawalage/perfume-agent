# rag_agent.py
"""
AI Agent for Perfume Formulation with RAG Pipeline
Pipeline: Questionnaire → RAG → GPT Plan → Boosts → Deterministic Blend → Validation → Robot Plan → Feedback
"""

import json
import os
from typing import List, Dict, Any, Optional
import numpy as np

# Define Document class early for type hints
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Conditional imports with fallbacks
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    from langchain_core.documents import Document as LangChainDocument
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
    # Use LangChain's Document class if available
    Document = LangChainDocument
    print("[OK] LangChain imports successful")
except ImportError as e:
    LANGCHAIN_AVAILABLE = False
    print(f"[WARNING] LangChain not available - RAG features disabled: {e}")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("[WARNING] OpenAI not available - using fallback mode")

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[INFO] Ollama not installed - install for local LLM support")

try:
    from huggingface_hub import InferenceClient
    HUGGINGFACE_AVAILABLE = True
except ImportError:
    HUGGINGFACE_AVAILABLE = False
    print("[INFO] HuggingFace not installed - install for HF Inference API")

class PerfumeRAGAgent:
    """
    AI Agent that uses RAG to retrieve perfume expertise and GPT to plan formulations
    """

    def __init__(self,
                 openai_api_key: Optional[str] = None,
                 llm_backend: str = "auto",  # "openai", "ollama", "huggingface", "auto"
                 ollama_model: str = "llama3.1:8b",  # Changed to smaller model to avoid GPU memory issues
                 hf_model: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"):

        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.hf_api_key = os.getenv("HUGGINGFACE_API_KEY")
        self.llm_backend = llm_backend
        self.ollama_model = ollama_model
        self.hf_model = hf_model

        self.vector_store = None
        self.knowledge_base = {}
        self.agent_enabled = False
        self.active_backend = None

        # Initialize components
        self._load_knowledge_base()
        if LANGCHAIN_AVAILABLE:
            self._initialize_vector_store()

        # Auto-select best available backend
        self._initialize_llm_backend()

    def _initialize_llm_backend(self):
        """Initialize the best available LLM backend"""

        if self.llm_backend == "auto":
            # Try in order: Ollama (free, local) → HuggingFace (free API) → OpenAI (paid)
            if OLLAMA_AVAILABLE:
                try:
                    # Test if Ollama server is running
                    ollama.list()
                    self.active_backend = "ollama"
                    self.agent_enabled = True
                    print(f"[OK] AI Agent using Ollama ({self.ollama_model})")
                    return
                except:
                    print("[INFO] Ollama not running, trying next backend...")

            if HUGGINGFACE_AVAILABLE and self.hf_api_key:
                self.active_backend = "huggingface"
                self.agent_enabled = True
                print(f"[OK] AI Agent using HuggingFace ({self.hf_model})")
                return

            if OPENAI_AVAILABLE and self.openai_api_key:
                openai.api_key = self.openai_api_key
                self.active_backend = "openai"
                self.agent_enabled = True
                print("[OK] AI Agent using OpenAI (GPT-4)")
                return

            print("[INFO] No LLM backend available - install Ollama or set API keys")

        elif self.llm_backend == "ollama":
            if OLLAMA_AVAILABLE:
                self.active_backend = "ollama"
                self.agent_enabled = True
                print(f"[OK] AI Agent using Ollama ({self.ollama_model})")
            else:
                print("[ERROR] Ollama not installed")

        elif self.llm_backend == "openai":
            if OPENAI_AVAILABLE and self.openai_api_key:
                openai.api_key = self.openai_api_key
                self.active_backend = "openai"
                self.agent_enabled = True
                print("[OK] AI Agent using OpenAI")
            else:
                print("[ERROR] OpenAI not available or no API key")

        elif self.llm_backend == "huggingface":
            if HUGGINGFACE_AVAILABLE and self.hf_api_key:
                self.active_backend = "huggingface"
                self.agent_enabled = True
                print(f"[OK] AI Agent using HuggingFace ({self.hf_model})")
            else:
                print("[ERROR] HuggingFace not available or no API key")

    def _load_knowledge_base(self):
        """Load perfume knowledge base from JSON"""
        kb_path = "kb/perfume_knowledge.json"
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as f:
                self.knowledge_base = json.load(f)
            print(f"[OK] Loaded knowledge base with {len(self.knowledge_base)} sections")
        else:
            print("[WARNING] Knowledge base not found, creating basic one")
            self.knowledge_base = {"fragrance_families": [], "formulation_principles": []}

    def _load_markdown_files(self) -> List[Document]:
        """Load all markdown files from kb/ directory"""
        documents = []
        kb_dir = "kb"

        try:
            # Load markdown files
            md_files = ["accords.md", "pyramids.md", "safety_rules.md"]

            for filename in md_files:
                filepath = os.path.join(kb_dir, filename)
                if os.path.exists(filepath):
                    with open(filepath, "r", encoding="utf-8") as f:
                        content = f.read()
                        doc_type = filename.replace(".md", "")
                        documents.append(Document(
                            page_content=content,
                            metadata={"type": doc_type, "source": filename}
                        ))
                        print(f"[OK] Loaded {filename}")

        except Exception as e:
            print(f"[WARNING] Error loading markdown files: {e}")

        return documents

    def _initialize_vector_store(self):
        """Initialize ChromaDB vector store with perfume knowledge"""
        if not LANGCHAIN_AVAILABLE:
            return

        try:
            # Create documents from knowledge base
            documents = []

            # 1. Load markdown knowledge files from kb/
            kb_docs = self._load_markdown_files()
            documents.extend(kb_docs)

            # 2. Add fragrance family knowledge from JSON
            for family in self.knowledge_base.get("fragrance_families", []):
                doc_text = f"""
                Fragrance Family: {family['name']}
                Description: {family['description']}
                Typical Notes: {', '.join(family['typical_notes'])}
                Characteristics: {family['characteristics']}
                Best For: {family['best_for']}
                """
                documents.append(Document(page_content=doc_text, metadata={"type": "family", "name": family['name']}))

            # 3. Add formulation principles
            for principle in self.knowledge_base.get("formulation_principles", []):
                doc_text = f"""
                Principle: {principle['principle']}
                Description: {principle['description']}
                Tips: {principle['tips']}
                """
                documents.append(Document(page_content=doc_text, metadata={"type": "principle"}))

            # 4. Add concentration info
            for conc_type, info in self.knowledge_base.get("concentration_types", {}).items():
                doc_text = f"""
                Concentration: {conc_type}
                Oil Concentration: {info['oil_concentration']}
                Longevity: {info['longevity']}
                Projection: {info['projection']}
                Use Case: {info['use_case']}
                """
                documents.append(Document(page_content=doc_text, metadata={"type": "concentration"}))

            # 5. Add blending tips
            for tip in self.knowledge_base.get("blending_tips", []):
                documents.append(Document(
                    page_content=f"Blending Tip: {tip}",
                    metadata={"type": "tip"}
                ))

            # 6. Add common pairings
            for pairing in self.knowledge_base.get("common_pairings", []):
                doc_text = f"""
                Combination: {pairing['combination']}
                Effect: {pairing['effect']}
                """
                documents.append(Document(page_content=doc_text, metadata={"type": "pairing"}))

            # Create embeddings and vector store
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            self.vector_store = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                collection_name="perfume_knowledge",
                persist_directory="./chroma_db"
            )
            print(f"[OK] Vector store initialized with {len(documents)} documents")

        except Exception as e:
            print(f"[ERROR] Failed to initialize vector store: {e}")
            self.vector_store = None

    def retrieve_relevant_knowledge(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        RAG STEP 1: Retrieve relevant knowledge from vector store
        """
        if not self.vector_store:
            return []

        try:
            results = self.vector_store.similarity_search(query, k=top_k)
            return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
        except Exception as e:
            print(f"[ERROR] Retrieval failed: {e}")
            return []

    def _call_llm(self, prompt: str) -> str:
        """Call the active LLM backend"""

        if self.active_backend == "ollama":
            try:
                response = ollama.chat(
                    model=self.ollama_model,
                    messages=[
                        {"role": "system", "content": "You are an expert perfumer. Respond only with valid JSON."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response['message']['content']
            except Exception as e:
                error_msg = str(e)
                if "memory" in error_msg.lower() or "status code: 500" in error_msg:
                    print(f"[WARNING] Ollama GPU memory issue detected. Disabling AI agent features.")
                    print(f"[INFO] To fix: Run 'ollama serve' with CPU mode or use a smaller model")
                    self.agent_enabled = False
                    self.active_backend = None
                raise

        elif self.active_backend == "huggingface":
            client = InferenceClient(token=self.hf_api_key)
            response = client.text_generation(
                prompt,
                model=self.hf_model,
                max_new_tokens=1500,
                temperature=0.7
            )
            return response

        elif self.active_backend == "openai":
            response = openai.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert perfumer. Respond only with valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500
            )
            return response.choices[0].message.content.strip()

        else:
            raise Exception("No active LLM backend")

    def generate_formulation_plan(self, questionnaire: Dict, retrieved_knowledge: List[Dict]) -> Dict:
        """
        RAG STEP 2: Use LLM to generate intelligent formulation plan
        """
        if not self.agent_enabled:
            return self._fallback_plan(questionnaire)

        try:
            # Build context from retrieved knowledge
            context = "\n\n".join([doc["content"] for doc in retrieved_knowledge])

            # Create prompt for GPT
            prompt = f"""You are an expert perfumer with 20 years of experience in fragrance formulation.

You have access to comprehensive perfume knowledge including:
- Fragrance Accords (Floral, Woody, Gourmand, Spicy, Citrus)
- Seasonal Pyramid Ratios (Summer, Winter, Daily, Special Occasions)
- Safety Rules (Sensitive skin guidelines, IFRA caps, photo-sensitivity)
- Formulation Principles
- Common Note Pairings

RETRIEVED EXPERT KNOWLEDGE (from knowledge base):
{context}

USER PREFERENCES:
- Overall Style: {questionnaire.get('overall', 'balanced')}
- Gender Vibe: {questionnaire.get('gender_vibe', 'unisex')}
- Longevity Desired: {questionnaire.get('longevity_0_5', 3)}/5
- Projection Desired: {questionnaire.get('projection_0_5', 3)}/5
- Occasions: {', '.join(questionnaire.get('occasions', ['daily']))}
- Sensitive Skin: {'Yes' if questionnaire.get('sensitive_skin') else 'No'}
- Loves: {', '.join(questionnaire.get('loves', []))}
- Dislikes: {', '.join(questionnaire.get('dislikes', []))}
- Scent Description: {questionnaire.get('scent_description', 'Not provided')}

Family Ratings (0-5):
{json.dumps(questionnaire.get('family_ratings', {}), indent=2)}

TASK:
Create a perfume formulation plan that:
1. Honors the user's preferences and family ratings
2. Creates a well-balanced fragrance with proper top/middle/base note distribution
3. Considers longevity and projection requirements
4. Respects skin sensitivity if applicable
5. Incorporates loved ingredients and avoids disliked ones

Provide your response in the following JSON format:
{{
  "concept": "Brief description of the fragrance concept (1-2 sentences)",
  "note_distribution": {{
    "top_percent": 30,
    "middle_percent": 45,
    "base_percent": 25
  }},
  "family_boosts": {{
    "citrus": 0.1,
    "floral": 0.05,
    ...
  }},
  "recommended_notes": [
    {{"note": "bergamot", "position": "top", "importance": "high"}},
    ...
  ],
  "reasoning": "Explanation of formulation choices",
  "tips": ["Specific blending tip 1", "Tip 2"]
}}

Respond ONLY with valid JSON, no other text.
"""

            # Call LLM (auto-selects backend)
            plan_text = self._call_llm(prompt)

            # Extract JSON from response (handle markdown code blocks)
            if "```json" in plan_text:
                plan_text = plan_text.split("```json")[1].split("```")[0].strip()
            elif "```" in plan_text:
                plan_text = plan_text.split("```")[1].split("```")[0].strip()

            plan = json.loads(plan_text)
            plan["ai_generated"] = True
            plan["model_used"] = f"{self.active_backend}: {self.ollama_model if self.active_backend == 'ollama' else 'GPT-4' if self.active_backend == 'openai' else self.hf_model}"

            print(f"[OK] {self.active_backend.upper()} formulation plan generated")
            return plan

        except Exception as e:
            print(f"[ERROR] LLM generation failed: {e}")
            return self._fallback_plan(questionnaire)

    def _fallback_plan(self, questionnaire: Dict) -> Dict:
        """Fallback plan when GPT is unavailable"""
        return {
            "concept": "Balanced fragrance based on user preferences",
            "note_distribution": {
                "top_percent": 30,
                "middle_percent": 45,
                "base_percent": 25
            },
            "family_boosts": {},
            "recommended_notes": [],
            "reasoning": "Using deterministic algorithm (GPT unavailable)",
            "tips": [],
            "ai_generated": False
        }

    def apply_ai_boosts(self, user_vec: np.ndarray, plan: Dict) -> np.ndarray:
        """
        Apply AI-recommended boosts to user preference vector
        """
        boosted_vec = user_vec.copy()

        family_boosts = plan.get("family_boosts", {})
        families = ["citrus", "floral", "woody", "green", "gourmand", "spicy", "powdery", "resinous"]

        for i, family in enumerate(families):
            if family in family_boosts:
                boost = family_boosts[family]
                boosted_vec[i] += boost

        # Renormalize
        total = boosted_vec.sum()
        if total > 0:
            boosted_vec = boosted_vec / total

        return boosted_vec

    def adjust_note_targets(self, base_targets: Dict, plan: Dict) -> Dict:
        """
        Adjust note distribution based on AI plan
        """
        if not plan.get("ai_generated"):
            return base_targets

        distribution = plan.get("note_distribution", {})

        adjusted = {
            "top": distribution.get("top_percent", base_targets.get("top", 30)),
            "middle": distribution.get("middle_percent", base_targets.get("middle", 45)),
            "base": distribution.get("base_percent", base_targets.get("base", 25))
        }

        return adjusted

    def full_pipeline(self, questionnaire: Dict, user_vec: np.ndarray, note_targets: Dict) -> Dict:
        """
        COMPLETE AI AGENT PIPELINE
        Questionnaire → RAG → GPT Plan → Boosts → Enhanced targets
        """
        result = {
            "agent_active": self.agent_enabled,
            "original_targets": note_targets.copy(),
            "original_vec": user_vec.copy(),
            "plan": {},
            "boosted_vec": user_vec.copy(),
            "adjusted_targets": note_targets.copy(),
            "retrieved_docs": []
        }

        if not self.agent_enabled:
            return result

        try:
            # Step 1: Build RAG query from questionnaire
            query = f"""
            Looking for fragrance formulation guidance:
            Overall style: {questionnaire.get('overall')}
            Gender: {questionnaire.get('gender_vibe')}
            Occasions: {', '.join(questionnaire.get('occasions', []))}
            Description: {questionnaire.get('scent_description', '')}
            """

            # Step 2: RAG Retrieval
            retrieved = self.retrieve_relevant_knowledge(query, top_k=5)
            result["retrieved_docs"] = retrieved

            # Step 3: GPT Planning
            plan = self.generate_formulation_plan(questionnaire, retrieved)
            result["plan"] = plan

            # Step 4: Apply AI Boosts
            boosted_vec = self.apply_ai_boosts(user_vec, plan)
            result["boosted_vec"] = boosted_vec

            # Step 5: Adjust note targets
            adjusted_targets = self.adjust_note_targets(note_targets, plan)
            result["adjusted_targets"] = adjusted_targets

            print("[OK] Full AI agent pipeline completed")

        except Exception as e:
            print(f"[ERROR] AI agent pipeline failed: {e}")

        return result

    def process_feedback(self, feedback: Dict):
        """
        Process user feedback to improve future recommendations
        """
        # Store feedback for future fine-tuning
        feedback_path = "ai_models/agent_feedback.jsonl"
        os.makedirs("ai_models", exist_ok=True)

        with open(feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback) + "\n")

        print(f"[OK] Feedback stored for future model improvements")

# Global agent instance
_agent_instance = None

def get_agent() -> PerfumeRAGAgent:
    """Get or create the global agent instance"""
    global _agent_instance
    if _agent_instance is None:
        _agent_instance = PerfumeRAGAgent()
    return _agent_instance
