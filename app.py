# app.py
import json, os
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

from models import Catalog, Questionnaire, FeedbackRequest

# ============== AI-POWERED FUZZY LOGIC SYSTEM ==============
# This system uses ONLY fuzzy logic + AI/ML (no crisp logic)
# ===========================================================

# Import fuzzy logic modules (REQUIRED)
from fuzzy_targets import compute_targets
from fuzzy_select import rank_oils_for_note
from fuzzy_allocate import allocate

print("[OK] Using AI-POWERED FUZZY LOGIC formulation system")
print("     - Fuzzy inference for note pyramid calculation")
print("     - Fuzzy quality assessment for oil selection")
print("     - Fuzzy allocation strategy with dynamic complexity")

# Shared utility modules (work for any formulation approach)
from logic_normalize import normalize_and_enforce
from logic_validate import validate
from logic_dispense import to_dispense_plan
from logic_consolidate import consolidate_ingredients
from nlp_dislike_detector import extract_dislikes_from_text, extract_loves_from_text

# Fuzzy logic is ALWAYS enabled
USE_FUZZY_LOGIC = True
# Enforce fuzzy-only mode
if not USE_FUZZY_LOGIC:
    raise RuntimeError("Fuzzy-only build: disable of fuzzy logic is not permitted")

# Import Serial Dispenser
try:
    from serial_dispenser import ArduinoDispenser
    SERIAL_ENABLED = True
    print("[OK] Serial dispenser module loaded")
except Exception as e:
    print(f"[WARNING] Serial dispenser not available: {e}")
    SERIAL_ENABLED = False
    ArduinoDispenser = None

# Import AI engine
try:
    from ai_engine import ai_engine
    AI_ENABLED = True
    print("[OK] AI Engine loaded successfully!")
except Exception as e:
    print(f"[WARNING] AI Engine not available: {e}")
    AI_ENABLED = False
    ai_engine = None

# Import RAG Agent
try:
    from rag_agent import get_agent
    rag_agent = get_agent()
    AGENT_ENABLED = rag_agent.agent_enabled
    print(f"[OK] RAG Agent loaded - Active: {AGENT_ENABLED}")
except Exception as e:
    print(f"[WARNING] RAG Agent not available: {e}")
    AGENT_ENABLED = False
    rag_agent = None

app = FastAPI(title="Perfume Agent API with AI")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load catalog & valve map
with open("oils.json", "r", encoding="utf-8") as f:
    CATALOG = Catalog(**json.load(f))
VALVE_MAP = (
    json.load(open("valve_map.json", "r", encoding="utf-8"))
    if os.path.exists("valve_map.json")
    else {}
)

@app.get("/health")
def health():
    try:
        ai_confidence = "low"
        if AI_ENABLED and ai_engine:
            try:
                history_len = len(ai_engine.user_history)
                ai_confidence = "high" if history_len > 50 else "medium" if history_len > 20 else "low"
            except:
                pass

        return {
            "status": "ok",
            "oils_count": len(CATALOG.oils),
            "ai_enabled": AI_ENABLED,
            "agent_enabled": AGENT_ENABLED,
            "fuzzy_logic_enabled": USE_FUZZY_LOGIC,
            "ai_confidence": ai_confidence,
            "formulation_mode": "Fuzzy + AI/ML Hybrid" if USE_FUZZY_LOGIC and AI_ENABLED else
                               "Fuzzy Logic" if USE_FUZZY_LOGIC else
                               "Crisp + AI/ML" if AI_ENABLED else "Crisp Logic"
        }
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/formulate")
def formulate(q: Questionnaire):
    import numpy as np
    from models import FAMILIES

    # DEBUG: Log what we received
    print("\n" + "="*60)
    print("FORMULATE REQUEST RECEIVED")
    print("="*60)
    print(f"Dislikes (manual): {q.dislikes}")
    print(f"Loves (manual): {q.loves}")
    print(f"Description: {q.scent_description[:100] if q.scent_description else 'None'}...")
    print("="*60 + "\n")

    # 0a) AUTOMATIC: Extract dislikes/loves from description text
    if q.scent_description:
        print("Analyzing description for preferences...")
        auto_dislikes = extract_dislikes_from_text(q.scent_description, CATALOG.oils)
        auto_loves = extract_loves_from_text(q.scent_description, CATALOG.oils)

        if auto_dislikes:
            print(f"  Auto-detected DISLIKES: {auto_dislikes}")
            # Merge with manual dislikes (avoid duplicates)
            q.dislikes = list(set(q.dislikes + auto_dislikes))

        if auto_loves:
            print(f"  Auto-detected LOVES: {auto_loves}")
            # Merge with manual loves
            q.loves = list(set(q.loves + auto_loves))

        print(f"  Final dislikes: {q.dislikes}")
        print(f"  Final loves: {q.loves}\n")

    # 0) Optional: AI-enhanced preference analysis from text description
    if AI_ENABLED and ai_engine and q.scent_description:
        try:
            nlp_preferences = ai_engine.analyze_text_preferences(q.scent_description)
            # Boost family ratings based on NLP analysis
            for i, family in enumerate(FAMILIES):
                if nlp_preferences[i] > 0:
                    current = q.family_ratings.get(family, 0)
                    # Blend NLP insight with user ratings
                    q.family_ratings[family] = int(min(5, current + nlp_preferences[i] * 0.3))
        except Exception as e:
            print(f"[WARNING] AI NLP analysis failed: {e}")
            # Continue without AI enhancement

    # 1) compute targets from questionnaire
    user_vec, note_targets, cap_factor = compute_targets(q)

    # 1.5) AI AGENT PIPELINE: Questionnaire → RAG → GPT Plan → Boosts
    agent_result = None
    if AGENT_ENABLED and rag_agent:
        try:
            print("[INFO] Running AI Agent pipeline...")
            agent_result = rag_agent.full_pipeline(
                questionnaire=q.dict(),
                user_vec=user_vec,
                note_targets=note_targets
            )

            # Use AI-boosted preferences and targets
            if agent_result.get("plan", {}).get("ai_generated"):
                user_vec = agent_result["boosted_vec"]
                note_targets = agent_result["adjusted_targets"]
                print("[OK] Using AI-enhanced preferences and targets")
        except Exception as e:
            print(f"[WARNING] AI Agent pipeline failed: {e}")
            # Continue with original targets

    # 2) rank oils for each note (AI-enhanced if available)
    if AI_ENABLED and ai_engine:
        try:
            ranked_top  = ai_engine.intelligent_oil_ranking(CATALOG.oils, user_vec, 0, q.loves, q.dislikes)
            ranked_mid  = ai_engine.intelligent_oil_ranking(CATALOG.oils, user_vec, 1, q.loves, q.dislikes)
            ranked_base = ai_engine.intelligent_oil_ranking(CATALOG.oils, user_vec, 2, q.loves, q.dislikes)
            print("[INFO] Using AI-enhanced oil ranking")
        except Exception as e:
            print(f"[WARNING] AI ranking failed: {e}, falling back to standard ranking")
            ranked_top  = rank_oils_for_note(CATALOG, user_vec, 0, q.loves, q.dislikes, cap_factor, use_fuzzy=USE_FUZZY_LOGIC)
            ranked_mid  = rank_oils_for_note(CATALOG, user_vec, 1, q.loves, q.dislikes, cap_factor, use_fuzzy=USE_FUZZY_LOGIC)
            ranked_base = rank_oils_for_note(CATALOG, user_vec, 2, q.loves, q.dislikes, cap_factor, use_fuzzy=USE_FUZZY_LOGIC)
    else:
        ranked_top  = rank_oils_for_note(CATALOG, user_vec, 0, q.loves, q.dislikes, cap_factor, use_fuzzy=USE_FUZZY_LOGIC)
        ranked_mid  = rank_oils_for_note(CATALOG, user_vec, 1, q.loves, q.dislikes, cap_factor, use_fuzzy=USE_FUZZY_LOGIC)
        ranked_base = rank_oils_for_note(CATALOG, user_vec, 2, q.loves, q.dislikes, cap_factor, use_fuzzy=USE_FUZZY_LOGIC)

    # 3) allocate (cap-aware across notes)
    if USE_FUZZY_LOGIC:
        # Fuzzy allocation with additional parameters
        recipe_items = allocate(
            note_targets,
            [ranked_top, ranked_mid, ranked_base],
            CATALOG,
            cap_factor,
            use_fuzzy=True,
            occasion=q.occasions[0] if q.occasions else "daily",
            concentration=q.concentration
        )
    else:
        # Crisp allocation
        recipe_items = allocate(
            note_targets,
            [ranked_top, ranked_mid, ranked_base],
            CATALOG,
            cap_factor,
        )

    # 4) normalize to 100.00% WITHOUT breaking caps
    recipe_items = normalize_and_enforce(
        recipe_items, CATALOG, q.sensitive_skin, decimals=2
    )

    # 4.5) CONSOLIDATE duplicate ingredients (same oil in multiple notes)
    # This merges duplicates for more efficient dispensing
    recipe_items_consolidated = consolidate_ingredients(recipe_items)

    # DEBUG: Show what oils were selected
    print("\nSELECTED OILS:")
    for item in recipe_items_consolidated:
        print(f"  - {item['oil_name']:30s} {item['percent']:5.2f}%")
    print()

    # 5) validate consolidated recipe; return 422 instead of 500 on failure while iterating
    try:
        validate(recipe_items_consolidated, CATALOG, note_targets, q.sensitive_skin)
    except AssertionError as e:
        raise HTTPException(status_code=422, detail=str(e))

    # 6) build recipe JSON from the *consolidated* list
    recipe = {
        "recipe_name": "Auto Blend",
        "note_targets": note_targets,
        "ingredients": [
            {
                "oil_name": r["oil_name"],
                "percent": r["percent"],
                "note": r["note"],  # Primary note
                "reason": r.get("reason", "score-based selection"),
            }
            for r in recipe_items_consolidated
        ],
        "dilution": {
            "type": q.concentration,
            "alcohol_pct": 100
            - {"EdC": 5, "EDT": 12, "EDP": 18, "Parfum": 25}[q.concentration],
            "bottle_ml": q.bottle_ml,
        },
    }

    # 7) convert to a dispenser plan (ml / grams) using CONSOLIDATED list
    plan = to_dispense_plan(
        recipe_items_consolidated,  # ⬅️ use consolidated list (no duplicates!)
        q.bottle_ml,
        q.concentration,
        CATALOG,
        VALVE_MAP,
    )

    # 8) AI insights (if available)
    ai_insights = {}
    if AI_ENABLED and ai_engine:
        try:
            ai_insights = ai_engine.get_ai_insights(q.dict(), recipe)
        except Exception as e:
            print(f"[WARNING] AI insights generation failed: {e}")
            # Continue without insights

    # 9) Agent insights (if available)
    agent_insights = {}
    if agent_result and agent_result.get("plan"):
        try:
            agent_plan = agent_result["plan"]
            agent_insights = {
                "concept": agent_plan.get("concept", ""),
                "reasoning": agent_plan.get("reasoning", ""),
                "tips": agent_plan.get("tips", []),
                "ai_generated": agent_plan.get("ai_generated", False),
                "model_used": agent_plan.get("model_used", ""),
                "retrieved_knowledge_count": len(agent_result.get("retrieved_docs", []))
            }
        except Exception as e:
            print(f"[WARNING] Agent insights generation failed: {e}")
            # Continue without agent insights

    return {
        "recipe": recipe,
        "dispense_plan": plan,
        "ai_insights": ai_insights,
        "ai_powered": AI_ENABLED,
        "agent_powered": AGENT_ENABLED,
        "fuzzy_powered": USE_FUZZY_LOGIC,
        "formulation_mode": "Fuzzy + AI/ML Hybrid" if USE_FUZZY_LOGIC and AI_ENABLED else
                           "Fuzzy Logic" if USE_FUZZY_LOGIC else
                           "Crisp + AI/ML" if AI_ENABLED else "Crisp Logic",
        "agent_insights": agent_insights
    }

@app.post("/feedback")
def submit_feedback(feedback: FeedbackRequest):
    """
    Submit user feedback to train the AI model and agent
    """
    if not AI_ENABLED and not AGENT_ENABLED:
        raise HTTPException(status_code=503, detail="AI/Agent not available")

    try:
        # Normalize rating to 0-1 scale
        normalized_rating = feedback.rating / 5.0

        feedback_count = 0

        # Learn from feedback in AI engine
        if AI_ENABLED and ai_engine:
            ai_engine.learn_from_feedback(
                feedback.questionnaire,
                feedback.recipe,
                normalized_rating
            )
            feedback_count = len(ai_engine.user_history)

        # Process feedback in RAG agent
        if AGENT_ENABLED and rag_agent:
            rag_agent.process_feedback({
                "questionnaire": feedback.questionnaire,
                "recipe": feedback.recipe,
                "rating": feedback.rating,
                "comments": feedback.comments,
                "normalized_rating": normalized_rating
            })

        return {
            "status": "success",
            "message": "Thank you for your feedback! The AI Agent will use this to improve future recommendations.",
            "total_feedback_count": feedback_count,
            "agent_active": AGENT_ENABLED
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing feedback: {str(e)}")

@app.get("/ai/stats")
def ai_stats():
    """
    Get AI engine statistics
    """
    if not AI_ENABLED or not ai_engine:
        return {"ai_enabled": False, "message": "AI engine not available"}

    return {
        "ai_enabled": True,
        "total_interactions": len(ai_engine.user_history),
        "model_trained": hasattr(ai_engine.preference_model, 'feature_importances_'),
        "confidence_level": "high" if len(ai_engine.user_history) > 50 else "medium" if len(ai_engine.user_history) > 20 else "low",
        "nlp_available": ai_engine.nlp_model is not None
    }

# ---------------- Serial Dispenser Endpoints ----------------

@app.post("/dispense")
def dispense_recipe(request_body: dict):
    """
    Execute a dispense plan on the Arduino-controlled dispenser

    Request body:
    {
        "dispense_plan": [...],  # From /formulate endpoint
        "port": "COM3",          # Optional, default COM3
        "auto_home": true        # Optional, home before dispensing
    }
    """
    if not SERIAL_ENABLED:
        raise HTTPException(
            status_code=503,
            detail="Serial dispenser not available. Check that pyserial is installed and serial_dispenser.py is present."
        )

    dispense_plan = request_body.get("dispense_plan")
    if not dispense_plan:
        raise HTTPException(status_code=400, detail="dispense_plan is required")

    port = request_body.get("port", "COM3")
    auto_home = request_body.get("auto_home", False)

    try:
        # Initialize dispenser
        dispenser = ArduinoDispenser(port=port)

        # Optional: Home before dispensing
        if auto_home:
            dispenser.home()

        # Execute recipe
        result = dispenser.dispense_recipe(dispense_plan)

        # Close connection
        dispenser.close()

        return {
            "status": "success" if result["success"] else "partial_failure",
            "message": "Dispense complete" if result["success"] else "Dispense completed with errors",
            "details": result
        }

    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"Cannot connect to Arduino: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dispense error: {str(e)}")

@app.post("/dispense/test")
def test_dispense(request_body: dict):
    """
    Test dispense from a single valve

    Request body:
    {
        "valve": 2,
        "grams": 1.5,
        "port": "COM3"  # Optional
    }
    """
    if not SERIAL_ENABLED:
        raise HTTPException(status_code=503, detail="Serial dispenser not available")

    valve = request_body.get("valve")
    grams = request_body.get("grams")

    if valve is None or grams is None:
        raise HTTPException(status_code=400, detail="valve and grams are required")

    port = request_body.get("port", "COM3")

    try:
        dispenser = ArduinoDispenser(port=port)
        success = dispenser.dispense(valve, f"Test_Valve_{valve}", grams)
        dispenser.close()

        return {
            "status": "success" if success else "failed",
            "valve": valve,
            "grams": grams
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test dispense error: {str(e)}")

@app.post("/dispense/home")
def home_dispenser(request_body: dict = None):
    """
    Home the Cartesian system

    Request body (optional):
    {
        "port": "COM3"
    }
    """
    if not SERIAL_ENABLED:
        raise HTTPException(status_code=503, detail="Serial dispenser not available")

    port = "COM3"
    if request_body:
        port = request_body.get("port", "COM3")

    try:
        dispenser = ArduinoDispenser(port=port)
        success = dispenser.home()
        dispenser.close()

        return {
            "status": "success" if success else "failed",
            "message": "Homing complete" if success else "Homing failed"
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Homing error: {str(e)}")

@app.get("/dispense/status")
def dispenser_status():
    """Get dispenser availability status"""
    return {
        "serial_enabled": SERIAL_ENABLED,
        "arduino_available": SERIAL_ENABLED,
        "message": "Serial dispenser ready" if SERIAL_ENABLED else "Serial dispenser not available"
    }

# ---------------- UI (HTML form) ----------------

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def questionnaire_ui():
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Bespoke Perfume Creator</title>
  <link href="https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500;600&display=swap" rel="stylesheet">
  <style>
    :root {
      --primary: #2c3e50;
      --accent: #d4af37;
      --bg: #fafafa;
      --card-bg: #ffffff;
      --text: #333;
      --text-light: #666;
      --border: #e0e0e0;
      --shadow: rgba(0,0,0,0.08);
      --shadow-hover: rgba(0,0,0,0.15);
    }

    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    body {
      font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
      background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
      min-height: 100vh;
      padding: 20px;
      color: var(--text);
      line-height: 1.6;
    }

    .container {
      max-width: 1200px;
      margin: 0 auto;
    }

    header {
      text-align: center;
      margin-bottom: 40px;
      padding: 40px 20px;
      background: var(--card-bg);
      border-radius: 16px;
      box-shadow: 0 4px 20px var(--shadow);
    }

    h1 {
      font-family: 'Playfair Display', serif;
      font-size: 3rem;
      font-weight: 700;
      color: var(--primary);
      margin-bottom: 10px;
      letter-spacing: -0.5px;
    }

    .subtitle {
      font-size: 1.1rem;
      color: var(--text-light);
      font-weight: 300;
    }

    .form-container {
      background: var(--card-bg);
      border-radius: 16px;
      padding: 40px;
      box-shadow: 0 4px 20px var(--shadow);
      margin-bottom: 30px;
      transition: transform 0.3s ease;
    }

    .form-container:hover {
      box-shadow: 0 8px 30px var(--shadow-hover);
    }

    fieldset {
      border: 2px solid var(--border);
      border-radius: 12px;
      padding: 24px;
      margin-bottom: 30px;
      background: #fafbfc;
      transition: all 0.3s ease;
    }

    fieldset:hover {
      border-color: var(--accent);
      background: #fff;
    }

    legend {
      font-family: 'Playfair Display', serif;
      font-size: 1.4rem;
      font-weight: 600;
      color: var(--primary);
      padding: 0 12px;
    }

    .form-group {
      margin-bottom: 20px;
    }

    label {
      display: block;
      font-weight: 500;
      margin-bottom: 8px;
      color: var(--text);
      font-size: 0.95rem;
    }

    select, input[type="number"], input[type="text"] {
      width: 100%;
      padding: 12px 16px;
      border: 2px solid var(--border);
      border-radius: 8px;
      font-size: 1rem;
      font-family: inherit;
      transition: all 0.3s ease;
      background: white;
    }

    select:focus, input:focus {
      outline: none;
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(212, 175, 55, 0.1);
    }

    .row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin-top: 16px;
    }

    .slider-container {
      margin-bottom: 20px;
    }

    .slider-wrapper {
      display: flex;
      align-items: center;
      gap: 12px;
    }

    input[type="range"] {
      flex: 1;
      height: 6px;
      border-radius: 3px;
      background: linear-gradient(to right, #e0e0e0 0%, var(--accent) 100%);
      outline: none;
      -webkit-appearance: none;
    }

    input[type="range"]::-webkit-slider-thumb {
      -webkit-appearance: none;
      appearance: none;
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
      transition: transform 0.2s ease;
    }

    input[type="range"]::-webkit-slider-thumb:hover {
      transform: scale(1.2);
    }

    input[type="range"]::-moz-range-thumb {
      width: 20px;
      height: 20px;
      border-radius: 50%;
      background: var(--accent);
      cursor: pointer;
      border: none;
      box-shadow: 0 2px 8px rgba(0,0,0,0.2);
    }

    .slider-value {
      min-width: 40px;
      text-align: center;
      font-weight: 600;
      color: var(--accent);
      font-size: 1.1rem;
    }

    .checkbox-wrapper {
      display: flex;
      align-items: center;
      gap: 10px;
      padding: 12px;
      background: white;
      border-radius: 8px;
      cursor: pointer;
      transition: background 0.3s ease;
    }

    .checkbox-wrapper:hover {
      background: #f8f9fa;
    }

    input[type="checkbox"] {
      width: 20px;
      height: 20px;
      cursor: pointer;
      accent-color: var(--accent);
    }

    .occasions-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
      gap: 12px;
      margin-top: 12px;
    }

    .occasion-chip {
      padding: 12px 16px;
      border: 2px solid var(--border);
      border-radius: 8px;
      text-align: center;
      cursor: pointer;
      transition: all 0.3s ease;
      background: white;
      font-weight: 500;
      user-select: none;
    }

    .occasion-chip:hover {
      border-color: var(--accent);
      background: #fffbf0;
    }

    .occasion-chip.selected {
      background: var(--accent);
      color: white;
      border-color: var(--accent);
    }

    button {
      width: 100%;
      padding: 16px 32px;
      background: linear-gradient(135deg, var(--primary) 0%, #34495e 100%);
      color: white;
      border: none;
      border-radius: 12px;
      font-size: 1.1rem;
      font-weight: 600;
      cursor: pointer;
      transition: all 0.3s ease;
      box-shadow: 0 4px 15px rgba(44, 62, 80, 0.3);
      font-family: 'Playfair Display', serif;
      letter-spacing: 0.5px;
    }

    button:hover {
      transform: translateY(-2px);
      box-shadow: 0 6px 25px rgba(44, 62, 80, 0.4);
    }

    button:active {
      transform: translateY(0);
    }

    button:disabled {
      opacity: 0.6;
      cursor: not-allowed;
      transform: none;
    }

    .result-container {
      background: var(--card-bg);
      border-radius: 16px;
      padding: 40px;
      box-shadow: 0 4px 20px var(--shadow);
      margin-top: 30px;
      display: none;
    }

    .result-container.show {
      display: block;
      animation: slideIn 0.5s ease;
    }

    @keyframes slideIn {
      from {
        opacity: 0;
        transform: translateY(20px);
      }
      to {
        opacity: 1;
        transform: translateY(0);
      }
    }

    .recipe-header {
      text-align: center;
      margin-bottom: 30px;
      padding-bottom: 20px;
      border-bottom: 2px solid var(--border);
    }

    .recipe-header h2 {
      font-family: 'Playfair Display', serif;
      font-size: 2rem;
      color: var(--primary);
      margin-bottom: 8px;
    }

    .ingredient-card {
      background: #fafbfc;
      border-left: 4px solid var(--accent);
      padding: 16px 20px;
      margin-bottom: 12px;
      border-radius: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
      transition: transform 0.2s ease;
    }

    .ingredient-card:hover {
      transform: translateX(5px);
      background: #f0f4f8;
    }

    .ingredient-name {
      font-weight: 600;
      color: var(--primary);
      font-size: 1.05rem;
    }

    .ingredient-note {
      display: inline-block;
      padding: 4px 12px;
      background: var(--accent);
      color: white;
      border-radius: 12px;
      font-size: 0.8rem;
      margin-left: 10px;
      font-weight: 500;
    }

    .ingredient-percent {
      font-size: 1.2rem;
      font-weight: 700;
      color: var(--accent);
    }

    .loading {
      text-align: center;
      padding: 40px;
      font-size: 1.1rem;
      color: var(--text-light);
    }

    .spinner {
      border: 4px solid #f3f3f3;
      border-top: 4px solid var(--accent);
      border-radius: 50%;
      width: 40px;
      height: 40px;
      animation: spin 1s linear infinite;
      margin: 20px auto;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }

    .error {
      background: #fee;
      border-left: 4px solid #c00;
      padding: 16px;
      border-radius: 8px;
      color: #c00;
      margin-top: 20px;
    }

    .info-badge {
      display: inline-block;
      background: #e3f2fd;
      color: #1976d2;
      padding: 6px 12px;
      border-radius: 6px;
      font-size: 0.9rem;
      margin: 5px 5px 0 0;
      font-weight: 500;
    }

    @media (max-width: 768px) {
      h1 { font-size: 2rem; }
      .form-container, header { padding: 20px; }
      .row { grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>✨ Bespoke Perfume Creator</h1>
      <p class="subtitle">Craft your signature scent with our AI-powered perfume formulation system</p>
    </header>

    <div class="form-container">
      <form id="perfumeForm">
        <fieldset>
          <legend>🎨 Fragrance Profile</legend>
          <div class="form-group">
            <label for="overall">Overall Character</label>
            <select id="overall">
              <option value="fresh">Fresh & Clean</option>
              <option value="floral">Floral & Romantic</option>
              <option value="warm_spicy">Warm & Spicy</option>
              <option value="sweet">Sweet & Gourmand</option>
              <option value="woody">Woody & Earthy</option>
              <option value="resinous">Resinous & Amber</option>
            </select>
          </div>

          <div class="form-group">
            <label for="gender_vibe">Gender Expression</label>
            <select id="gender_vibe">
              <option value="unisex">Unisex</option>
              <option value="feminine">Feminine</option>
              <option value="masculine">Masculine</option>
            </select>
          </div>

          <div class="form-group">
            <label>Occasions</label>
            <div class="occasions-grid">
              <div class="occasion-chip" data-value="daily">Daily Wear</div>
              <div class="occasion-chip" data-value="special">Special Events</div>
              <div class="occasion-chip" data-value="summer">Summer</div>
              <div class="occasion-chip" data-value="winter">Winter</div>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>⚗️ Performance & Specifications</legend>

          <div class="slider-container">
            <label>Longevity</label>
            <div class="slider-wrapper">
              <input type="range" id="longevity" min="0" max="5" value="3" step="1">
              <span class="slider-value" id="longevity-val">3</span>
            </div>
          </div>

          <div class="slider-container">
            <label>Projection (Sillage)</label>
            <div class="slider-wrapper">
              <input type="range" id="projection" min="0" max="5" value="3" step="1">
              <span class="slider-value" id="projection-val">3</span>
            </div>
          </div>

          <div class="row">
            <div class="form-group">
              <label for="bottle_ml">Bottle Size (ml)</label>
              <input id="bottle_ml" type="number" min="5" step="5" value="30">
            </div>

            <div class="form-group">
              <label for="concentration">Concentration</label>
              <select id="concentration">
                <option value="EdC">Eau de Cologne (3-5%)</option>
                <option value="EDT">Eau de Toilette (5-15%)</option>
                <option value="EDP" selected>Eau de Parfum (15-20%)</option>
                <option value="Parfum">Parfum (20-30%)</option>
              </select>
            </div>
          </div>

          <div class="checkbox-wrapper">
            <input id="sensitive" type="checkbox">
            <label for="sensitive" style="margin: 0; cursor: pointer;">I have sensitive skin</label>
          </div>
        </fieldset>

        <fieldset>
          <legend>🌸 Scent Family Preferences</legend>
          <div class="slider-container">
            <label>Citrus</label>
            <div class="slider-wrapper">
              <input type="range" id="citrus" min="0" max="5" value="5" step="1">
              <span class="slider-value" id="citrus-val">5</span>
            </div>
          </div>

          <div class="slider-container">
            <label>Floral</label>
            <div class="slider-wrapper">
              <input type="range" id="floral" min="0" max="5" value="3" step="1">
              <span class="slider-value" id="floral-val">3</span>
            </div>
          </div>

          <div class="slider-container">
            <label>Woody</label>
            <div class="slider-wrapper">
              <input type="range" id="woody" min="0" max="5" value="2" step="1">
              <span class="slider-value" id="woody-val">2</span>
            </div>
          </div>

          <div class="slider-container">
            <label>Green</label>
            <div class="slider-wrapper">
              <input type="range" id="green" min="0" max="5" value="4" step="1">
              <span class="slider-value" id="green-val">4</span>
            </div>
          </div>

          <div class="slider-container">
            <label>Gourmand</label>
            <div class="slider-wrapper">
              <input type="range" id="gourmand" min="0" max="5" value="1" step="1">
              <span class="slider-value" id="gourmand-val">1</span>
            </div>
          </div>

          <div class="slider-container">
            <label>Spicy</label>
            <div class="slider-wrapper">
              <input type="range" id="spicy" min="0" max="5" value="1" step="1">
              <span class="slider-value" id="spicy-val">1</span>
            </div>
          </div>

          <div class="slider-container">
            <label>Powdery</label>
            <div class="slider-wrapper">
              <input type="range" id="powdery" min="0" max="5" value="1" step="1">
              <span class="slider-value" id="powdery-val">1</span>
            </div>
          </div>

          <div class="slider-container">
            <label>Resinous</label>
            <div class="slider-wrapper">
              <input type="range" id="resinous" min="0" max="5" value="1" step="1">
              <span class="slider-value" id="resinous-val">1</span>
            </div>
          </div>
        </fieldset>

        <fieldset>
          <legend>💝 Personal Preferences</legend>
          <div class="form-group">
            <label for="loves">Ingredients I Love</label>
            <input id="loves" type="text" placeholder="e.g., jasmine, sandalwood, vanilla">
          </div>

          <div class="form-group">
            <label for="dislikes">Ingredients to Avoid</label>
            <input id="dislikes" type="text" placeholder="e.g., patchouli, musk">
          </div>

          <div class="form-group">
            <label for="scent_description">🤖 Describe Your Perfect Scent (AI-Powered)</label>
            <textarea id="scent_description" rows="3" style="width: 100%; padding: 12px; border: 2px solid var(--border); border-radius: 8px; font-family: inherit; font-size: 1rem; resize: vertical;" placeholder="e.g., I love fresh morning dew with a hint of jasmine, something that reminds me of spring gardens..."></textarea>
            <small style="color: var(--text-light); font-size: 0.85rem;">Our AI will analyze your description using natural language processing</small>
          </div>
        </fieldset>

        <button type="submit" id="generateBtn">Create My Perfume</button>
      </form>
    </div>

    <div class="result-container" id="results">
      <div class="recipe-header">
        <h2>Your Custom Perfume</h2>
      </div>
      <div id="recipeContent"></div>
    </div>
  </div>

<script>
// Wait for DOM to be fully loaded
document.addEventListener('DOMContentLoaded', function() {
  // Slider value updates
  const sliders = ['longevity', 'projection', 'citrus', 'floral', 'woody', 'green', 'gourmand', 'spicy', 'powdery', 'resinous'];
  sliders.forEach(id => {
    const slider = document.getElementById(id);
    const display = document.getElementById(id + '-val');
    if (slider && display) {
      slider.addEventListener('input', (e) => {
        display.textContent = e.target.value;
      });
    } else {
      console.error(`Slider or display not found for: ${id}`);
    }
  });

  // Occasion chips selection
  const occasionChips = document.querySelectorAll('.occasion-chip');
  const selectedOccasions = new Set();

  occasionChips.forEach(chip => {
    chip.addEventListener('click', () => {
      const value = chip.dataset.value;
      if (selectedOccasions.has(value)) {
        selectedOccasions.delete(value);
        chip.classList.remove('selected');
      } else {
        selectedOccasions.add(value);
        chip.classList.add('selected');
      }
    });
  });

  // Make selectedOccasions accessible globally for form submission
  window.selectedOccasions = selectedOccasions;

  // Form submission
  document.getElementById('perfumeForm').addEventListener('submit', async (e) => {
    e.preventDefault();

    const btn = document.getElementById('generateBtn');
    const results = document.getElementById('results');
    const recipeContent = document.getElementById('recipeContent');

    btn.disabled = true;
    btn.textContent = 'Creating Your Perfume...';

    recipeContent.innerHTML = '<div class="loading"><div class="spinner"></div>Formulating your bespoke fragrance...</div>';
    results.classList.add('show');

    const get = id => document.getElementById(id).value;
    const fam = (id) => Number(document.getElementById(id).value);

    const body = {
      overall: get('overall'),
      family_ratings: {
        citrus: fam('citrus'), floral: fam('floral'), woody: fam('woody'), green: fam('green'),
        gourmand: fam('gourmand'), spicy: fam('spicy'), powdery: fam('powdery'), resinous: fam('resinous')
      },
      gender_vibe: get('gender_vibe'),
      longevity_0_5: Number(get('longevity')),
      projection_0_5: Number(get('projection')),
      occasions: Array.from(window.selectedOccasions || []),
      sensitive_skin: document.getElementById('sensitive').checked,
      dislikes: get('dislikes').split(',').map(s=>s.trim()).filter(Boolean),
      loves: get('loves').split(',').map(s=>s.trim()).filter(Boolean),
      bottle_ml: Number(get('bottle_ml')),
      concentration: get('concentration'),
      scent_description: get('scent_description')
    };

    console.log('Sending request with body:', body);

    // Store for potential feedback
    window.currentQuestionnaire = body;

    try {
      const res = await fetch('/formulate', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body)
      });

      console.log('Response status:', res.status);

      if (!res.ok) {
        const errorText = await res.text();
        console.error('Error response:', errorText);
        throw new Error(`Failed to generate recipe: ${res.status} - ${errorText}`);
      }

      const data = await res.json();
      console.log('Received data:', data);
      displayRecipe(data);
    } catch (error) {
      console.error('Error details:', error);
      recipeContent.innerHTML = '<div class="error">\\u274C Error: ' + error.message + '</div>';
    } finally {
      btn.disabled = false;
      btn.textContent = 'Create My Perfume';
    }
  });
}); // End of DOMContentLoaded

function displayRecipe(data) {
  console.log('displayRecipe called with:', data);

  if (!data || !data.recipe) {
    console.error('Invalid data received:', data);
    document.getElementById('recipeContent').innerHTML = '<div class="error">\\u274C Error: Invalid recipe data received</div>';
    return;
  }

  const recipe = data.recipe;
  const plan = data.dispense_plan;
  const aiInsights = data.ai_insights || {};
  const aiPowered = data.ai_powered || false;
  const agentPowered = data.agent_powered || false;
  const agentInsights = data.agent_insights || {};

  // Store for feedback
  window.currentRecipe = recipe;

  let html = '<div style="margin-bottom: 30px;">';

  // Agent/AI Badge
  if (agentPowered && agentInsights.ai_generated) {
    html += '<div style="text-align: center; margin-bottom: 20px;">';
    html += '<span style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 10px 20px; border-radius: 25px; font-size: 1rem; font-weight: 700; box-shadow: 0 4px 15px rgba(240, 147, 251, 0.4);">\\uD83E\\uDDE0 AI Agent + RAG Powered</span>';
    html += '</div>';
  } else if (aiPowered) {
    html += '<div style="text-align: center; margin-bottom: 20px;">';
    html += '<span style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 8px 16px; border-radius: 20px; font-size: 0.9rem; font-weight: 600;">\\uD83E\\uDD16 AI-Enhanced Formula</span>';
    html += '</div>';
  }

  // Agent Insights Section
  if (agentInsights && Object.keys(agentInsights).length > 0 && agentInsights.concept) {
    html += '<div style="background: linear-gradient(135deg, #ffecd215 0%, #fcb69f15 100%); padding: 25px; border-radius: 12px; margin-bottom: 30px; border-left: 4px solid #ff6b6b;">';
    html += "<h3 style='font-family: Playfair Display, serif; color: #ff6b6b; margin-bottom: 12px;'>\\uD83C\\uDFA8 AI Perfumer Vision</h3>";

    if (agentInsights.concept) {
      html += `<p style="margin: 12px 0; font-style: italic; color: #555; font-size: 1.05rem;">"${agentInsights.concept}"</p>`;
    }

    if (agentInsights.reasoning) {
      html += `<p style="margin: 12px 0;"><strong>Formulation Strategy:</strong> ${agentInsights.reasoning}</p>`;
    }

    if (agentInsights.model_used) {
      html += `<p style="margin: 8px 0; font-size: 0.9rem; color: #888;"><strong>Model:</strong> ${agentInsights.model_used} | <strong>Knowledge Retrieved:</strong> ${agentInsights.retrieved_knowledge_count || 0} expert documents</p>`;
    }

    if (agentInsights.tips && agentInsights.tips.length > 0) {
      html += '<p style="margin: 12px 0;"><strong>Expert Tips:</strong></p><ul style="margin: 5px 0; padding-left: 20px;">';
      agentInsights.tips.forEach(tip => {
        html += `<li style="margin: 6px 0;">${tip}</li>`;
      });
      html += '</ul>';
    }
    html += '</div>';
  }

  // AI Insights Section
  if (aiInsights && Object.keys(aiInsights).length > 0) {
    html += '<div style="background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%); padding: 20px; border-radius: 12px; margin-bottom: 30px; border-left: 4px solid #667eea;">';
    html += '<h3 style="font-family: Playfair Display, serif; color: #667eea; margin-bottom: 12px;">\\uD83E\\uDDE0 AI Insights</h3>';

    if (aiInsights.confidence_score) {
      const confidencePct = (aiInsights.confidence_score * 100).toFixed(0);
      html += `<p style="margin: 8px 0;"><strong>Confidence:</strong> ${confidencePct}% (${aiInsights.personalization_level} personalization)</p>`;
    }

    if (aiInsights.predicted_satisfaction) {
      const satisfactionPct = (aiInsights.predicted_satisfaction * 100).toFixed(0);
      html += `<p style="margin: 8px 0;"><strong>Predicted Satisfaction:</strong> ${satisfactionPct}%</p>`;
    }

    if (aiInsights.suggestions && aiInsights.suggestions.length > 0) {
      html += '<p style="margin: 8px 0;"><strong>Notes:</strong></p><ul style="margin: 5px 0; padding-left: 20px;">';
      aiInsights.suggestions.forEach(s => {
        html += `<li style="margin: 4px 0;">${s}</li>`;
      });
      html += '</ul>';
    }
    html += '</div>';
  }

  // Note targets
  html += '<h3 style="font-family: Playfair Display, serif; color: var(--primary); margin-bottom: 16px;">\\uD83D\\uDCCA Composition</h3>';
  html += '<div style="display: flex; gap: 12px; margin-bottom: 30px; flex-wrap: wrap;">';
  html += `<span class="info-badge">Top: ${recipe.note_targets.top.toFixed(0)}%</span>`;
  html += `<span class="info-badge">Middle: ${recipe.note_targets.middle.toFixed(0)}%</span>`;
  html += `<span class="info-badge">Base: ${recipe.note_targets.base.toFixed(0)}%</span>`;
  html += '</div>';

  // Ingredients
  html += '<h3 style="font-family: Playfair Display, serif; color: var(--primary); margin-bottom: 16px;">\\uD83E\\uDDEA Essential Oils</h3>';
  recipe.ingredients.forEach(ing => {
    const noteLabels = {'top': 'Top', 'middle': 'Middle', 'base': 'Base'};
    html += `
      <div class="ingredient-card">
        <div>
          <span class="ingredient-name">${ing.oil_name}</span>
          <span class="ingredient-note">${noteLabels[ing.note]}</span>
        </div>
        <span class="ingredient-percent">${ing.percent.toFixed(2)}%</span>
      </div>
    `;
  });

  // Dilution info
  html += '<div style="margin-top: 30px; padding: 20px; background: #f0f4f8; border-radius: 12px;">';
  html += `<h3 style="font-family: Playfair Display, serif; color: var(--primary); margin-bottom: 12px;">\\uD83D\\uDCA7 Dilution</h3>`;
  html += `<p><strong>Type:</strong> ${recipe.dilution.type}</p>`;
  html += `<p><strong>Alcohol:</strong> ${recipe.dilution.alcohol_pct.toFixed(1)}%</p>`;
  html += `<p><strong>Bottle Size:</strong> ${recipe.dilution.bottle_ml}ml</p>`;
  html += '</div>';

  // Dispense Plan (Robot Instructions)
  if (plan && plan.length > 0) {
    html += '<div style="margin-top: 30px; padding: 20px; background: #fff; border: 2px solid #28a745; border-radius: 12px;">';
    html += `<h3 style="font-family: Playfair Display, serif; color: #28a745; margin-bottom: 16px;">\\uD83E\\uDD16 Robot Dispense Instructions</h3>`;
    html += '<div style="overflow-x: auto;">';
    html += '<table style="width: 100%; border-collapse: collapse; font-size: 0.95rem;">';
    html += '<thead><tr style="background: #28a74515; border-bottom: 2px solid #28a745;">';
    html += '<th style="padding: 12px; text-align: left; font-weight: 600;">Valve #</th>';
    html += '<th style="padding: 12px; text-align: left; font-weight: 600;">Ingredient</th>';
    html += '<th style="padding: 12px; text-align: right; font-weight: 600;">Volume (ml)</th>';
    html += '<th style="padding: 12px; text-align: right; font-weight: 600;">Weight (g)</th>';
    html += '</tr></thead><tbody>';

    plan.forEach((item, idx) => {
      const rowBg = idx % 2 === 0 ? '#f9f9f9' : '#fff';
      html += `<tr style="background: ${rowBg}; border-bottom: 1px solid #eee;">`;
      html += `<td style="padding: 10px; font-weight: 600; color: #28a745;">${item.valve}</td>`;
      html += `<td style="padding: 10px;">${item.oil_name}</td>`;
      html += `<td style="padding: 10px; text-align: right;">${item.ml}</td>`;
      html += `<td style="padding: 10px; text-align: right;">${item.grams || 'N/A'}</td>`;
      html += '</tr>';
    });

    html += '</tbody></table>';
    html += '</div>';

    // Add Start Dispensing Button
    html += '<div style="margin-top: 20px; text-align: center;">';
    html += '<button onclick="startDispensing()" id="dispenseBtn" style="width: auto; padding: 16px 48px; background: linear-gradient(135deg, #28a745 0%, #20c997 100%); color: white; border: none; border-radius: 12px; font-size: 1.1rem; font-weight: 600; cursor: pointer; box-shadow: 0 4px 15px rgba(40, 167, 69, 0.4); font-family: Playfair Display, serif;">\\uD83E\\uDD16 Start Dispensing</button>';
    html += '</div>';
    html += '<div id="dispenseStatus" style="margin-top: 16px; padding: 16px; border-radius: 8px; display: none;"></div>';

    html += '</div>';
  }

  // Feedback section
  if (aiPowered) {
    html += '<div style="margin-top: 30px; padding: 25px; background: #fff; border: 2px solid #667eea; border-radius: 12px;">';
    html += '<h3 style="font-family: Playfair Display, serif; color: #667eea; margin-bottom: 16px;">\\uD83D\\uDCAC Help Our AI Learn</h3>';
    html += '<p style="margin-bottom: 12px; color: var(--text-light);">Rate this recommendation to help improve future suggestions</p>';
    html += '<div style="display: flex; gap: 8px; margin-bottom: 12px;">';
    for (let i = 1; i <= 5; i++) {
      html += `<button onclick="submitRating(${i})" style="padding: 10px 16px; background: #f0f0f0; border: 2px solid #ddd; border-radius: 8px; cursor: pointer; transition: all 0.3s; font-size: 1.2rem;" onmouseover="this.style.background='#667eea'; this.style.color='white'; this.style.borderColor='#667eea'" onmouseout="this.style.background='#f0f0f0'; this.style.color='#333'; this.style.borderColor='#ddd'">\\u2B50 ${i}</button>`;
    }
    html += '</div>';
    html += '<div id="feedbackMessage" style="margin-top: 12px; font-weight: 500;"></div>';
    html += '</div>';
  }

  html += '</div>';

  document.getElementById('recipeContent').innerHTML = html;
  window.currentDispensePlan = plan || null;
  document.getElementById('results').scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Submit rating function
async function submitRating(rating) {
  if (!window.currentQuestionnaire || !window.currentRecipe) {
    alert('Please generate a recipe first');
    return;
  }

  try {
    const res = await fetch('/feedback', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        questionnaire: window.currentQuestionnaire,
        recipe: window.currentRecipe,
        rating: rating,
        comments: ''
      })
    });

    const data = await res.json();

    if (res.ok) {
      document.getElementById('feedbackMessage').innerHTML =
        `<span style="color: #28a745;">\\u2713 Thank you! Your ${rating}-star rating has been recorded. Total feedback: ${data.total_feedback_count}</span>`;
    } else {
      document.getElementById('feedbackMessage').innerHTML =
        '<span style="color: #dc3545;">\\u2717 Failed to submit feedback</span>';
    }
  } catch (error) {
    document.getElementById('feedbackMessage').innerHTML =
      '<span style="color: #dc3545;">\\u2717 Error: ' + error.message + '</span>';
  }
}

// Start dispensing function
async function startDispensing() {
  if (!window.currentDispensePlan) {
    alert('No dispense plan available. Please generate a recipe first.');
    return;
  }

  const btn = document.getElementById('dispenseBtn');
  const status = document.getElementById('dispenseStatus');

  // Disable button
  btn.disabled = true;
  btn.textContent = '\\u23F3 Dispensing in Progress...';
  btn.style.opacity = '0.6';

  // Show status box
  status.style.display = 'block';
  status.style.background = '#fff3cd';
  status.style.border = '2px solid #ffc107';
  status.style.color = '#856404';
  status.innerHTML = '<div class="spinner"></div><p style="margin-top: 12px; font-weight: 600;">\\uD83E\\uDD16 Connecting to Arduino...</p>';

  try {
    const res = await fetch('/dispense', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        dispense_plan: window.currentDispensePlan,
        port: 'COM13',
        auto_home: true  // Auto-home before dispensing
      })
    });

    const data = await res.json();

    if (res.ok && data.status === 'success') {
      // Success
      status.style.background = '#d4edda';
      status.style.border = '2px solid #28a745';
      status.style.color = '#155724';
      status.innerHTML = `
        <h4 style="margin-bottom: 12px; font-size: 1.2rem;">\\u2705 Dispensing Complete!</h4>
        <p><strong>Status:</strong> ${data.message}</p>
        <p style="margin-top: 8px; font-size: 0.9rem;">Your perfume is ready!</p>
      `;

      btn.textContent = '\\u2705 Dispensing Complete';
      btn.style.background = 'linear-gradient(135deg, #28a745 0%, #20c997 100%)';
    } else {
      // Partial failure
      status.style.background = '#fff3cd';
      status.style.border = '2px solid #ffc107';
      status.style.color = '#856404';
      status.innerHTML = `
        <h4 style="margin-bottom: 12px; font-size: 1.2rem;">\\u26A0\\uFE0F Dispensing Completed with Issues</h4>
        <p><strong>Status:</strong> ${data.message || 'Some errors occurred'}</p>
        <p style="margin-top: 8px; font-size: 0.9rem;">Check the console for details.</p>
      `;

      btn.disabled = false;
      btn.textContent = '\\uD83D\\uDD04 Retry Dispensing';
      btn.style.opacity = '1';
    }
  } catch (error) {
    // Error
    status.style.background = '#f8d7da';
    status.style.border = '2px solid #dc3545';
    status.style.color = '#721c24';

    let errorMsg = error.message;
    if (errorMsg.includes('503')) {
      errorMsg = 'Arduino not connected. Please check USB connection and ensure pyserial is installed.';
    } else if (errorMsg.includes('500')) {
      errorMsg = 'Dispensing error occurred. Check Arduino connection and calibration.';
    }

    status.innerHTML = `
      <h4 style="margin-bottom: 12px; font-size: 1.2rem;">\\u274C Dispensing Failed</h4>
      <p><strong>Error:</strong> ${errorMsg}</p>
      <p style="margin-top: 8px; font-size: 0.9rem;">
        <strong>Troubleshooting:</strong><br>
        \\u2022 Check Arduino is connected to COM13<br>
        \\u2022 Ensure USB cable is plugged in<br>
        \\u2022 Verify pyserial is installed: <code>pip install pyserial</code><br>
        \\u2022 Check Arduino sketch is uploaded and running
      </p>
    `;

    btn.disabled = false;
    btn.textContent = '\\uD83D\\uDD04 Retry Dispensing';
    btn.style.opacity = '1';
  }
}
</script>
</body>
</html>
    """

if __name__ == "__main__":
    import uvicorn, os
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
