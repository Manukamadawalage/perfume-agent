# Perfume Formulation Agent

An AI-powered perfume formulation system that creates custom fragrance recipes based on user preferences and automatically dispenses them using Arduino-controlled hardware.

## Features

### Core Formulation
- **Preference-based recipe generation** - Takes questionnaire input (scent families, occasions, intensity, etc.)
- **Smart oil ranking** - Uses role weights and family vectors to select optimal oils
- **Safety-first allocation** - Respects IFRA safety caps (max_pct) for each oil
- **Note pyramid structure** - Builds professional top/middle/base note compositions
- **Cap-safe normalization** - Ensures recipes total exactly 100% without breaking safety limits

### Advanced Intelligence
- **NLP-based preference detection** - Automatically extracts likes/dislikes from natural language
  - Example: "I love rose but allergic to patchouli" → Auto-detects preferences
- **Essential oils protection** - Ensures fixatives and blenders aren't missed (Frankincense, Sandalwood, Cedarwood, Lavender)
- **Precious oils exception** - Allows expensive/strong oils to be used at optimal low percentages (Jasmine, Rose, Ylang-Ylang)
- **Quality filtering** - Only includes oils that score well for your preferences (min_score threshold)
- **Ingredient consolidation** - Merges duplicate oils across notes for efficient dispensing

### Hardware Integration
- **Arduino Mega dispensing** - Serial communication with Cartesian dispensing system
- **Flow rate calibration** - Accurate weight-based dispensing with calibration wizard
- **Valve mapping** - Maps oils to physical dispenser valve positions
- **Autonomous operation** - Arduino handles positioning and dispensing independently

## Quick Start

### System Status: ✅ FULLY OPERATIONAL
- Web interface with beautiful UI
- Recipe formulation with 16 essential oils
- Arduino serial communication (COM13)
- All core features working

### 3-Step Start

**1. Start the Server**
```bash
# Option A: Use the batch file
start_server.bat

# Option B: Command line
python app.py
uvicorn app:app --reload --port 8000
```

**2. Open Web Interface**
Open browser to: http://localhost:8000

**3. Create Your Perfume!**
- Fill out fragrance profile (citrus, floral, woody preferences)
- Click "Create My Perfume"
- Review your custom recipe
- Click "Start Dispensing" (if Arduino connected to COM13)

**Stop Server:**
```bash
stop_server.bat
# OR press Ctrl+C in terminal
```

### First Time Setup

1. **Install dependencies:**
   ```bash
   python -m venv .venv
   .venv\Scripts\Activate.ps1  # Windows
   # source .venv/bin/activate  # macOS/Linux
   pip install -r requirements.txt
   ```

2. **Configure Arduino** (if using hardware):
   - Connect Arduino Mega to **COM13**
   - See ARDUINO_SETUP_GUIDE.md for sketch upload
   - Test with: `close_arduino_ide.bat` (if port locked)

## Architecture

### System Overview

This is a **full-stack IoT application** combining:
- **Frontend:** HTML5 + JavaScript (browser-based UI)
- **Backend:** Python + FastAPI (formulation logic)
- **Hardware:** Arduino Mega (Cartesian dispensing system)

**How it works:**
```
User Browser → FastAPI Server → Formulation Engine → Arduino → Physical Dispensing
```

**📖 Detailed Architecture:** See [WEB_ARCHITECTURE.md](WEB_ARCHITECTURE.md) for complete system workflow and communication protocols.

### Technology Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Frontend | HTML5 + CSS3 + JavaScript | User interface |
| Backend | Python 3.x + FastAPI | Business logic & API |
| Server | Uvicorn (ASGI) | HTTP server (port 8000) |
| Communication | JSON over HTTP | Frontend ↔ Backend |
| Hardware I/O | PySerial (UART) | Backend ↔ Arduino (COM13) |
| Data Storage | JSON files | Essential oils database |
| Microcontroller | Arduino Mega 2560 | Stepper motors + valves |

## Project Structure

```
perfume-agent/
├── 📄 Documentation
│   ├── README.md                   # This file - Quick start guide
│   ├── WEB_ARCHITECTURE.md         # Complete system architecture
│   └── ARDUINO_SETUP_GUIDE.md      # Hardware setup guide
│
├── 🌐 Web Application
│   ├── app.py                      # FastAPI server + embedded HTML/JS
│   └── models.py                   # Data models (Questionnaire, Recipe)
│
├── 🧪 Formulation Engine (Core Logic)
│   ├── logic_targets.py            # User preference vector
│   ├── logic_select.py             # Oil ranking algorithm
│   ├── logic_allocate_improved.py  # Smart allocation (quality filtering)
│   ├── logic_consolidate.py        # Merge duplicate oils
│   ├── logic_normalize.py          # Cap-safe 100% normalization
│   ├── logic_validate.py           # Validation & sanity checks
│   ├── logic_dispense.py           # % → ml/grams conversion
│   ├── logic_utils.py              # Utility functions
│   └── nlp_dislike_detector.py     # NLP preference extraction
│
├── 🤖 Hardware Interface
│   ├── serial_dispenser.py         # Arduino serial communication
│   └── calibration.py              # Flow rate calibration wizard
│
├── 🧠 AI/RAG (Optional, currently disabled)
│   ├── ai_engine.py                # Machine learning engine
│   └── rag_agent.py                # RAG-based assistant
│
├── 📊 Data Files
│   ├── oils.json                   # 16 essential oils database
│   ├── valve_map.json              # Oil → Valve mapping
│   └── valve_calibration.json      # Flow rates (g/s)
│
├── 📚 Knowledge Base
│   └── kb/
│       ├── pyramids.md             # Note pyramid structures
│       ├── safety_rules.md         # IFRA safety guidelines
│       └── accords.md              # Fragrance accords
│
└── 🔧 Utilities
    ├── start_server.bat            # Easy server startup
    ├── stop_server.bat             # Easy server shutdown
    ├── close_arduino_ide.bat       # Fix COM port locks
    └── find_port_user.bat          # Diagnose port issues
```

## Configuration

### Recipe Quality Settings

Edit `logic_allocate_improved.py:31` to adjust quality controls:

```python
def allocate(...,
             min_score=0.12,    # Minimum quality score (0-1)
             max_oils=10,       # Maximum oils per recipe
             min_percent=1.0,   # Minimum % per oil
             allow_essential_oils=True):  # Always include fixatives
```

**Recommended configurations:**

**Budget-Friendly** (Fewer oils, focused):
```python
min_score = 0.15
max_oils = 8
min_percent = 2.0
```

**Balanced** (Current default):
```python
min_score = 0.12
max_oils = 10
min_percent = 1.0
```

**Premium/Complex** (More variety):
```python
min_score = 0.08
max_oils = 15
min_percent = 0.5
```

### Safety Mechanisms

**Essential Oils Whitelist** (`logic_allocate_improved.py:13`)
These oils are always considered even if score is low:
- Frankincense (fixative for longevity)
- Sandalwood (base fixative)
- Cedarwood (woody structure)
- Lavender (universal blender)

**Precious Oils Exception** (`logic_allocate_improved.py:22`)
These oils can be used at 0.5%+ instead of normal 1.0% minimum:
- Jasmine absolute (very strong, expensive)
- Rose absolute (very strong, expensive)
- Ylang-Ylang (can be overwhelming >1%)
- Clove bud (very potent)
- Vanilla absolute (strong, works at low %)

## API Usage

### POST /formulate

Create a perfume recipe from preferences.

**Request body:**
```json
{
  "scent_families": {
    "citrus": 4,
    "floral": 5,
    "woody": 3,
    "oriental": 0,
    "fresh": 2,
    "fruity": 1
  },
  "occasions": ["daily", "summer"],
  "longevity": "moderate",
  "projection": "moderate",
  "overall_character": "romantic",
  "scent_description": "I love rose and jasmine but allergic to patchouli",
  "loves": ["bergamot"],
  "dislikes": ["vanilla"],
  "sensitive_skin": false,
  "bottle_ml": 30,
  "concentration_pct": 15
}
```

**Response:**
```json
{
  "recipe": {
    "items": [
      {"oil_name": "Bergamot (expressed)", "percent": 12.0, "note": "top"},
      {"oil_name": "Lavender", "percent": 15.0, "note": "middle"},
      {"oil_name": "Rose absolute", "percent": 10.0, "note": "middle"},
      {"oil_name": "Sandalwood", "percent": 15.0, "note": "base"},
      ...
    ],
    "total_percent": 100.0
  },
  "dispense_plan": [
    {"oil_name": "Bergamot (expressed)", "grams": 0.54, "ml": 0.6, "valve_id": 0},
    ...
  ],
  "metadata": {
    "total_oils": 8,
    "total_concentrate_ml": 4.5,
    "total_concentrate_g": 4.05,
    ...
  }
}
```

### Natural Language Preference Detection

The system automatically detects preferences from `scent_description`:

**Supported patterns:**
- Allergies: "allergic to patchouli", "sensitive to vanilla"
- Dislikes: "hate musk", "avoid lavender", "don't like sandalwood"
- Likes: "love rose", "want jasmine", "prefer bergamot"

**Examples:**
```
"I love fresh citrus scents but allergic to patchouli"
→ Loves: [Citrus oils], Dislikes: [Patchouli]

"Something floral with rose, please avoid vanilla - too sweet"
→ Loves: [Rose], Dislikes: [Vanilla]

"I want woody notes without sandalwood (gives me headaches)"
→ Loves: [Woody oils], Dislikes: [Sandalwood]
```

## Arduino Hardware Setup

### Requirements
- Arduino Mega 2560
- X-Y Cartesian system (lead screw based)
- Solenoid valves (one per oil)
- Stepper motors + drivers
- Scale/weight sensor (for calibration)

### Hardware Connection

1. **Upload Arduino sketch** (create your own based on serial protocol below)

2. **Serial Protocol:**
   ```
   Python → Arduino: <valve_id,time_ms>
   Arduino → Python: MOVING / POSITIONED / DISPENSING / DONE / ERROR
   ```

3. **Connect to PC:**
   - Update COM port in `serial_dispenser.py:__init__`
   - Default: COM3 (Windows) or /dev/ttyUSB0 (Linux)

### Calibration

Run the calibration wizard to measure flow rates:

```bash
python calibration.py
```

This will:
1. Connect to Arduino
2. Prompt you to select valve and oil
3. Dispense for 10 seconds
4. Ask you to weigh the result
5. Calculate flow rate (grams/second)
6. Save to `valve_calibration.json`

**Example calibration file:**
```json
{
  "valve_0_Bergamot_expressed": 0.85,
  "valve_1_Lemon": 0.82,
  "valve_2_Lavender": 0.78,
  "valve_3_Rose_absolute": 0.45,
  ...
}
```

### Dispenser API Endpoints

**POST /dispense** - Execute recipe on hardware
```json
{
  "dispense_plan": [...],
  "port": "COM3"
}
```

**POST /dispense/test** - Test single valve
```json
{
  "valve_id": 0,
  "time_ms": 5000,
  "port": "COM3"
}
```

**POST /dispense/home** - Home the Cartesian system
```json
{
  "port": "COM3"
}
```

**GET /dispense/status** - Check if dispenser is connected

## Troubleshooting

### Arduino / Serial Issues

**Issue: Cannot connect to Arduino on COM13**

**Solution:**
1. Check Arduino is connected via USB
2. Close Arduino IDE Serial Monitor (locks the port)
3. Use utility: `close_arduino_ide.bat` to force close Arduino IDE
4. Check COM port in Device Manager (Windows)
5. Update port in code if needed: `serial_dispenser.py` line 22

**Issue: "PermissionError: Access denied" on COM13**

**Solution:**
```bash
# Use diagnostic tool
find_port_user.bat

# Force close Arduino IDE
close_arduino_ide.bat
```

**Issue: Port 8000 already in use**

**Solution:**
```bash
# Use stop script
stop_server.bat

# OR find and kill process manually
netstat -ano | findstr :8000
taskkill /F /PID <PID>
```

### Recipe Issues

**Issue: Same ingredients appear multiple times in recipe**

**Cause:** Oils with good role_weights across multiple notes (e.g., Lavender works in top AND middle)

**Solution:** The system automatically consolidates duplicates before dispensing. Check console output for "CONSOLIDATED RECIPE" section.

**Issue: Disliked oils still appear in recipe**

**Cause:** Old issue - fixed in current version

**Solution:** Ensure you're using `logic_allocate_improved.py` (not old `logic_allocate.py`). Check console output shows "Auto-detected DISLIKES: [...]"

**Issue: All 15 oils used in every recipe with tiny amounts (0.40%, 0.73%)**

**Cause:** Settings too relaxed

**Solution:** Adjust quality controls in `logic_allocate_improved.py`:
```python
min_score = 0.12   # Increase for stricter selection
max_oils = 10      # Decrease for more focused recipes
min_percent = 1.0  # Increase to avoid trace amounts
```

**Issue: Important oils (fixatives) missing from recipe**

**Cause:** Low score for user preferences

**Solution:** Essential oils (Frankincense, Sandalwood, Cedarwood, Lavender) are automatically included even with low scores. Check `ESSENTIAL_OILS` list in `logic_allocate_improved.py:13`.

**Issue: Serial communication error "No module named 'serial'"**

**Cause:** pyserial not installed

**Solution:**
```bash
pip install pyserial
```

**Issue: Expensive oils like Jasmine excluded**

**Cause:** Amount needed is below min_percent threshold

**Solution:** Precious oils (Jasmine, Rose, Ylang-Ylang, Clove, Vanilla) can be used at 0.5%+ instead of 1.0%. Check `PRECIOUS_OILS` list in `logic_allocate_improved.py:22`.

## Data Files

### oils.json
Oil catalog with properties:
```json
{
  "name": "Bergamot (expressed)",
  "density_g_ml": 0.88,
  "max_pct": 0.40,
  "role_weights": [0.9, 0.1, 0.0],
  "scent_family": {
    "citrus": 0.95,
    "fresh": 0.75,
    ...
  }
}
```

### valve_map.json
Maps oils to dispenser valve positions:
```json
{
  "Bergamot (expressed)": 0,
  "Lemon": 1,
  "Lavender": 2,
  ...
}
```

### valve_calibration.json
Flow rate data (grams/second) for each valve+oil combination.

## Extending the System

### Add New Oils
1. Edit `oils.json` - Add oil with properties
2. Edit `valve_map.json` - Assign valve ID
3. Run `calibration.py` - Calibrate flow rate

### Modify Safety Mechanisms
Edit `logic_allocate_improved.py`:
- `ESSENTIAL_OILS` (line 13) - Add fixatives that should always be considered
- `PRECIOUS_OILS` (line 22) - Add oils that work well at <1%

### Integrate LLM
The system is designed to work with LLMs for natural language processing:
- `ai_engine.py` - AI integration layer
- `rag_agent.py` - RAG-based assistant
- Current NLP uses regex patterns (no API keys needed)

## Technical Details

- **Framework:** FastAPI
- **Algorithm:** Deterministic rule-based (no external APIs required)
- **Safety:** IFRA-compliant cap enforcement
- **Hardware:** Serial communication (9600 baud, 2s timeout)
- **Precision:** Cap-safe normalization to exactly 100.00%

## License

[Add your license here]

## Credits

Perfume Agent - AI-powered fragrance formulation system
testing VS for git 
this is the final test for full work with a team 

