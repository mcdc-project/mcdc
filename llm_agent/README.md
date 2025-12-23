# MCDC-Onboarding

> [!WARNING]
> This agent is still in testing. Generated scripts may require manual review and corrections. Please report any issues or unexpected behavior!

An interactive AI agent that guides you through building Monte Carlo particle transport simulations using the [MCDC](https://github.com/CEMeNT-PSAAP/MCDC) Python package. Perfect for beginners learning nuclear engineering concepts!

## What It Does

MCDC-Agent walks you through an 8-step workflow to create complete MCDC simulations:

1. **Materials** – Define what your geometry is made of (water, uranium, steel, etc.)
2. **Surfaces** – Create geometric boundaries (planes, spheres, cylinders)
3. **Cells** – Build regions of space filled with materials
4. **Hierarchy** – Group cells into larger regions
5. **Source** – Specify where particles start and their energy
6. **Tally** – Set up detectors to measure flux, reactions, and more
7. **Settings** – Configure simulation parameters (particles, batches)
8. **Run** – Generate a working Python script with `mcdc.run()`

The agent uses Google's Gemini AI to answer questions, show relevant examples from the MCDC documentation, and help you write correct code.

---

## Setup

### 1. Prerequisites

- Python 3.8 or higher
- pip package manager
- A Google account (for Gemini API key)

### 2. Install Dependencies

From the **project root directory** (`MCDC/`):

```bash
pip install -r llm_agent/requirements.txt
pip install -e .
```

Key dependencies:

- `langchain` – AI agent framework
- `langchain-google-genai` – Gemini integration
- `langchain-chroma` – Vector database for documentation
- `pandas` – Nuclear data processing

### 3. Build the Knowledge Base

Run these three scripts **in order** to prepare the documentation and vector index:

```bash
# Step 1: Extract API docs from MCDC source code
python -m llm_agent.generate_docs

# Step 2: Scrape RTD docs and extract example scripts
python -m llm_agent.scrape_api

# Step 3: Build the Chroma vectorstore from all sources
python -m llm_agent.build_index
```

> **Note:** The vectorstore (`llm_agent/vectorstore/`) is gitignored and must be built locally.

### 4. Set Your Gemini API Key

1. Go to [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Click **"Create API key"**
4. Choose **"Create API key in new project"** (or an existing project)
5. Copy the API key (it starts with `AIza...`)

Set it as an environment variable:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

> **Never** commit your API key to version control.
Base configuration uses gemini-3-flash-preview model. If you encounter rate limits, use model="gemini-2.5-flash" or "gemini-2-flash". You can change it in `llm_agent/utils.py` on line 23.

---

## Usage

### Interactive Mode (Default)

Run the agent for a guided, step-by-step experience:

```bash
python -m llm_agent.onboarding.agent
```

The interactive session will guide you through building a simulation step by step.

### Batch Execution Mode

Generate a complete script from a text prompt without interaction:

1. **Create an input prompt file** (e.g., `my_prompt.txt`):

   ```
   Create a k-eigenvalue simulation of a 10cm sphere of uranium-235 
   surrounded by 5cm of water moderator. Use 10000 particles per batch 
   and 50 active batches. Tally the flux on a 10x10x10 mesh.
   ```

2. **Run in batch mode:**

   ```bash
   python -m llm_agent.onboarding.agent --batch --input_file my_prompt.txt --output my_simulation.py
   ```

#### Batch Mode Options

| Flag | Description |
|------|-------------|
| `--batch` | Enable batch mode (required) |
| `--input_file PATH` | Path to text file containing the simulation prompt |
| `--output PATH` | Output filename (default: `mcdc_input.py`) |
| `--dry-run` | Preview the decomposed plan without executing |
| `--no-validate` | Disable dry-run validation on the generated script |
| `--verbose` | Enable debug logging |

#### Example: Preview Plan Only

```bash
python -m llm_agent.onboarding.agent --batch --input_file my_prompt.txt --dry-run
```

---

## Project Structure

```
llm_agent/
├── generate_docs.py      # Extract API from MCDC source
├── scrape_api.py         # Scrape RTD + regression tests
├── build_index.py        # Build Chroma vectorstore
├── utils.py              # Shared utilities (LLM, retriever)
├── onboarding/
│   ├── agent.py          # Main interactive agent
│   ├── batch_executor.py # Automated batch execution
│   ├── decomposer.py     # LLM-based prompt decomposition
│   ├── tools.py          # Agent tools (write_code, search, etc.)
│   ├── script_builder.py # Script state management
│   ├── validator.py      # Dry-run validation
│   ├── debugger.py       # Script debugging handler
│   └── ...
├── scraped_docs/         # Generated docs & examples
└── vectorstore/          # Chroma DB (gitignored)
```

---

## Feedback

Any feedback is very helpful for improving the agent!
