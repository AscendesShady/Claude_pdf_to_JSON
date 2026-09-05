# PDF to QLoRA JSONL

Converts PDF books/manuals into instruction-tuning JSONL datasets (ChatML + Alpaca)
for QLoRA fine-tuning, using a local Ollama model to generate grounded Q&A pairs.

## Setup on a new machine

1. **Python dependencies** (this repo, versioned in git):
   ```
   pip install -r requirements.txt
   ```

2. **Ollama** (a separate application, not a Python package - install once per machine):
   - Install from [ollama.com](https://ollama.com) and make sure the Ollama service is running.
   - Models are *not* stored in this repo and never get committed to git - they're
     multi-gigabyte files Ollama keeps in its own local store, entirely separate from
     this codebase. This app never touches that storage directly; it only talks to
     Ollama's local REST API (`http://localhost:11434`), so it works the same on any
     machine as long as Ollama is running there.
   - Pull at least one model before running the app: `ollama pull llama3.1:8b`
     (or whichever model you want).

3. **Run it**:
   ```
   python main.py
   ```
   The model dropdown always reflects what's actually installed on the current
   machine (refreshed on launch, or via the Refresh button) - the app never assumes
   a fixed model list. Want to try a model that isn't in the dropdown? Pull it via
   `ollama pull <name>` in a terminal, then hit Refresh (or restart the app).

## Notes

- `output/` (generated JSONL + `review.xlsx`) and `*.pdf` source books are gitignored -
  they're per-machine artifacts/inputs, not part of the codebase.
- See the in-app log panel for what the pipeline is doing at each stage; `review.xlsx`
  is for manually marking generated pairs Keep/Reject/Needs Fix before trusting them
  for training.
