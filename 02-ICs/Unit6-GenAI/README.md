# Unit6-GenAI — Incremental Capstone Project

This small Flask app demonstrates the GenAI models for generating the Ad text based on the prompts and parameters. Has a nice ad-generator UI (`templates/index.html`) along with every model's execution and performance report (`templates/model_report.html`).

Quick setup (clone + run)

Follow these steps to clone the repository and run the app locally (PowerShell shown):

1. Clone the repo and change into the example folder

```powershell
# Clone the repository (this will create a local folder `JupyterNbWorks`)
git clone https://github.com/aravindk777/JupyterNbWorks.git
# Change to the Unit6-GenAI example directory
cd .\JupyterNbWorks\02-ICs\Unit6-GenAI
```

2. Create & activate a virtual environment, then install dependencies

```powershell
# Create a venv and activate (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install packages
pip install -r requirements.txt
```

If you don't have CUDA 12.4 (or a compatible GPU) or prefer CPU-only, use the CPU helper and install PyTorch CPU wheels as shown below instead:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
# Install PyTorch CPU packages explicitly (example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

3. Run the Flask app (development)

```powershell
# From the project folder (Unit6-GenAI)
# Start the app with flask runner (recommended for development)
python -m flask --app .\app.py run --port 5000 --debug

# Or (alternative):
# python app.py
```

4. Open in your browser

- Main UI: http://127.0.0.1:5000/
- Model performance report: http://127.0.0.1:5000/report
- My final Summary: http://127.0.0.1:5000/summary

5. Run the unit tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

6. Notes & troubleshooting

- The `requirements.txt` includes CUDA wheels for PyTorch (e.g., `torch==2.10.0+cu130`). If you don't have the matching CUDA runtime, prefer `requirements-cpu.txt` or install CPU wheels directly as shown above.
- Some models in `MODELS_LIST` are large and require GPUs and lots of RAM; for quick local testing use the lighter models included by default (e.g., `EleutherAI/gpt-neo-125M`, `distilgpt2`, `gpt2`). You can change the selected model from the UI dropdown.
- The app writes a report JSON to `reports/last_report.json` after a generation finishes; the `/report` route reads that file (if present) and renders the dynamic `model_report.html` or falls back to sample data.
