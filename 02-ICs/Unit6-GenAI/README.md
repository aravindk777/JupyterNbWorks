# Unit6-GenAI — Report Summary Demo

This small Flask app demonstrates a polished report summary page (`templates/summary.html`) and an ad-generator UI (`templates/index.html`). I added a demo route `/report` that renders the new summary template with sample data and a Chart.js example.

Quick setup

1. Create a virtual environment (recommended) and install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

If you don't have CUDA 12.4 or want a CPU-only setup, use the CPU helper file and install PyTorch using the CPU index as shown below:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
# Install PyTorch CPU packages explicitly (example):
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

Run the app (development)

```powershell
# Activate venv first
python app.py
```

Open in your browser:
- Main UI: http://127.0.0.1:5000/
- Report demo: http://127.0.0.1:5000/report

Run tests

```powershell
python -m unittest discover -s tests -p "test_*.py" -v
```

What I changed
- `templates/summary.html`: New polished, Bootstrap-based summary template (Jinja-ready).
- `app.py`: Added `/report` demo route that renders `summary.html` with sample data and Chart.js config.
- `README.md`: Usage and run instructions.
- `tests/test_summary_template.py`: Simple unit tests that validate the template exists and contains key sections.
- `requirements-cpu.txt`: CPU-friendly dependency helper and installation notes.

Next steps (optional)
- Add unit tests for the Flask routes.
- Add environment-aware requirements and a `requirements-dev.txt` with pinned CPU-friendly packages.
- Wire a real model-run to populate `summary.html` dynamically after a run completes.

If you'd like, I can: add a small test file (`tests/test_app.py`) for route smoke tests, or replace the CUDA-specific torch wheels in `requirements.txt` with CPU-friendly pins. Which would you prefer?
