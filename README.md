# StrixEyE — AI-GPR Tunnel Detection Desktop App

*Eyes in the sky · Intelligence on the ground*

A Streamlit-based desktop/web application around the research pipeline in
`gpr_tunnel_detection_research.ipynb` for AI-based underground tunnel
detection from Ground-Penetrating Radar (GPR) B-scans, branded with the
StrixEyE dark theme (`app/assets/`, `.streamlit/config.toml`).

## Features

- **Physics-based synthetic GPR simulator** for full drone surveys.
- **Classical preprocessing** chain (dewow, time-zero, background removal,
  SEC/AGC gain).
- **Compact U-Net** segmentation of tunnel hyperbolas.
- **Hyperbola inversion** for tunnel position, depth, and soil-velocity
  estimation.
- **Probability heat-map fusion** over parallel survey lines.
- **Interactive 3-D digital twin** (Plotly) with quantitative axis errors.
- **GPU/CPU device selection** with session-wide application.
- **Step-by-step guided workflow** with theory explanations.
- **StrixEyE branded UI**: dark theme, logo banner, pipeline status chips,
  metric cards, and JSON/CSV report downloads.
- **English / Français UI**: full in-app translation (sidebar selector),
  covering tabs, buttons, messages, tutorial, theory panels, and plots.
- **Stale-result protection**: changing any scene/survey parameter
  automatically invalidates previously computed results.
- **Headless smoke test** (`scripts/smoke_test_app.py`) based on
  `streamlit.testing.AppTest`.

## Quick start (Windows PowerShell)

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate it
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Train the U-Net (saves a versioned checkpoint, e.g. models/unet_v1.pt)
python -m scripts.train

# 5. Launch the app
streamlit run app/app.py
```

Then open the URL shown in the terminal (usually http://localhost:8501).

## Project structure

```
.
├── src/              # Refactored notebook pipeline
│   ├── config.py
│   ├── simulator.py
│   ├── preprocess.py
│   ├── model.py
│   ├── localize.py
│   ├── fusion.py
│   └── twin.py
├── app/              # Streamlit UI
│   ├── app.py
│   ├── components.py
│   ├── state.py
│   ├── i18n.py       # English/French translation layer
│   └── assets/       # StrixEyE logo + app icon
├── scripts/
│   ├── train.py      # Versioned checkpoint training
│   └── smoke_test_app.py  # Headless UI smoke test
├── models/           # Saved checkpoints
├── outputs/          # Exported JSON/CSV/PNG reports
├── requirements.txt
└── README.md
```

## Training

```powershell
python -m scripts.train --device cuda:0 --epochs 8 --batch-size 8
```

If no GPU is available, pass `--device cpu`. The script auto-detects CUDA
and saves checkpoints as `models/unet_v1.pt`, `models/unet_v2.pt`, etc.

## App workflow

1. **Scene & trajectory setup** – define tunnels and flight pattern.
2. **B-scan generation** – simulate one radargram per flight line.
3. **Preprocessing** – visualise the classical conditioning chain.
4. **AI detection** – U-Net probability maps and segmentation overlay.
5. **Localization** – hyperbola fit → depth / velocity / position.
6. **Heat map** – 2-D plan-view fusion with threshold slider.
7. **3-D digital twin** – interactive ground truth vs reconstruction.
8. **Summary report** – metrics + JSON/CSV/PNG export.

## Deployment (Streamlit Community Cloud)

The repo is ready for a one-click deploy:

1. Push to GitHub (already done if you read this online).
2. On [share.streamlit.io](https://share.streamlit.io), choose
   **New app → existing repo**, pick `strixeye-gpr`, main file
   `app/app.py`.
3. Deploy — `requirements.txt` pins CPU-only PyTorch and the trained
   checkpoint in `models/` is committed, so no build step is needed.

Private repos work on the free tier (1 private app).

## Testing

```powershell
python scripts/smoke_test_app.py
```

Loads the app headlessly, runs the full pipeline, verifies all results
land in session state, checks that lines without tunnel crossings stay
true negatives, confirms config changes invalidate stale results, and
verifies the French translation end-to-end (render + full pipeline).

## Notes

- The U-Net input is fixed at 128×128 samples, matching the research
  notebook geometry (10.24 m line length, 0.08 m trace spacing).
- The digital-twin radius is a user prior because a single hyperbola
  constrains the cylinder top but not the void radius.
- All computations run locally; no cloud or external data is used.

## License

Research-seed codebase provided as-is for academic and educational use.
