# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

The notebooks and slides for **AI Fundamentals for Healthcare**, a four-session course for
**clinical and health-system professionals**. Every notebook pairs an ML concept with a
clinical problem that genuinely suits it, and covers theory, implementation, and
interpretation.

This repository is a slimmed-down copy of the `healthcare` branch of `cair-ml-ai-course`,
keeping only what the four sessions use.

## The data rule

**No real or identifiable patient data, ever.** Every patient in this course is
simulated. Data comes from one of three places:

1. A `generate_*()` function defined inline in the notebook
2. A shared generator script at the repository root (see below)
3. A public dataset bundled with scikit-learn (`load_breast_cancer`, `load_diabetes`, …)

Do not add a dataset that requires credentialed access (MIMIC, eICU) or that contains
real patient records. Every notebook that generates patient data must say in markdown
that the data is simulated.

Clinical values must be physiologically plausible, and clipped to plausible ranges:

| Measure | Range |
|---|---|
| Heart rate | 40–180 bpm |
| SpO2 | 70–100 % |
| Temperature | 34–41.5 °C |
| Systolic BP | 60–210 mmHg |
| Respiratory rate | 6–45 /min |
| WBC | 1–40 ×10⁹/L |
| Creatinine | 25–710 µmol/L |

### Units: Canadian (SI)

The audience is Canadian, so every lab value uses the SI units a Canadian lab report shows.
Never use US conventional units (mg/dL, g/dL, BUN) in labels, column names, generated values,
or prose.

| Test | Unit | Typical adult range | From US units |
|---|---|---|---|
| Creatinine | µmol/L | 50–110 | mg/dL × 88.4 |
| Urea (not BUN) | mmol/L | 2.5–7.5 | BUN mg/dL × 0.357 |
| Glucose | mmol/L | 3.9–5.5 fasting; diabetes ≥ 7.0 | mg/dL ÷ 18 |
| Hemoglobin | g/L | 120–160 | g/dL × 10 |
| Albumin | g/L | 35–50 | g/dL × 10 |
| Bilirubin | µmol/L | < 20 | mg/dL × 17.1 |
| Cholesterol, LDL, HDL | mmol/L | | mg/dL ÷ 38.67 |
| Sodium, potassium, lactate | mmol/L | | unchanged |
| Free T4 / free T3 | pmol/L | 9–25 / 3.1–6.8 | |
| HbA1c | % | < 6.0 | Diabetes Canada reports % |
| Temperature, weight, height | °C, kg, cm | | |

When a published equation is defined in US units (CKD-EPI eGFR uses mg/dL), take the input
in SI and convert inside the function. The eGFR calculator in `s1_01_intro_explicit_vs_learning`
shows the pattern. When converting an existing generator, scale its noise and slope
coefficients by the same factor so the simulated relationships are unchanged.

Public datasets bundled with scikit-learn (`load_diabetes`) keep their own scaling. Do not
describe their features in US units.

### Terminology and spelling: Canadian

Use the terms a Canadian clinician or health-system leader would use, not US ones.

| Use | Not |
|---|---|
| long-term care (LTC) | skilled nursing facility (SNF), nursing home |
| home care | home health |
| family physician | primary care physician (PCP) |
| emergency department (ED) | emergency room (ER) |
| provincial health plan, publicly funded | insurance, payer, Medicare/Medicaid, co-pay |
| diagnosis codes (ICD-10-CA), CIHI case mix groups | billing codes, ICD-10-CM, DRGs |
| Health Canada medical device licence | FDA clearance (mention FDA only as "elsewhere") |
| PHIPA / provincial privacy law | HIPAA |
| costs in CAD (`*_cad`) | USD |

Spelling follows Canadian (CMAJ) style:
- `-ize` endings: visualize, normalize, utilization, recognize. `-our`/`-re` stay: colour, behaviour, centre, litre, tumour.
- Medical words take the simplified spelling: hemoglobin, hemodynamic, anemia, hyperglycemia, pediatrics, anesthetic, edema.
- "program", not "programme". "practise" (verb) and "licence" (noun) are correct Canadian.

Do not rename library arguments or identifiers that other code depends on just to fix spelling.

The direction of effect must be clinically correct: deterioration means heart rate up,
blood pressure down, SpO2 down, respiratory rate up, lactate up. Class prevalence should
reflect a realistic case mix, not a balanced sample. Most patients are not septic.

## Development Environment

**Python**: 3.12 (pinned in `pyproject.toml` as `>=3.12,<3.13`)
**Package manager**: `uv`

### Environment setup

⚠️ **The environment lives outside the repository, at `~/.venvs/ai-fundamentals-healthcare`.**
It must not live under the OneDrive checkout: OneDrive de-materializes the binaries and
symlinks, which silently corrupts the venv. A `.nosync` suffix does *not* prevent this.

To create or update it:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/ai-fundamentals-healthcare uv sync
```

For day-to-day work, call the interpreter directly — no environment variable needed:

```bash
~/.venvs/ai-fundamentals-healthcare/bin/python -m jupyter lab
~/.venvs/ai-fundamentals-healthcare/bin/python some_script.py
```

`uv run` also works, but only with the variable set in the same command:

```bash
UV_PROJECT_ENVIRONMENT=~/.venvs/ai-fundamentals-healthcare uv run jupyter lab
```

Do **not** export `UV_PROJECT_ENVIRONMENT` globally in your shell profile — every other
uv project on the machine would then share this one environment.

The stray `.venv` *file* in the repository root is deliberate. It makes a bare `uv sync`
fail loudly rather than quietly building a new environment inside OneDrive. Leave it.

### Executing a notebook

```bash
cd notebooks/<session_folder> && ~/.venvs/ai-fundamentals-healthcare/bin/python -m nbconvert --execute \
    --to notebook --inplace --ExecutePreprocessor.timeout=600 <notebook>.ipynb
```

Run from the notebook's own folder: relative paths (`../../data`, `sys.path.append('../..')`)
assume it.

Deep learning notebooks need `--ExecutePreprocessor.timeout=1800`.

Note: the `timeout` shell command is **not** available on this machine (macOS without
coreutils). Rely on nbconvert's own timeout.

## Code Architecture

### Core Dependencies
- **Data**: NumPy, Pandas
- **Plotting**: Matplotlib, Seaborn, Plotly
- **Machine learning**: scikit-learn, XGBoost
- **Deep learning**: PyTorch, torchvision
- **Interpretation**: SHAP
- **Interactive**: Jupyter Lab, ipywidgets
- **Trees**: Graphviz

Do not add new major dependencies without justification.

### Repository Structure

```
├── notebooks/                          # All educational content, grouped by session
│   ├── session_1_how_machines_learn/   # Programming vs learning, regression, gradient descent
│   ├── session_2_classical_ml/         # Data quality, trees to XGBoost, SVM, PCA, evaluation, SHAP
│   ├── session_3_deep_learning/        # RNN/LSTM, vitals forecasting, CNN, ViT, quantization
│   ├── session_4_models_to_agents/     # No notebooks yet; README points to supporting ones
│   └── extras/                         # Streaming from a simulated ward feed
├── slides/                             # The four session decks (.pptx)
├── data/encounters/                    # Synthetic hospital encounters extract
├── synthetic_vitals_generator.py       # Bedside-monitor trajectories, clinical archetypes
├── bedside_monitor_simulator.py        # Simulated MQTT ward feed (mirrors paho-mqtt API, no paho needed)
├── generate_encounters_dataset.py      # Builds the encounters CSV, defects included
├── pyproject.toml
└── uv.lock
```

The session folders follow the slide decks in `slides/`. A new notebook goes in the folder for
the session whose deck covers its concept. Notebooks sit two levels below the repository root,
so shared files are reached with `../../` (`../../data/...`, `sys.path.append('../..')`). A
link to a notebook in another session uses `../<session_folder>/<notebook>.ipynb`.

### Shared generators

Most notebooks define their data generator **inline**, because notebooks must be
self-contained and runnable independently. The exceptions are deliberate:

- `synthetic_vitals_generator.py` — used by the two trajectory-forecasting notebooks,
  which need identical multi-channel data
- `bedside_monitor_simulator.py` — used by the streaming notebook
- `generate_encounters_dataset.py` — builds a committed CSV, run once

## Notebook Architecture

Each notebook follows the same educational shape:

- Theoretical foundations
- Mathematical intuition, visualized
- Hands-on implementation on a clinical problem
- Interactive parameter exploration with ipywidgets
- Model interpretation, with SHAP where it applies
- Comparison against other algorithms
- Best practices and honest limitations

## Development Guidelines

### When adding content

1. Follow the pattern: theory → implementation → interpretation → limitations
2. Use the existing dependency stack
3. Include interactive widgets where parameter exploration teaches something
4. Include interpretation, not just accuracy
5. State the clinical caveat: these are teaching examples, not validated tools

### Editing notebooks

Notebooks are megabytes of JSON with embedded base64 images. **Do not read them with the
Read tool** and do not hand-edit the JSON. Load them with the `json` module, or use
`nbtool.py` if it is available in the session scratchpad.

After any edit, clear outputs and re-execute. Stored figure images carry the old labels,
so a text-only edit leaves stale visuals behind.

### Known issues

`s2_02_decision_trees_to_boosting` still has leftover marine content from before the
healthcare re-theme: "navigation safety" wording in sections 2–3, a fuel-savings calculation
in USD per liter in the gradient boosting section, and variables named `X_ship` / `X_vessel`.

To find code cells that will not parse:

```bash
~/.venvs/ai-fundamentals-healthcare/bin/python -c "
import json, glob, ast
for p in sorted(glob.glob('notebooks/*/*.ipynb')):
    nb = json.load(open(p))
    for i, c in enumerate(nb['cells']):
        if c['cell_type'] != 'code': continue
        s = ''.join(c['source'])
        if not s.strip() or s.lstrip()[:1] in ('%', '!'): continue
        try: ast.parse('\n'.join('pass' if l.lstrip()[:1] in ('%','!') else l for l in s.split('\n')))
        except SyntaxError as e: print(p, i, e.msg)
"
```

## Important Notes

- This is an **educational repository**: prioritize clarity over production optimization
- Interactive widgets require Jupyter Lab or Notebook
- Each notebook is self-contained and runnable independently
- Synthetic datasets are seeded, so results reproduce across machines
