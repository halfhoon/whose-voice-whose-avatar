# Whose Voice, Whose Avatar?

Code and data for the ACL Findings 2026 paper:

**Whose Voice, Whose Avatar? Gender Matching Bias in Multimodal AI Teammates**

Paper: https://aclanthology.org/2026.findings-acl.2057.pdf

This repository contains the released scenarios, audio stimuli, avatar images, inference scripts, and analysis code used for the paper's multimodal teammate-selection experiments.

## Repository Layout

```text
whose-voice-whose-avatar/
├── data/
│   ├── scenarios.csv
│   ├── voices.csv
│   ├── audio/
│   ├── images/
│   │   ├── photorealistic/
│   │   ├── stylized/
│   │   └── pixel_art/
│   └── prompts/
├── inference/
├── analysis/
└── results/
```

## Data

`data/scenarios.csv` lists the released game scenarios.

Key columns:

- `scenario_id`: public scenario identifier, e.g. `BR-01`
- `genre`: `BR`, `SD`, `SV`, `CS`, or `CZ`
- `stake`: `high` or `low`
- `use_task1`: whether the scenario is used in Task 1
- `use_task2`: whether the scenario is used in Task 2
- `situation`: game context shown to the model
- `proposition`: teammate suggestion paired with the audio clip

`data/voices.csv` contains the 10 voice identities and gender labels used in the experiments.

`data/audio/` contains the generated speech clips, named by scenario and voice:

```text
BR_01_adam.wav
SD_08_hope1.wav
```

`data/images/` contains 8 male-presenting and 8 female-presenting avatars for each visual style:

```text
male_01.png ... male_08.png
female_01.png ... female_08.png
```

## Experimental Tasks

The release follows the paper's task split.

- `task1`: photorealistic avatars only
- `task2`: stylized and pixel-art avatars
- `all`: Task 1 and Task 2 together

Condition counts:

- `task1`: `35 scenarios x 10 voices x 16 avatar pairs x 2 orders = 11,200` trials
- `task2`: `10 scenarios x 10 voices x 16 avatar pairs x 2 orders x 2 styles = 6,400` trials
- `all`: `17,600` trials

Task 2 uses the same scenario subset for the stylized and pixel-art conditions. Scenario overlap between Task 1 and Task 2 is encoded directly in `data/scenarios.csv`.

## Models

The scripts include wrappers for the models evaluated in the paper.

API models:

- `gemini-2.5-pro`
- `gemini-3-flash-preview`
- `gemini-2.5-flash`
- `gemini-2.5-flash-lite`

Local/open-weight models:

- `qwen-2.5-omni-7b`
- `phi-4-multimodal`
- `minicpm-o-2.6`
- `gemma-3n-e4b-it`
- `gemma-3n-e2b-it`
- `interactiveomni-8b`

The wrappers do not set custom temperature values for the API models. Local models may require checkpoint access, GPU memory, and model-specific dependencies beyond the minimal requirements listed here.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Set API keys in `.env` only for the backends you plan to run.

Common variables:

- `GEMINI_API_KEY` or `GOOGLE_API_KEY`
- `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN` for gated Hugging Face checkpoints

## Running Inference

Run commands from the repository root.

API model example:

```bash
python inference/run_api_model.py --model gemini-2.5-pro --task task1
python inference/run_api_model.py --model gemini-2.5-pro --task task2
python inference/run_api_model.py --model gemini-2.5-pro --task all
```

Local/open-weight model example:

```bash
python inference/run_local_model.py --model qwen-2.5-omni-7b --task task1 --device cuda
python inference/run_local_model.py --model gemma-3n-e4b-it --task task2 --device cuda
python inference/run_local_model.py --model interactiveomni-8b --task all --device cuda:0
```

Outputs are written under `results/<model>/`.

Expected files:

```text
results/<model>/
├── task1_photorealistic.csv
├── task2_stylized.csv
└── task2_pixel_art.csv
```

Each row is one trial. Main columns include:

- `scenario`
- `voice`
- `voice_gender`
- `avatar_1_gender`, `avatar_1_id`
- `avatar_2_gender`, `avatar_2_id`
- `order`
- `follow`
- `follow_gender`
- `reason`
- `status`

## Analysis

```bash
python analysis/run_analysis.py --model gemini-2.5-pro
python analysis/run_analysis.py --model qwen-2.5-omni-7b --no-r
python analysis/run_analysis.py --model all
```

The analysis code produces:

- preprocessed trial-level CSV files,
- descriptive statistics,
- generated R scripts for GLMM analyses,
- GLMM summary JSON files when R is available,
- cross-style comparison outputs when all visual styles are present.

The implemented analyses follow the paper's workflow:

- voice-based selection,
- voice-avatar matching tendency,
- stake-level moderation,
- position effects,
- cross-style comparison.

## Citation

If you use this repository, please cite the paper linked above. For questions, open a GitHub issue or contact the authors through the information listed in the ACL Anthology paper.
