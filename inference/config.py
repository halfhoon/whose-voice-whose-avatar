from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT_DIR / "data"
AUDIO_DIR = DATA_DIR / "audio"
IMAGE_DIR = DATA_DIR / "images"
PROMPT_DIR = DATA_DIR / "prompts" / "follow_choice"
USER_PROMPT_FILE = DATA_DIR / "prompts" / "user_prompt.txt"
SCENARIO_FILE = DATA_DIR / "scenarios.csv"
VOICE_FILE = DATA_DIR / "voices.csv"
RESULTS_DIR = ROOT_DIR / "results"

VOICE_ORDER = [
    "adam",
    "brad",
    "jeff",
    "josh",
    "titan",
    "ember",
    "eve",
    "hope1",
    "ivanna",
    "lily_wolff",
]

IMAGE_PAIRS = [
    (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7), (8, 8),
    (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 1),
]

API_MODELS = (
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
)

LOCAL_MODELS = (
    "qwen-2.5-omni-7b",
    "phi-4-multimodal",
    "minicpm-o-2.6",
    "gemma-3n-e4b-it",
    "gemma-3n-e2b-it",
    "interactiveomni-8b",
)

SUPPORTED_MODELS = API_MODELS + LOCAL_MODELS


@dataclass(frozen=True)
class ConditionSpec:
    name: str
    task: str
    style: str
    output_filename: str


CONDITIONS = {
    "photorealistic": ConditionSpec(
        name="photorealistic",
        task="task1",
        style="photorealistic",
        output_filename="task1_photorealistic.csv",
    ),
    "stylized": ConditionSpec(
        name="stylized",
        task="task2",
        style="stylized",
        output_filename="task2_stylized.csv",
    ),
    "pixel_art": ConditionSpec(
        name="pixel_art",
        task="task2",
        style="pixel_art",
        output_filename="task2_pixel_art.csv",
    ),
}

TASK_TO_CONDITIONS = {
    "task1": ("photorealistic",),
    "task2": ("stylized", "pixel_art"),
    "all": ("photorealistic", "stylized", "pixel_art"),
}


def load_scenarios() -> list[dict]:
    with SCENARIO_FILE.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_voices() -> list[dict]:
    with VOICE_FILE.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def voice_gender_map() -> dict[str, str]:
    return {row["voice_name"]: row["gender"] for row in load_voices()}


def scenario_records_by_id() -> dict[str, dict]:
    return {row["scenario_id"]: row for row in load_scenarios()}


def scenario_ids_for_condition(condition_name: str) -> list[str]:
    spec = CONDITIONS[condition_name]
    flag_col = "use_task1" if spec.task == "task1" else "use_task2"
    return [row["scenario_id"] for row in load_scenarios() if row[flag_col] == "1"]
