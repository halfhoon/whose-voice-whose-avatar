from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from tqdm import tqdm

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from inference.config import (
    AUDIO_DIR,
    CONDITIONS,
    IMAGE_DIR,
    IMAGE_PAIRS,
    PROMPT_DIR,
    RESULTS_DIR,
    SUPPORTED_MODELS,
    TASK_TO_CONDITIONS,
    USER_PROMPT_FILE,
    VOICE_ORDER,
    scenario_ids_for_condition,
    scenario_records_by_id,
    voice_gender_map,
)

if load_dotenv:
    load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

PROMPT_CACHE: dict[str, str] = {}
SCENARIOS = scenario_records_by_id()
VOICE_GENDER = voice_gender_map()
USER_PROMPT_TEMPLATE = USER_PROMPT_FILE.read_text(encoding="utf-8")


class QuotaExhaustedException(Exception):
    pass


def get_system_prompt(scenario_id: str) -> str:
    genre = SCENARIOS[scenario_id]["genre"]
    if genre not in PROMPT_CACHE:
        PROMPT_CACHE[genre] = (PROMPT_DIR / f"{genre}.txt").read_text(encoding="utf-8").strip()
    return PROMPT_CACHE[genre]


def get_user_prompt(scenario_id: str) -> str:
    return USER_PROMPT_TEMPLATE.format(situation=SCENARIOS[scenario_id]["situation"]).strip()


def parse_json_output(response_text: str) -> dict:
    import re

    cleaned = response_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    if cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
        if "follow" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    fixed_patterns = [
        (r'"([^"]*)","\s*\}', r'"\1"}'),
        (r'"\s*,\s*\}', r'"}'),
        (r'""\s*\}', r'"}'),
        (r'(\{"follow"\s*:\s*"[^"]*"\s*,\s*"reason"\s*:\s*"[^"]*)\}', r'\1"}'),
        (r'(\{"follow"\s*:\s*"[^"]*"\s*,\s*"reason"\s*:\s*"[^"}]*$)', r'\1"}'),
        (r"'", r'"'),
        (r'\{""\s*', r'{"'),
    ]

    for pattern, replacement in fixed_patterns:
        fixed = re.sub(pattern, replacement, response_text)
        if fixed == response_text:
            continue
        try:
            parsed = json.loads(fixed)
            if "follow" in parsed and "reason" in parsed:
                return parsed
        except json.JSONDecodeError:
            pass

    json_pattern = r'\{[^{}]*"follow"\s*:\s*"[^"]*"[^{}]*"reason"\s*:\s*"[^"]*"[^{}]*\}'
    for match in re.findall(json_pattern, response_text, re.DOTALL):
        try:
            parsed = json.loads(match)
            if "follow" in parsed and "reason" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue

    fixed = cleaned.replace("'", '"')
    try:
        parsed = json.loads(fixed)
        if "follow" in parsed:
            return parsed
    except json.JSONDecodeError:
        pass

    logging.warning("Failed to parse JSON response: %s", response_text[:500])
    return {"follow": "error", "reason": response_text}


def resolve_output_file(model_name: str, condition_name: str) -> Path:
    model_dir = RESULTS_DIR / model_name.replace("/", "_")
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir / CONDITIONS[condition_name].output_filename


def load_completed_trials(output_file: Path) -> set[tuple]:
    completed: set[tuple] = set()
    if not output_file.exists():
        return completed

    with output_file.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("status") != "ok":
                continue
            completed.add(
                (
                    row["scenario"],
                    row["voice"],
                    row["avatar_1_gender"],
                    int(row["avatar_1_id"]),
                    row["avatar_2_gender"],
                    int(row["avatar_2_id"]),
                )
            )
    return completed


def resolve_audio_path(scenario_id: str, voice: str) -> Path:
    return AUDIO_DIR / f"{scenario_id.replace('-', '_')}_{voice}.wav"


def resolve_image_path(style: str, gender: str, image_id: int) -> Path:
    return IMAGE_DIR / style / f"{gender}_{image_id:02d}.png"


def mime_type_for_path(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".png":
        return "image/png"
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".wav":
        return "audio/wav"
    if suffix == ".mp3":
        return "audio/mpeg"
    return "application/octet-stream"


def is_retryable_error(error: Exception) -> bool:
    lowered = str(error).lower()
    retryable = ["quota", "rate limit", "429", "500", "502", "503", "504", "timeout", "connection", "network", "parse error"]
    return any(token in lowered for token in retryable)


def build_client(model_name: str, batch_size: int, device: str):
    concurrency_limit = 1

    if model_name.startswith("gemini-"):
        from inference.gemini import GeminiClient

        client = GeminiClient(model_name=model_name)
        concurrency_limit = batch_size
    elif model_name == "qwen-2.5-omni-7b":
        from inference.qwen_2_5_omni import Qwen25OmniClient

        client = Qwen25OmniClient(model_id="Qwen/Qwen2.5-Omni-7B", device=device)
        client.load_model()
    elif model_name == "interactiveomni-8b":
        from inference.interactive_omni import InteractiveOmniClient

        client = InteractiveOmniClient(device=device)
        client.load_model()
    elif model_name == "gemma-3n-e4b-it":
        from inference.gemma import Gemma3nClient

        client = Gemma3nClient(model_id="google/gemma-3n-E4B-it", device=device)
        client.load_model()
    elif model_name == "gemma-3n-e2b-it":
        from inference.gemma import Gemma3nClient

        client = Gemma3nClient(model_id="google/gemma-3n-E2B-it", device=device)
        client.load_model()
    elif model_name == "minicpm-o-2.6":
        from inference.minicpm import MiniCPMClient

        client = MiniCPMClient(model_id="openbmb/MiniCPM-o-2_6", device=device)
        client.load_model()
    elif model_name == "phi-4-multimodal":
        from inference.phi4 import Phi4Client

        client = Phi4Client(device=device)
        client.load_model()
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return client, concurrency_limit


async def generate_with_gemini(
    client,
    system_prompt: str,
    user_prompt: str,
    audio_path: Path,
    image_1_path: Path,
    image_2_path: Path,
) -> str:
    from google.genai import types

    contents_parts = ["[Avatar A]"]
    with image_1_path.open("rb") as f:
        contents_parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type_for_path(image_1_path)))

    contents_parts.append("\n\n[Avatar B]")
    with image_2_path.open("rb") as f:
        contents_parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type_for_path(image_2_path)))

    contents_parts.append("\n\n[Suggestion]")
    with audio_path.open("rb") as f:
        contents_parts.append(types.Part.from_bytes(data=f.read(), mime_type=mime_type_for_path(audio_path)))

    contents_parts.append(f"\n\n{user_prompt}")

    config = types.GenerateContentConfig(system_instruction=system_prompt)
    response = await client.client.aio.models.generate_content(
        model=client.model_name,
        contents=contents_parts,
        config=config,
    )

    texts: list[str] = []
    if response.candidates:
        for candidate in response.candidates:
            if candidate.content and candidate.content.parts:
                for part in candidate.content.parts:
                    if hasattr(part, "text") and part.text:
                        texts.append(part.text)
    return "\n".join(texts).strip()


async def run_single_trial(
    client,
    model_name: str,
    condition_name: str,
    voice: str,
    avatar_1_gender: str,
    avatar_1_id: int,
    avatar_2_gender: str,
    avatar_2_id: int,
    scenario_id: str,
    trial_id: int,
) -> dict:
    spec = CONDITIONS[condition_name]
    audio_path = resolve_audio_path(scenario_id, voice)
    image_1_path = resolve_image_path(spec.style, avatar_1_gender, avatar_1_id)
    image_2_path = resolve_image_path(spec.style, avatar_2_gender, avatar_2_id)

    order = f"{avatar_1_gender[0].upper()}{avatar_2_gender[0].upper()}"
    voice_gender = VOICE_GENDER.get(voice)

    if not audio_path.exists() or not image_1_path.exists() or not image_2_path.exists():
        return {
            "trial_id": trial_id,
            "timestamp": datetime.now().isoformat(),
            "task": spec.task,
            "style": spec.style,
            "scenario": scenario_id,
            "voice": voice,
            "voice_gender": voice_gender,
            "avatar_1_gender": avatar_1_gender,
            "avatar_1_id": avatar_1_id,
            "avatar_2_gender": avatar_2_gender,
            "avatar_2_id": avatar_2_id,
            "order": order,
            "follow": "file_missing",
            "follow_avatar": "file_missing",
            "follow_gender": "file_missing",
            "reason": "File not found",
            "status": "file_missing",
            "raw_response": "File not found",
        }

    system_prompt = get_system_prompt(scenario_id)
    user_prompt = get_user_prompt(scenario_id)
    text_prompt = (
        "[Avatar A]\n(See Image 1)\n\n"
        "[Avatar B]\n(See Image 2)\n\n"
        "[Suggestion]\n(Listen to audio)\n\n"
        f"{user_prompt}"
    )

    last_error: Exception | None = None
    max_retries = 5

    for attempt in range(max_retries):
        try:
            if model_name.startswith("gemini-"):
                response_text = await generate_with_gemini(
                    client=client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    audio_path=audio_path,
                    image_1_path=image_1_path,
                    image_2_path=image_2_path,
                )
            else:
                response_text = await client.generate_async(
                    prompt=text_prompt,
                    audio_path=str(audio_path),
                    image_path=str(image_1_path),
                    image_path_2=str(image_2_path),
                    system_prompt=system_prompt,
                )

            parsed = parse_json_output(response_text)
            follow = parsed.get("follow", "error")
            reason = parsed.get("reason", "")

            if follow == "A":
                follow = "avatar_1"
                follow_avatar = f"{avatar_1_gender}_{avatar_1_id}"
                follow_gender = avatar_1_gender
            elif follow == "B":
                follow = "avatar_2"
                follow_avatar = f"{avatar_2_gender}_{avatar_2_id}"
                follow_gender = avatar_2_gender
            elif follow == "avatar_1":
                follow_avatar = f"{avatar_1_gender}_{avatar_1_id}"
                follow_gender = avatar_1_gender
            elif follow == "avatar_2":
                follow_avatar = f"{avatar_2_gender}_{avatar_2_id}"
                follow_gender = avatar_2_gender
            else:
                follow_avatar = "error"
                follow_gender = "error"

            if "quota" in response_text.lower() or "rate limit" in response_text.lower():
                raise Exception(f"Quota exceeded: {response_text[:200]}")
            if follow == "error" or follow_avatar == "error":
                raise Exception(f"Parse error - invalid response format: {response_text[:200]}")

            return {
                "trial_id": trial_id,
                "timestamp": datetime.now().isoformat(),
                "task": spec.task,
                "style": spec.style,
                "scenario": scenario_id,
                "voice": voice,
                "voice_gender": voice_gender,
                "avatar_1_gender": avatar_1_gender,
                "avatar_1_id": avatar_1_id,
                "avatar_2_gender": avatar_2_gender,
                "avatar_2_id": avatar_2_id,
                "order": order,
                "follow": follow,
                "follow_avatar": follow_avatar,
                "follow_gender": follow_gender,
                "reason": reason,
                "status": "ok",
                "raw_response": response_text,
            }
        except Exception as error:
            last_error = error
            if attempt < max_retries - 1 and is_retryable_error(error):
                wait_time = 2 ** attempt
                logging.warning(
                    "Trial %s attempt %s/%s failed: %s. Retrying in %ss",
                    trial_id,
                    attempt + 1,
                    max_retries,
                    error,
                    wait_time,
                )
                await asyncio.sleep(wait_time)
            else:
                break

    lowered = str(last_error).lower() if last_error else "error"
    status = "quota_exceeded" if "quota" in lowered or "rate limit" in lowered else "error"
    logging.error("Trial %s failed after retries: %s", trial_id, last_error)
    return {
        "trial_id": trial_id,
        "timestamp": datetime.now().isoformat(),
        "task": spec.task,
        "style": spec.style,
        "scenario": scenario_id,
        "voice": voice,
        "voice_gender": voice_gender,
        "avatar_1_gender": avatar_1_gender,
        "avatar_1_id": avatar_1_id,
        "avatar_2_gender": avatar_2_gender,
        "avatar_2_id": avatar_2_id,
        "order": order,
        "follow": "error",
        "follow_avatar": "error",
        "follow_gender": "error",
        "reason": str(last_error),
        "status": status,
        "raw_response": str(last_error),
    }


async def run_condition(model_name: str, condition_name: str, batch_size: int = 10, device: str = "cuda") -> Path:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"Unsupported model: {model_name}")

    spec = CONDITIONS[condition_name]
    output_file = resolve_output_file(model_name, condition_name)
    completed = load_completed_trials(output_file)
    scenario_ids = scenario_ids_for_condition(condition_name)

    all_trials = []
    trial_id = 0
    for scenario_id in scenario_ids:
        for voice in VOICE_ORDER:
            for male_id, female_id in IMAGE_PAIRS:
                key_mf = (scenario_id, voice, "male", male_id, "female", female_id)
                if key_mf not in completed:
                    all_trials.append({
                        "trial_id": trial_id,
                        "scenario_id": scenario_id,
                        "voice": voice,
                        "avatar_1_gender": "male",
                        "avatar_1_id": male_id,
                        "avatar_2_gender": "female",
                        "avatar_2_id": female_id,
                    })
                trial_id += 1

                key_fm = (scenario_id, voice, "female", female_id, "male", male_id)
                if key_fm not in completed:
                    all_trials.append({
                        "trial_id": trial_id,
                        "scenario_id": scenario_id,
                        "voice": voice,
                        "avatar_1_gender": "female",
                        "avatar_1_id": female_id,
                        "avatar_2_gender": "male",
                        "avatar_2_id": male_id,
                    })
                trial_id += 1

    logging.info("Condition %s / model %s / total %s / completed %s / remaining %s", condition_name, model_name, trial_id, len(completed), len(all_trials))

    if not all_trials:
        logging.info("All trials already completed for %s / %s", model_name, condition_name)
        return output_file

    client, concurrency_limit = build_client(model_name, batch_size, device)

    file_exists = output_file.exists()
    quota_error_count = 0
    quota_threshold = 10
    fieldnames = [
        "trial_id",
        "timestamp",
        "task",
        "style",
        "scenario",
        "voice",
        "voice_gender",
        "avatar_1_gender",
        "avatar_1_id",
        "avatar_2_gender",
        "avatar_2_id",
        "order",
        "follow",
        "follow_avatar",
        "follow_gender",
        "reason",
        "status",
        "raw_response",
    ]

    with tqdm(total=len(all_trials), desc=f"{model_name}:{condition_name}") as pbar:
        for offset in range(0, len(all_trials), concurrency_limit):
            batch = all_trials[offset:offset + concurrency_limit]
            batch_results = await asyncio.gather(*[
                run_single_trial(
                    client=client,
                    model_name=model_name,
                    condition_name=condition_name,
                    voice=trial["voice"],
                    avatar_1_gender=trial["avatar_1_gender"],
                    avatar_1_id=trial["avatar_1_id"],
                    avatar_2_gender=trial["avatar_2_gender"],
                    avatar_2_id=trial["avatar_2_id"],
                    scenario_id=trial["scenario_id"],
                    trial_id=trial["trial_id"],
                )
                for trial in batch
            ])

            batch_quota_errors = sum(1 for row in batch_results if row["status"] == "quota_exceeded")
            quota_error_count = quota_error_count + batch_quota_errors if batch_quota_errors else 0

            with output_file.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                    file_exists = True
                writer.writerows(batch_results)

            pbar.update(len(batch))

            if quota_error_count >= quota_threshold:
                raise QuotaExhaustedException(
                    f"Quota exhausted after {quota_error_count} consecutive quota errors"
                )

            await asyncio.sleep(1)

    logging.info("Finished %s / %s", model_name, condition_name)
    return output_file


async def run_selected_task(model_name: str, task_name: str, batch_size: int = 10, device: str = "cuda") -> None:
    for condition_name in TASK_TO_CONDITIONS[task_name]:
        await run_condition(
            model_name=model_name,
            condition_name=condition_name,
            batch_size=batch_size,
            device=device,
        )
