from __future__ import annotations

import json

from evovideo_skill.external_benchmarks import build_storybench_suite, build_vbench_suite


def test_vbench_expands_official_sampling_policy(tmp_path):
    info = tmp_path / "VBench_full_info.json"
    info.write_text(
        json.dumps(
            [
                {"prompt_en": "a stable subject", "dimension": ["subject_consistency"]},
                {"prompt_en": "a static scene", "dimension": ["temporal_flickering"]},
            ]
        ),
        encoding="utf-8",
    )
    suite = build_vbench_suite(
        info,
        dimensions=["subject_consistency", "temporal_flickering"],
        samples_per_prompt=2,
        temporal_flickering_samples=3,
        seed_offset=10,
    )
    assert suite["metadata"]["prompt_count"] == 2
    assert suite["metadata"]["sample_count"] == 5
    filenames = [task["metadata"]["official_filename"] for task in suite["tasks"]]
    assert "a stable subject-1.mp4" in filenames
    assert "a static scene-2.mp4" in filenames
    assert {task["metadata"]["split"] for task in suite["tasks"]} == {"test"}


def test_vbench_limited_subset_is_hash_selected_per_dimension(tmp_path):
    info = tmp_path / "VBench_full_info.json"
    info.write_text(
        json.dumps(
            [
                {"prompt_en": f"subject {index}", "dimension": ["subject_consistency"]}
                for index in range(10)
            ]
            + [
                {"prompt_en": f"action {index}", "dimension": ["human_action"]}
                for index in range(10)
            ]
        ),
        encoding="utf-8",
    )
    suite = build_vbench_suite(
        info,
        dimensions=["subject_consistency", "human_action"],
        samples_per_prompt=1,
        temporal_flickering_samples=1,
        limit_prompts_per_dimension=3,
        seed_offset=17,
    )
    assert len(suite["tasks"]) == 6
    assert suite["metadata"]["selection"] == "deterministic_hash_per_dimension"
    assert {task["metadata"]["vbench_dimension"] for task in suite["tasks"]} == {
        "subject_consistency",
        "human_action",
    }


def test_storybench_story_gen_preserves_steps_and_exports_four_samples(tmp_path):
    tasks = tmp_path / "story_gen.json"
    tasks.write_text(
        json.dumps(
            [
                {
                    "texts": ["A person opens a box.", "The person removes a red cup."],
                    "exact_frames_per_prompt": [16, 24],
                    "background": "A quiet kitchen.",
                    "storybench_mode": "story_gen",
                    "comment": "example-a",
                }
            ]
        ),
        encoding="utf-8",
    )
    suite = build_storybench_suite(tasks, samples_per_prompt=4, seed_offset=20)
    assert suite["metadata"]["example_count"] == 1
    assert len(suite["tasks"]) == 4
    assert suite["tasks"][0]["duration_seconds"] == 5
    assert suite["tasks"][0]["metadata"]["temporal_steps"] == [
        "A person opens a box.",
        "The person removes a red cup.",
    ]
    assert suite["tasks"][-1]["metadata"]["official_filename"] == "fn3.npz"


def test_storybench_limited_subset_is_step_duration_stratified(tmp_path):
    tasks = tmp_path / "story_gen.json"
    payload = []
    for step_count in (1, 2, 3, 4):
        for total_frames in (24, 64, 120):
            payload.append(
                {
                    "texts": [f"step {index}" for index in range(step_count)],
                    "exact_frames_per_prompt": [total_frames // step_count] * step_count,
                    "storybench_mode": "story_gen",
                    "comment": f"steps-{step_count}-frames-{total_frames}",
                }
            )
    tasks.write_text(json.dumps(payload), encoding="utf-8")
    suite = build_storybench_suite(tasks, samples_per_prompt=1, limit=12, seed_offset=9)
    assert len(suite["tasks"]) == 12
    assert suite["metadata"]["selection"] == "deterministic_step_duration_stratified"
    assert len(suite["metadata"]["selected_strata"]) == 12
