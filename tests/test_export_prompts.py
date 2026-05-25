from __future__ import annotations

import json

from scripts.export_prompts import export_prompts


def test_export_prompts_uses_latest_epoch_and_rank(tmp_path) -> None:
    snapshots = tmp_path / "prompt_populations"
    snapshots.mkdir()
    (snapshots / "generator_epoch_001.json").write_text(
        json.dumps(
            [
                {"rank": 1, "prompt": "old generator", "prompt_hash": "old-g"},
            ]
        ),
        encoding="utf-8",
    )
    (snapshots / "generator_epoch_002.json").write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "prompt": "best generator",
                    "prompt_hash": "new-g",
                    "reward": 0.8,
                    "avg_benchmark": 0.7,
                    "task_ids": ["task/0"],
                    "frontier_task_ids": ["task/0"],
                },
                {"rank": 2, "prompt": "second generator", "prompt_hash": "second-g"},
            ]
        ),
        encoding="utf-8",
    )
    (snapshots / "distinguisher_epoch_002.json").write_text(
        json.dumps(
            [
                {
                    "rank": 1,
                    "prompt": "best distinguisher",
                    "prompt_hash": "new-d",
                    "reward": 0.9,
                    "avg_benchmark": 0.75,
                    "task_ids": ["task/0"],
                    "frontier_task_ids": ["task/0"],
                }
            ]
        ),
        encoding="utf-8",
    )

    result = export_prompts(tmp_path)

    assert (tmp_path / "exported_prompts" / "generator.md").read_text(
        encoding="utf-8"
    ) == "best generator\n"
    assert (tmp_path / "exported_prompts" / "distinguisher.md").read_text(
        encoding="utf-8"
    ) == "best distinguisher\n"
    manifest = json.loads(
        (tmp_path / "exported_prompts" / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["epoch"] == 2
    assert manifest["generator"]["prompt_hash"] == "new-g"
    assert manifest["generator"]["frontier_task_ids"] == ["task/0"]
    assert result["generator_path"].endswith("generator.md")


def test_export_prompts_can_select_rank(tmp_path) -> None:
    snapshots = tmp_path / "prompt_populations"
    snapshots.mkdir()
    (snapshots / "generator_epoch_001.json").write_text(
        json.dumps(
            [
                {"rank": 1, "prompt": "first generator"},
                {"rank": 2, "prompt": "second generator"},
            ]
        ),
        encoding="utf-8",
    )
    (snapshots / "distinguisher_epoch_001.json").write_text(
        json.dumps(
            [
                {"rank": 1, "prompt": "first distinguisher"},
                {"rank": 2, "prompt": "second distinguisher"},
            ]
        ),
        encoding="utf-8",
    )

    export_prompts(tmp_path, output_dir=tmp_path / "chosen", rank=2)

    assert (tmp_path / "chosen" / "generator.md").read_text(
        encoding="utf-8"
    ) == "second generator\n"
    assert (tmp_path / "chosen" / "distinguisher.md").read_text(
        encoding="utf-8"
    ) == "second distinguisher\n"


def test_export_prompts_can_select_different_role_ranks(tmp_path) -> None:
    snapshots = tmp_path / "prompt_populations"
    snapshots.mkdir()
    (snapshots / "generator_epoch_001.json").write_text(
        json.dumps(
            [
                {"rank": 1, "prompt": "first generator"},
                {"rank": 2, "prompt": "second generator"},
            ]
        ),
        encoding="utf-8",
    )
    (snapshots / "distinguisher_epoch_001.json").write_text(
        json.dumps(
            [
                {"rank": 1, "prompt": "first distinguisher"},
                {"rank": 2, "prompt": "second distinguisher"},
            ]
        ),
        encoding="utf-8",
    )

    export_prompts(
        tmp_path,
        output_dir=tmp_path / "mixed",
        generator_rank=2,
        distinguisher_rank=1,
    )
    manifest = json.loads((tmp_path / "mixed" / "manifest.json").read_text())

    assert (tmp_path / "mixed" / "generator.md").read_text(
        encoding="utf-8"
    ) == "second generator\n"
    assert (tmp_path / "mixed" / "distinguisher.md").read_text(
        encoding="utf-8"
    ) == "first distinguisher\n"
    assert manifest["generator_rank"] == 2
    assert manifest["distinguisher_rank"] == 1
