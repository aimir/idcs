from __future__ import annotations

from datetime import datetime

from idcs import telemetry


class _FixedDatetime:
    @classmethod
    def now(cls, tz):  # type: ignore[no-untyped-def]
        return datetime(2026, 5, 25, 1, 30, 0, tzinfo=tz)


def test_create_run_dir_uses_suffix_when_timestamp_exists(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(telemetry, "datetime", _FixedDatetime)

    first = telemetry.create_run_dir(tmp_path)
    second = telemetry.create_run_dir(tmp_path)

    assert first.name == "20260525T013000Z"
    assert second.name == "20260525T013000Z-1"
