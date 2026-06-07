from __future__ import annotations

import json
from argparse import Namespace
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping


def _utc_now() -> datetime:
    return datetime.now(UTC)


def _to_iso_z(value: datetime) -> str:
    return value.astimezone(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return _to_iso_z(value)
    if isinstance(value, Namespace):
        return {key: _normalize_value(raw) for key, raw in vars(value).items()}
    if isinstance(value, dict):
        return {str(key): _normalize_value(raw) for key, raw in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]
    return value


def make_run_id(stage: str, now: datetime | None = None) -> str:
    timestamp = (now or _utc_now()).astimezone(UTC).strftime("%Y%m%d_%H%M%S")
    return f"{stage}_{timestamp}"


@dataclass
class RunContext:
    stage: str
    output_dir: Path
    run_id: str
    started_at: datetime
    parameters: dict[str, Any]
    inputs: dict[str, Any] = field(default_factory=dict)
    label: str = ""
    notes: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def _write_json(self, filename: str, payload: Mapping[str, Any]) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / filename
        path.write_text(json.dumps(_normalize_value(dict(payload)), indent=2, ensure_ascii=False), encoding="utf-8")

    def _write_with_stage_variant(self, filename: str, payload: Mapping[str, Any]) -> None:
        self._write_json(filename, payload)
        stage_filename = self._stage_variant_filename(filename)
        self._write_json(stage_filename, payload)

    def _stage_variant_filename(self, filename: str) -> str:
        path = Path(filename)
        suffix = "".join(path.suffixes)
        stem = path.name[: -len(suffix)] if suffix else path.name
        return f"{stem}.{self.stage}{suffix}"

    def write_initial_metadata(self) -> None:
        self._write_with_stage_variant(
            "_run.json",
            {
                "version": 1,
                "stage": self.stage,
                "run_id": self.run_id,
                "started_at": _to_iso_z(self.started_at),
                "output_dir": self.output_dir,
                "label": self.label,
                "notes": self.notes,
                "parameters": self.parameters,
                "extra": self.extra,
            },
        )
        self._write_with_stage_variant("_inputs.json", self.inputs)

    def write_validation(self, payload: Mapping[str, Any]) -> None:
        self._write_with_stage_variant("_validation.json", payload)

    def write_artifacts(self, payload: Mapping[str, Any]) -> None:
        self._write_with_stage_variant("_artifacts.json", payload)

    def write_summary(self, payload: Mapping[str, Any]) -> None:
        self._write_with_stage_variant("_summary.json", payload)


def create_run_context(
    stage: str,
    output_dir: Path | str,
    args: Namespace | Mapping[str, Any] | None = None,
    inputs: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    run_label: str = "",
    notes: str = "",
    extra: Mapping[str, Any] | None = None,
    now: datetime | None = None,
) -> RunContext:
    started_at = now or _utc_now()
    output_path = Path(output_dir).expanduser().resolve()
    normalized_args = _normalize_value(args or {})
    if not isinstance(normalized_args, dict):
        raise TypeError("args must normalize to a dictionary")

    return RunContext(
        stage=stage,
        output_dir=output_path,
        run_id=run_id or make_run_id(stage, started_at),
        started_at=started_at,
        parameters=normalized_args,
        inputs=dict(_normalize_value(inputs or {})),
        label=run_label,
        notes=notes,
        extra=dict(_normalize_value(extra or {})),
    )
