"""Exact button-label action tables for ViZDoom scenarios."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

ActionTable: TypeAlias = Sequence[Sequence[str]]
RESERVED_ACTION_SET_NAMES = frozenset(
    {"all", "filtered", "discrete", "multi_discrete"}
)


@dataclass(frozen=True)
class CustomActionSpec:
    preset: str | None
    table: tuple[tuple[str, ...], ...]
    meanings: tuple[str, ...]
    masks: tuple[tuple[int, ...], ...]
    table_hash: str


def minimal_action_table(buttons: Sequence[str]) -> tuple[tuple[str, ...], ...]:
    """Return noop plus one action per available binary button."""
    return ((), *((button,) for button in buttons))


def normalize_action_table(
    table: Any,
    *,
    buttons: Sequence[str],
    context: str = "action table",
) -> CustomActionSpec:
    if isinstance(table, (str, bytes, bytearray)) or not isinstance(table, Sequence):
        raise ValueError(f"{context} must be a non-empty list of actions")
    if not table:
        raise ValueError(f"{context} must contain at least one action")
    button_to_index = {name: index for index, name in enumerate(buttons)}
    normalized: list[tuple[str, ...]] = []
    meanings: list[str] = []
    masks: list[tuple[int, ...]] = []
    seen_masks: set[int] = set()
    for action_index, raw_action in enumerate(table):
        if isinstance(raw_action, (str, bytes, bytearray)) or not isinstance(
            raw_action, Sequence
        ):
            raise ValueError(f"{context} action {action_index} must be a list of button labels")
        labels: list[str] = []
        mask = 0
        for label in raw_action:
            if not isinstance(label, str):
                raise ValueError(f"{context} action {action_index} labels must be strings")
            if label in labels:
                raise ValueError(
                    f"{context} action {action_index} contains duplicate button {label!r}"
                )
            try:
                button_index = button_to_index[label]
            except KeyError as exc:
                valid = ", ".join(repr(name) for name in button_to_index)
                raise ValueError(
                    f"{context} action {action_index} contains unknown button {label!r}; "
                    f"valid labels: {valid}"
                ) from exc
            labels.append(label)
            mask |= 1 << button_index
        if mask in seen_masks:
            raise ValueError(f"{context} action {action_index} duplicates an earlier action")
        normalized.append(tuple(labels))
        meanings.append("noop" if not labels else "_".join(label.lower() for label in labels))
        masks.append((mask,))
        seen_masks.add(mask)
    payload = json.dumps(masks, separators=(",", ":"), ensure_ascii=True)
    return CustomActionSpec(
        preset=None,
        table=tuple(normalized),
        meanings=tuple(meanings),
        masks=tuple(masks),
        table_hash=hashlib.sha256(payload.encode("ascii")).hexdigest(),
    )


def resolve_custom_action(
    value: Any,
    *,
    buttons: Sequence[str],
) -> CustomActionSpec:
    preset: str | None = None
    table = value
    if isinstance(value, str):
        name = value.strip().casefold()
        if name not in {"minimal", "discrete"}:
            valid = ", ".join(sorted(RESERVED_ACTION_SET_NAMES | {"minimal"}))
            raise ValueError(
                f"unknown use_restricted_actions value {value!r}; valid values: {valid}"
            )
        preset = "minimal"
        table = minimal_action_table(buttons)
    resolved = normalize_action_table(
        table,
        buttons=buttons,
        context=f"action set {preset!r}" if preset else "action table",
    )
    return CustomActionSpec(
        preset=preset,
        table=resolved.table,
        meanings=resolved.meanings,
        masks=resolved.masks,
        table_hash=resolved.table_hash,
    )


__all__ = [
    "ActionTable",
    "CustomActionSpec",
    "RESERVED_ACTION_SET_NAMES",
    "minimal_action_table",
    "normalize_action_table",
    "resolve_custom_action",
]
