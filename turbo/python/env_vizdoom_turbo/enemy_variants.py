from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

DEFEND_LINE_PLUS_GAME = "VizdoomDefendLine-Plus-v1"
DEFEND_LINE_PLUS_ALIAS = "defend_the_line_plus"
BASIC_PLUS_GAME = "VizdoomBasic-Plus-v1"
BASIC_PLUS_ALIAS = "basic_plus"
_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "enemy_variants"
_PLUS_SCENARIOS = {
    BASIC_PLUS_ALIAS: (BASIC_PLUS_GAME, _ASSET_ROOT / "basic" / "catalog.json"),
    DEFEND_LINE_PLUS_ALIAS: (
        DEFEND_LINE_PLUS_GAME,
        _ASSET_ROOT / "defend_the_line" / "catalog.json",
    ),
}
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9-]*$")
_SAFE_ROLE = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_CVAR = re.compile(r"^[a-z_][a-z0-9_]*$")
_SAFE_SPRITE = re.compile(r"^[A-Z0-9]{4}$")


@dataclass(frozen=True)
class EnemyVariant:
    role: str
    selector_cvar: str
    variant_id: str
    decorate_id: str
    scenario_index: int
    actor: str
    sprite: str | None
    frames: tuple[tuple[str, Path, str], ...]


def plus_scenario_alias(value: str | Path | None) -> str | None:
    normalized = str(value or "").strip().casefold().removesuffix(".cfg")
    for alias, (game, _catalog_path) in _PLUS_SCENARIOS.items():
        if normalized in {game.casefold(), alias}:
            return alias
    return None


def is_defend_line_plus(value: str | Path | None) -> bool:
    return plus_scenario_alias(value) == DEFEND_LINE_PLUS_ALIAS


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_path(alias: str) -> Path:
    try:
        return _PLUS_SCENARIOS[alias][1]
    except KeyError as exc:
        raise ValueError(f"unknown Plus scenario alias: {alias!r}") from exc


def _catalog_document(alias: str) -> dict[str, Any]:
    catalog_path = _catalog_path(alias)
    try:
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid enemy-variant catalog: {catalog_path}") from exc
    if document.get("schema_version") != 2:
        raise RuntimeError("enemy-variant catalog must use schema_version 2")
    return document


def _catalog_asset(alias: str, relative_path: str, label: str) -> Path:
    path = (_catalog_path(alias).parent / relative_path).resolve()
    root = _ASSET_ROOT.resolve()
    if root not in path.parents:
        raise RuntimeError(f"{label} escapes the enemy-variant asset directory")
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    return path


def plus_scenario(alias: str) -> tuple[Path, str]:
    document = _catalog_document(alias)
    raw = document.get("scenario")
    if not isinstance(raw, dict):
        raise RuntimeError("enemy-variant catalog must declare its scenario")
    config_path = _catalog_asset(alias, str(raw.get("config") or ""), "Plus config")
    wad_path = _catalog_asset(alias, str(raw.get("wad") or ""), "Plus WAD")
    expected_hash = str(raw.get("wad_sha256") or "")
    if _sha256(wad_path) != expected_hash:
        raise RuntimeError(f"Plus scenario WAD failed integrity check: {wad_path}")
    return config_path, expected_hash


def defend_line_plus_scenario() -> tuple[Path, str]:
    return plus_scenario(DEFEND_LINE_PLUS_ALIAS)


def _manifest_frames(
    raw: Mapping[str, Any],
    *,
    alias: str,
    variant_id: str,
    sprite: str | None,
) -> tuple[tuple[str, Path, str], ...]:
    raw_manifest = raw.get("manifest")
    if sprite is None:
        if raw_manifest is not None:
            raise RuntimeError("original enemy variants must not declare a manifest")
        return ()
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise RuntimeError(f"enemy variant {variant_id!r} must declare a manifest")
    manifest_path = _catalog_asset(alias, raw_manifest, f"{variant_id} manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid enemy-variant manifest: {manifest_path}") from exc
    if manifest.get("id") != variant_id or manifest.get("sprite") != sprite:
        raise RuntimeError(f"enemy-variant manifest identity mismatch: {manifest_path}")
    raw_frames = manifest.get("frames")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise RuntimeError(f"enemy variant {variant_id!r} manifest has no frames")
    frames: list[tuple[str, Path, str]] = []
    for frame in raw_frames:
        if not isinstance(frame, dict):
            raise RuntimeError(f"invalid frame for enemy variant {variant_id!r}")
        lump = str(frame.get("lump") or "")
        relative_patch = str(frame.get("patch") or "")
        expected_hash = str(frame.get("patch_sha256") or "")
        if len(lump) > 8 or not lump.isascii() or not lump.isalnum():
            raise RuntimeError(f"invalid WAD lump name: {lump!r}")
        path = (manifest_path.parent / relative_patch).resolve()
        if manifest_path.parent not in path.parents:
            raise RuntimeError("enemy-variant frame escapes its manifest directory")
        if not path.is_file() or _sha256(path) != expected_hash:
            raise RuntimeError(f"enemy-variant frame failed integrity check: {path}")
        frames.append((lump, path, expected_hash))
    return tuple(frames)


def load_enemy_catalog(
    alias: str,
) -> tuple[Mapping[str, tuple[EnemyVariant, ...]], str]:
    document = _catalog_document(alias)
    raw_roles = document.get("roles")
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise RuntimeError("enemy-variant catalog must contain roles")
    resolved_roles: dict[str, tuple[EnemyVariant, ...]] = {}
    for role, raw_role in raw_roles.items():
        if not isinstance(role, str) or not _SAFE_ROLE.fullmatch(role):
            raise RuntimeError(f"invalid enemy role: {role!r}")
        if not isinstance(raw_role, dict):
            raise RuntimeError(f"enemy role {role!r} must be an object")
        selector_cvar = str(raw_role.get("selector_cvar") or "")
        if not _SAFE_CVAR.fullmatch(selector_cvar):
            raise RuntimeError(f"invalid selector cvar for enemy role {role!r}")
        raw_variants = raw_role.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise RuntimeError(f"enemy role {role!r} must contain variants")
        variants: list[EnemyVariant] = []
        seen_ids: set[str] = set()
        seen_indices: set[int] = set()
        for raw in raw_variants:
            if not isinstance(raw, dict):
                raise RuntimeError("enemy-variant catalog entries must be objects")
            variant_id = str(raw.get("id") or "")
            decorate_id = str(raw.get("decorate_id") or "")
            actor = str(raw.get("actor") or "")
            scenario_index = raw.get("scenario_index")
            if not _SAFE_IDENTIFIER.fullmatch(variant_id):
                raise RuntimeError(f"invalid enemy variant id: {variant_id!r}")
            if not decorate_id.isidentifier() or not actor.isidentifier():
                raise RuntimeError(f"invalid actor identifiers for {variant_id!r}")
            if (
                isinstance(scenario_index, bool)
                or not isinstance(scenario_index, int)
                or scenario_index < 0
            ):
                raise RuntimeError(f"invalid scenario index for {variant_id!r}")
            if variant_id in seen_ids or scenario_index in seen_indices:
                raise RuntimeError(
                    f"enemy role {role!r} has duplicate ids or scenario indices"
                )
            seen_ids.add(variant_id)
            seen_indices.add(scenario_index)
            raw_sprite = raw.get("sprite")
            sprite = None if raw_sprite is None else str(raw_sprite)
            if sprite is not None and not _SAFE_SPRITE.fullmatch(sprite):
                raise RuntimeError(f"invalid sprite prefix for {variant_id!r}")
            variants.append(
                EnemyVariant(
                    role=role,
                    selector_cvar=selector_cvar,
                    variant_id=variant_id,
                    decorate_id=decorate_id,
                    scenario_index=scenario_index,
                    actor=actor,
                    sprite=sprite,
                    frames=_manifest_frames(
                        raw,
                        alias=alias,
                        variant_id=variant_id,
                        sprite=sprite,
                    ),
                )
            )
        defaults = raw_role.get("default_variants")
        if (
            not isinstance(defaults, list)
            or not defaults
            or any(value not in seen_ids for value in defaults)
        ):
            raise RuntimeError(f"enemy role {role!r} has invalid default variants")
        resolved_roles[role] = tuple(variants)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return (
        MappingProxyType(resolved_roles),
        hashlib.sha256(canonical).hexdigest(),
    )


def load_defend_line_catalog(
) -> tuple[Mapping[str, tuple[EnemyVariant, ...]], str]:
    return load_enemy_catalog(DEFEND_LINE_PLUS_ALIAS)


def _requested_ids(
    role: str,
    value: Sequence[str] | None,
    defaults: Sequence[str],
) -> tuple[str, ...]:
    if value is None:
        raw_ids = tuple(str(item) for item in defaults)
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError(f"enemy_variants[{role!r}] must be a sequence of ids")
        raw_ids = tuple(str(item).strip() for item in value)
    if not raw_ids:
        raise ValueError(f"enemy_variants[{role!r}] must select at least one variant")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError(f"enemy_variants[{role!r}] cannot contain duplicates")
    return raw_ids


def resolve_enemy_variants(
    alias: str,
    requested: Mapping[str, Sequence[str]] | Sequence[str] | None,
) -> tuple[Mapping[str, tuple[EnemyVariant, ...]], str]:
    catalog, catalog_hash = load_enemy_catalog(alias)
    document = _catalog_document(alias)
    document_roles = document["roles"]
    environment = (
        "Defend the Line"
        if alias == DEFEND_LINE_PLUS_ALIAS
        else str(document.get("environment") or "Plus environment")
    )
    if requested is None:
        requested_by_role: Mapping[str, Sequence[str]] = {}
    elif isinstance(requested, Mapping):
        unknown_roles = sorted(set(requested) - set(catalog))
        if unknown_roles:
            raise ValueError(f"unknown {environment} enemy role(s): {unknown_roles}")
        requested_by_role = requested
    elif isinstance(requested, Sequence) and not isinstance(
        requested, (str, bytes, bytearray)
    ):
        shorthand_role = "shooter" if "shooter" in catalog else next(iter(catalog))
        requested_by_role = {shorthand_role: requested}
    else:
        raise TypeError("enemy_variants must be a role mapping or sequence of ids")

    selected: dict[str, tuple[EnemyVariant, ...]] = {}
    for role, variants in catalog.items():
        by_id = {variant.variant_id: variant for variant in variants}
        raw_ids = _requested_ids(
            role,
            requested_by_role.get(role),
            document_roles[role]["default_variants"],
        )
        unknown = [variant_id for variant_id in raw_ids if variant_id not in by_id]
        if unknown:
            choices = ", ".join(by_id)
            raise ValueError(
                f"unknown {role} variant(s): {unknown}; choose from {choices}"
            )
        selected[role] = tuple(by_id[variant_id] for variant_id in raw_ids)
    return MappingProxyType(selected), catalog_hash


def resolve_defend_line_variants(
    requested: Mapping[str, Sequence[str]] | Sequence[str] | None,
) -> tuple[Mapping[str, tuple[EnemyVariant, ...]], str]:
    return resolve_enemy_variants(DEFEND_LINE_PLUS_ALIAS, requested)
