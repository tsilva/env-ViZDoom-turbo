from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

_ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "surface_variants"
_CATALOG_PATHS = {
    "basic_plus": _ASSET_ROOT / "basic" / "catalog.json",
    "defend_the_line_plus": _ASSET_ROOT / "defend_the_line" / "catalog.json",
}
_SAFE_IDENTIFIER = re.compile(r"^[a-z][a-z0-9-]*$")
_SAFE_ROLE = re.compile(r"^[a-z][a-z0-9_]*$")
_SAFE_CVAR = re.compile(r"^[a-z_][a-z0-9_]*$")
_SAFE_TEXTURE = re.compile(r"^[A-Z0-9_]{1,8}$")


@dataclass(frozen=True)
class SurfaceVariant:
    role: str
    selector_cvar: str
    variant_id: str
    scenario_index: int
    texture: str
    namespace: str
    theme: str | None
    asset: Path | None
    asset_sha256: str | None


@dataclass(frozen=True)
class TextureSetVariant:
    role: str
    selector_cvar: str
    variant_id: str
    scenario_index: int
    theme: str | None
    surfaces: Mapping[str, SurfaceVariant]


AppearanceVariant = SurfaceVariant | TextureSetVariant


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _catalog_path(alias: str) -> Path:
    try:
        return _CATALOG_PATHS[alias]
    except KeyError as exc:
        raise ValueError(f"unknown Plus scenario alias: {alias!r}") from exc


def _catalog_document(alias: str) -> dict[str, Any]:
    catalog_path = _catalog_path(alias)
    try:
        document = json.loads(catalog_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid surface-variant catalog: {catalog_path}") from exc
    if document.get("schema_version") not in {1, 2}:
        raise RuntimeError("surface-variant catalog must use schema_version 1 or 2")
    return document


def _catalog_asset(alias: str, relative_path: str, label: str) -> Path:
    path = (_catalog_path(alias).parent / relative_path).resolve()
    root = _ASSET_ROOT.resolve()
    if root not in path.parents:
        raise RuntimeError(f"{label} escapes the surface-variant asset directory")
    if not path.is_file():
        raise RuntimeError(f"missing {label}: {path}")
    return path


def _texture_wrap_is_compatible(
    manifest: Mapping[str, Any],
    raw_texture: Mapping[str, Any],
) -> bool:
    if manifest.get("schema_version") == 1:
        return (
            raw_texture.get("seamless_left_right") is True
            and raw_texture.get("seamless_top_bottom") is True
        )
    if manifest.get("schema_version") != 2:
        return False
    processing = manifest.get("processing")
    wrap = processing.get("wrap") if isinstance(processing, dict) else None
    final_ratios = wrap.get("final_ratios") if isinstance(wrap, dict) else None
    threshold = wrap.get("threshold") if isinstance(wrap, dict) else None
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, int | float)
        or not math.isfinite(threshold)
        or threshold < 0
        or not isinstance(final_ratios, dict)
    ):
        return False
    ratios = (final_ratios.get("x"), final_ratios.get("y"))
    return (
        raw_texture.get("wrap_x_within_threshold") is True
        and raw_texture.get("wrap_y_within_threshold") is True
        and all(
            not isinstance(ratio, bool)
            and isinstance(ratio, int | float)
            and math.isfinite(ratio)
            and 0 <= ratio <= threshold
            for ratio in ratios
        )
    )


def _manifest_asset(
    raw: Mapping[str, Any],
    *,
    alias: str,
    role: str,
    variant_id: str,
    namespace: str,
    texture: str,
    theme: str | None,
) -> tuple[Path | None, str | None]:
    raw_manifest = raw.get("manifest")
    if variant_id == "original":
        if raw_manifest is not None:
            raise RuntimeError("original surface variants must not declare a manifest")
        return None, None
    if not isinstance(raw_manifest, str) or not raw_manifest:
        raise RuntimeError(f"surface variant {variant_id!r} must declare a manifest")
    manifest_path = _catalog_asset(alias, raw_manifest, f"{variant_id} manifest")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid surface-variant manifest: {manifest_path}") from exc
    if (
        manifest.get("id") != variant_id
        or manifest.get("role") != role
        or manifest.get("namespace") != namespace
        or manifest.get("lump") != texture
        or manifest.get("theme") != theme
    ):
        raise RuntimeError(f"surface-variant manifest identity mismatch: {manifest_path}")
    raw_texture = manifest.get("texture")
    if not isinstance(raw_texture, dict):
        raise RuntimeError(f"surface variant {variant_id!r} has no texture metadata")
    relative_asset = str(raw_texture.get("png") or "")
    expected_hash = str(raw_texture.get("png_sha256") or "")
    asset = (manifest_path.parent / relative_asset).resolve()
    if manifest_path.parent not in asset.parents:
        raise RuntimeError("surface-variant texture escapes its manifest directory")
    if not asset.is_file() or _sha256(asset) != expected_hash:
        raise RuntimeError(f"surface-variant texture failed integrity check: {asset}")
    if (
        raw_texture.get("size") != [64, 64]
        or raw_texture.get("fully_opaque") is not True
        or raw_texture.get("colors_in_playpal") is not True
        or not _texture_wrap_is_compatible(manifest, raw_texture)
    ):
        raise RuntimeError(f"surface variant {variant_id!r} failed compatibility checks")
    return asset, expected_hash


def _surface_variant(
    alias: str,
    raw: Mapping[str, Any],
    *,
    role: str,
    selector_cvar: str,
) -> SurfaceVariant:
    variant_id = str(raw.get("id") or "")
    scenario_index = raw.get("scenario_index")
    texture = str(raw.get("texture") or "")
    namespace = str(raw.get("namespace") or "")
    raw_theme = raw.get("theme")
    theme = None if raw_theme is None else str(raw_theme)
    if not _SAFE_IDENTIFIER.fullmatch(variant_id):
        raise RuntimeError(f"invalid surface variant id: {variant_id!r}")
    if (
        isinstance(scenario_index, bool)
        or not isinstance(scenario_index, int)
        or scenario_index < 0
    ):
        raise RuntimeError(f"invalid scenario index for {variant_id!r}")
    if not _SAFE_TEXTURE.fullmatch(texture):
        raise RuntimeError(f"invalid texture name for {variant_id!r}")
    if namespace not in {"texture", "flat"}:
        raise RuntimeError(f"invalid namespace for {variant_id!r}")
    if theme is not None and not _SAFE_IDENTIFIER.fullmatch(theme):
        raise RuntimeError(f"invalid theme for {variant_id!r}")
    asset, asset_sha256 = _manifest_asset(
        raw,
        alias=alias,
        role=role,
        variant_id=variant_id,
        namespace=namespace,
        texture=texture,
        theme=theme,
    )
    return SurfaceVariant(
        role=role,
        selector_cvar=selector_cvar,
        variant_id=variant_id,
        scenario_index=scenario_index,
        texture=texture,
        namespace=namespace,
        theme=theme,
        asset=asset,
        asset_sha256=asset_sha256,
    )


def _texture_set_variant(
    alias: str,
    raw: Mapping[str, Any],
    *,
    role: str,
    selector_cvar: str,
) -> TextureSetVariant:
    variant_id = str(raw.get("id") or "")
    scenario_index = raw.get("scenario_index")
    raw_theme = raw.get("theme")
    theme = None if raw_theme is None else str(raw_theme)
    if not _SAFE_IDENTIFIER.fullmatch(variant_id):
        raise RuntimeError(f"invalid texture-set id: {variant_id!r}")
    if (
        isinstance(scenario_index, bool)
        or not isinstance(scenario_index, int)
        or scenario_index < 0
    ):
        raise RuntimeError(f"invalid scenario index for {variant_id!r}")
    if theme is not None and not _SAFE_IDENTIFIER.fullmatch(theme):
        raise RuntimeError(f"invalid theme for {variant_id!r}")
    raw_surfaces = raw.get("surfaces")
    if not isinstance(raw_surfaces, dict) or set(raw_surfaces) != {
        "wall",
        "floor",
        "ceiling",
    }:
        raise RuntimeError(f"texture set {variant_id!r} must define wall, floor, and ceiling")
    surfaces: dict[str, SurfaceVariant] = {}
    for surface_role, raw_surface in raw_surfaces.items():
        if not isinstance(raw_surface, dict):
            raise RuntimeError(f"invalid {surface_role} surface in texture set {variant_id!r}")
        surface = _surface_variant(
            alias,
            raw_surface,
            role=surface_role,
            selector_cvar=selector_cvar,
        )
        if surface.scenario_index != scenario_index or surface.theme != theme:
            raise RuntimeError(
                f"texture set {variant_id!r} has inconsistent {surface_role} metadata"
            )
        surfaces[surface_role] = surface
    return TextureSetVariant(
        role=role,
        selector_cvar=selector_cvar,
        variant_id=variant_id,
        scenario_index=scenario_index,
        theme=theme,
        surfaces=MappingProxyType(surfaces),
    )


def load_surface_catalog(
    alias: str,
) -> tuple[Mapping[str, tuple[AppearanceVariant, ...]], str]:
    document = _catalog_document(alias)
    raw_roles = document.get("roles")
    if not isinstance(raw_roles, dict) or not raw_roles:
        raise RuntimeError("surface-variant catalog must contain roles")
    resolved_roles: dict[str, tuple[AppearanceVariant, ...]] = {}
    for role, raw_role in raw_roles.items():
        if not isinstance(role, str) or not _SAFE_ROLE.fullmatch(role):
            raise RuntimeError(f"invalid surface role: {role!r}")
        if not isinstance(raw_role, dict):
            raise RuntimeError(f"surface role {role!r} must be an object")
        selector_cvar = str(raw_role.get("selector_cvar") or "")
        if not _SAFE_CVAR.fullmatch(selector_cvar):
            raise RuntimeError(f"invalid selector cvar for surface role {role!r}")
        raw_variants = raw_role.get("variants")
        if not isinstance(raw_variants, list) or not raw_variants:
            raise RuntimeError(f"surface role {role!r} must contain variants")
        variants: list[AppearanceVariant] = []
        seen_ids: set[str] = set()
        seen_indices: set[int] = set()
        for raw in raw_variants:
            if not isinstance(raw, dict):
                raise RuntimeError("surface-variant catalog entries must be objects")
            variant = (
                _texture_set_variant(
                    alias,
                    raw,
                    role=role,
                    selector_cvar=selector_cvar,
                )
                if "surfaces" in raw
                else _surface_variant(
                    alias,
                    raw,
                    role=role,
                    selector_cvar=selector_cvar,
                )
            )
            if variant.variant_id in seen_ids or variant.scenario_index in seen_indices:
                raise RuntimeError(f"surface role {role!r} has duplicate ids or scenario indices")
            seen_ids.add(variant.variant_id)
            seen_indices.add(variant.scenario_index)
            variants.append(variant)
        defaults = raw_role.get("default_variants")
        if (
            not isinstance(defaults, list)
            or not defaults
            or any(value not in seen_ids for value in defaults)
        ):
            raise RuntimeError(f"surface role {role!r} has invalid default variants")
        resolved_roles[role] = tuple(variants)
    _validated_themes(document, resolved_roles)
    canonical = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    return MappingProxyType(resolved_roles), hashlib.sha256(canonical).hexdigest()


def _validated_themes(
    document: Mapping[str, Any],
    roles: Mapping[str, tuple[AppearanceVariant, ...]],
) -> Mapping[str, Mapping[str, str]]:
    raw_themes = document.get("themes")
    if not isinstance(raw_themes, dict) or not raw_themes:
        raise RuntimeError("surface-variant catalog must contain themes")
    ids_by_role = {
        role: {variant.variant_id: variant for variant in variants}
        for role, variants in roles.items()
    }
    themes: dict[str, Mapping[str, str]] = {}
    assigned: set[str] = set()
    for theme_id, raw_theme in raw_themes.items():
        if not isinstance(theme_id, str) or not _SAFE_IDENTIFIER.fullmatch(theme_id):
            raise RuntimeError(f"invalid surface theme: {theme_id!r}")
        if not isinstance(raw_theme, dict) or not str(raw_theme.get("display_name") or ""):
            raise RuntimeError(f"surface theme {theme_id!r} must have a display name")
        raw_variants = raw_theme.get("variants")
        if not isinstance(raw_variants, dict) or set(raw_variants) != set(roles):
            raise RuntimeError(f"surface theme {theme_id!r} must cover every role")
        resolved: dict[str, str] = {}
        for role, raw_variant_id in raw_variants.items():
            variant_id = str(raw_variant_id)
            variant = ids_by_role[role].get(variant_id)
            if variant is None or variant.theme != theme_id:
                raise RuntimeError(f"surface theme {theme_id!r} has invalid {role} variant")
            if variant_id in assigned:
                raise RuntimeError(f"surface variant {variant_id!r} belongs to two themes")
            assigned.add(variant_id)
            resolved[role] = variant_id
        themes[theme_id] = MappingProxyType(resolved)
    return MappingProxyType(themes)


def load_surface_themes(alias: str) -> Mapping[str, Mapping[str, str]]:
    catalog, _catalog_hash = load_surface_catalog(alias)
    return _validated_themes(_catalog_document(alias), catalog)


def load_defend_line_surface_catalog() -> tuple[Mapping[str, tuple[AppearanceVariant, ...]], str]:
    return load_surface_catalog("defend_the_line_plus")


def load_defend_line_surface_themes() -> Mapping[str, Mapping[str, str]]:
    return load_surface_themes("defend_the_line_plus")


def _requested_ids(
    role: str,
    value: Sequence[str] | None,
    defaults: Sequence[str],
) -> tuple[str, ...]:
    if value is None:
        raw_ids = tuple(str(item) for item in defaults)
    else:
        if isinstance(value, (str, bytes, bytearray)):
            raise TypeError(f"surface_variants[{role!r}] must be a sequence of ids")
        raw_ids = tuple(str(item).strip() for item in value)
    if not raw_ids:
        raise ValueError(f"surface_variants[{role!r}] must select at least one variant")
    if len(set(raw_ids)) != len(raw_ids):
        raise ValueError(f"surface_variants[{role!r}] cannot contain duplicates")
    return raw_ids


def resolve_surface_variants(
    alias: str,
    requested: Mapping[str, Sequence[str]] | None,
) -> tuple[Mapping[str, tuple[AppearanceVariant, ...]], str]:
    catalog, catalog_hash = load_surface_catalog(alias)
    document = _catalog_document(alias)
    document_roles = document["roles"]
    environment = (
        "Defend the Line"
        if alias == "defend_the_line_plus"
        else str(document.get("environment") or "Plus environment")
    )
    if requested is None:
        requested_by_role: Mapping[str, Sequence[str]] = {}
    elif isinstance(requested, Mapping):
        unknown_roles = sorted(set(requested) - set(catalog))
        if unknown_roles:
            raise ValueError(f"unknown {environment} surface role(s): {unknown_roles}")
        requested_by_role = requested
    else:
        raise TypeError("surface_variants must be a role mapping")

    selected: dict[str, tuple[AppearanceVariant, ...]] = {}
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
            raise ValueError(f"unknown {role} surface variant(s): {unknown}; choose from {choices}")
        selected[role] = tuple(by_id[variant_id] for variant_id in raw_ids)
    return MappingProxyType(selected), catalog_hash


def resolve_defend_line_surface_variants(
    requested: Mapping[str, Sequence[str]] | None,
) -> tuple[Mapping[str, tuple[AppearanceVariant, ...]], str]:
    return resolve_surface_variants("defend_the_line_plus", requested)
