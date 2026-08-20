#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

from build_defend_line_plus import read_wad, write_wad

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
SOURCE_ROOT = PACKAGE_ROOT / "scenarios" / "basic_plus"
ENEMY_ASSET_ROOT = (
    PACKAGE_ROOT / "python" / "env_vizdoom_turbo" / "assets" / "enemy_variants"
)
SURFACE_ASSET_ROOT = (
    PACKAGE_ROOT / "python" / "env_vizdoom_turbo" / "assets" / "surface_variants"
)
ENEMY_CATALOG = ENEMY_ASSET_ROOT / "basic" / "catalog.json"
SURFACE_CATALOG = SURFACE_ASSET_ROOT / "basic" / "catalog.json"
OUTPUT_WAD = ENEMY_ASSET_ROOT / "basic" / "basic_plus.wad"


def _resolved_asset(catalog: Path, relative_path: str) -> Path:
    path = (catalog.parent / relative_path).resolve()
    if catalog.parents[1].resolve() not in path.parents:
        raise RuntimeError(f"asset escapes packaged variant root: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acc", type=Path, required=True)
    parser.add_argument("--acc-include", type=Path, required=True)
    args = parser.parse_args()
    enemy_catalog = json.loads(ENEMY_CATALOG.read_text(encoding="utf-8"))
    surface_catalog = json.loads(SURFACE_CATALOG.read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="basic-plus-") as directory:
        behavior = Path(directory) / "BEHAVIOR.o"
        subprocess.run(
            [
                str(args.acc),
                "-i",
                str(args.acc_include),
                str(SOURCE_ROOT / "SCRIPTS.acs"),
                str(behavior),
            ],
            check=True,
        )
        replacements = {
            "BEHAVIOR": behavior.read_bytes(),
            "SCRIPTS": (SOURCE_ROOT / "SCRIPTS.acs").read_bytes(),
        }
        source = read_wad(REPOSITORY_ROOT / "scenarios" / "basic.wad")
        output = [(name, replacements.get(name, payload)) for name, payload in source]
        output.extend(
            [
                ("DECORATE", (SOURCE_ROOT / "DECORATE.txt").read_bytes()),
                ("S_START", b""),
            ]
        )
        for role in enemy_catalog["roles"].values():
            for variant in role["variants"]:
                relative_manifest = variant.get("manifest")
                if not relative_manifest:
                    continue
                manifest_path = _resolved_asset(ENEMY_CATALOG, relative_manifest)
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                for frame in manifest["frames"]:
                    output.append(
                        (
                            frame["lump"],
                            (manifest_path.parent / frame["patch"]).read_bytes(),
                        )
                    )
        output.append(("S_END", b""))

        namespace_markers = {
            "texture": ("TX_START", "TX_END"),
            "flat": ("F_START", "F_END"),
        }
        for namespace, (start, end) in namespace_markers.items():
            output.append((start, b""))
            for role in surface_catalog["roles"].values():
                for variant in role["variants"]:
                    for surface in variant.get("surfaces", {}).values():
                        relative_manifest = surface.get("manifest")
                        if not relative_manifest or surface["namespace"] != namespace:
                            continue
                        manifest_path = _resolved_asset(
                            SURFACE_CATALOG,
                            relative_manifest,
                        )
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        texture = manifest["texture"]
                        output.append(
                            (
                                manifest["lump"],
                                (manifest_path.parent / texture["png"]).read_bytes(),
                            )
                        )
            output.append((end, b""))
        write_wad(OUTPUT_WAD, output)


if __name__ == "__main__":
    main()
