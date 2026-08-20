#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import struct
import subprocess
import tempfile
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PACKAGE_ROOT.parent
SOURCE_ROOT = PACKAGE_ROOT / "scenarios" / "defend_the_line_plus"
ENEMY_ASSET_ROOT = (
    PACKAGE_ROOT / "python" / "env_vizdoom_turbo" / "assets" / "enemy_variants" / "defend_the_line"
)
SURFACE_ASSET_ROOT = (
    PACKAGE_ROOT / "python" / "env_vizdoom_turbo" / "assets" / "surface_variants" / "defend_the_line"
)


def read_wad(path: Path) -> list[tuple[str, bytes]]:
    data = path.read_bytes()
    magic, count, directory = struct.unpack_from("<4sii", data, 0)
    if magic not in {b"IWAD", b"PWAD"}:
        raise ValueError(f"{path} is not a WAD")
    lumps: list[tuple[str, bytes]] = []
    for index in range(count):
        offset, size, raw_name = struct.unpack_from("<ii8s", data, directory + index * 16)
        name = raw_name.rstrip(b"\0").decode("ascii")
        lumps.append((name, data[offset : offset + size]))
    return lumps


def write_wad(path: Path, lumps: list[tuple[str, bytes]]) -> None:
    data = bytearray(b"PWAD" + struct.pack("<ii", len(lumps), 0))
    entries: list[tuple[int, int, str]] = []
    for name, payload in lumps:
        entries.append((len(data), len(payload), name))
        data.extend(payload)
    struct.pack_into("<i", data, 8, len(data))
    for offset, size, name in entries:
        data.extend(struct.pack("<ii8s", offset, size, name.encode().ljust(8, b"\0")))
    path.write_bytes(data)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--acc", type=Path, required=True)
    parser.add_argument("--acc-include", type=Path, required=True)
    args = parser.parse_args()
    enemy_catalog = json.loads((ENEMY_ASSET_ROOT / "catalog.json").read_text(encoding="utf-8"))
    surface_catalog = json.loads((SURFACE_ASSET_ROOT / "catalog.json").read_text(encoding="utf-8"))

    with tempfile.TemporaryDirectory(prefix="defend-line-plus-") as directory:
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
        source = read_wad(REPOSITORY_ROOT / "scenarios" / "defend_the_line.wad")
        output: list[tuple[str, bytes]] = [
            (name, replacements.get(name, payload)) for name, payload in source
        ]
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
                manifest_path = ENEMY_ASSET_ROOT / relative_manifest
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
                    relative_manifest = variant.get("manifest")
                    if not relative_manifest or variant["namespace"] != namespace:
                        continue
                    manifest_path = SURFACE_ASSET_ROOT / relative_manifest
                    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                    texture = manifest["texture"]
                    output.append(
                        (
                            manifest["lump"],
                            (manifest_path.parent / texture["png"]).read_bytes(),
                        )
                    )
            output.append((end, b""))
        write_wad(ENEMY_ASSET_ROOT / "defend_the_line_plus.wad", output)


if __name__ == "__main__":
    main()
