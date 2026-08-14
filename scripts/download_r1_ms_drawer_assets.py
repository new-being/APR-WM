"""Audit and download only the assets used as drawers by ManiSkill 3.0.1.

The upstream ``partnet_mobility_cabinet`` group also includes cabinet doors.
R1-MS0 intentionally avoids that broader group and derives the exact drawer
IDs from the installed ManiSkill version's metadata.

No mirror URL can be supplied to this script.  Every remote source must come
from the installed ManiSkill ``DataSource`` registry, and the complete source
metadata is written before any network operation starts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlparse


AUDIT_SCHEMA_VERSION = 1
DRAWER_ENV_ID = "OpenCabinetDrawer-v1"
DRAWER_UPSTREAM_GROUP = "partnet_mobility_cabinet"


def tree_digest(root: Path) -> tuple[str, int, int]:
    digest = hashlib.sha256()
    files = 0
    size = 0
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        files += 1
    return digest.hexdigest(), files, size


def _json_path(path: Any) -> str | None:
    return None if path is None else str(Path(path).resolve())


def _official_sources(data_source: Any) -> list[dict[str, str]]:
    """Return only source locations declared by the installed registry."""

    sources: list[dict[str, str]] = []
    if data_source.hf_repo_id:
        sources.append({"kind": "hf_repo_id", "value": data_source.hf_repo_id})
    if data_source.github_url:
        sources.append({"kind": "github_url", "value": data_source.github_url})
    if data_source.url:
        sources.append({"kind": "url", "value": data_source.url})
    return sources


def validate_registered_source(data_source_id: str, data_source: Any) -> None:
    """Reject missing, insecure, or ambiguous non-registry transport metadata."""

    sources = _official_sources(data_source)
    if not sources:
        raise ValueError(f"{data_source_id} has no source in ManiSkill metadata")
    for source in sources:
        if source["kind"] in {"url", "github_url"}:
            parsed = urlparse(source["value"])
            if parsed.scheme != "https" or not parsed.hostname:
                raise ValueError(
                    f"{data_source_id} has a non-HTTPS registered source: "
                    f"{source['value']}"
                )
        elif source["kind"] == "hf_repo_id":
            repo_id = source["value"]
            if repo_id.startswith(("/", ".")) or repo_id.count("/") != 1:
                raise ValueError(
                    f"{data_source_id} has an invalid registered hf_repo_id: {repo_id}"
                )


def asset_source_row(asset_id: str, data_source_id: str, data_source: Any) -> dict[str, Any]:
    validate_registered_source(data_source_id, data_source)
    sources = _official_sources(data_source)
    target_path = Path(data_source.target_path)
    output_dir = Path(data_source.output_dir)
    resolved_target = target_path if target_path.is_absolute() else output_dir / target_path
    alternatives = [
        source
        for source in sources
        if source["kind"] in {"hf_repo_id", "github_url"}
    ]
    return {
        "asset_id": asset_id,
        "data_source_id": data_source_id,
        "source_type": data_source.source_type,
        "url": data_source.url,
        "github_url": data_source.github_url,
        "hf_repo_id": data_source.hf_repo_id,
        "checksum": data_source.checksum,
        "target_path": _json_path(target_path),
        "output_dir": _json_path(output_dir),
        "resolved_target_path": _json_path(resolved_target),
        "zip_dirname": data_source.zip_dirname,
        "filename": data_source.filename,
        "official_sources": sources,
        "official_alternative_sources": alternatives,
        "has_official_alternative_source": bool(alternatives),
        "has_archive_checksum": data_source.checksum is not None,
    }


def build_source_audit(
    *,
    asset_ids: Iterable[str],
    data_sources: dict[str, Any],
    data_groups: dict[str, Any],
    package_version: str,
    package_dir: Path,
    package_asset_dir: Path,
    metadata_path: Path,
    asset_root: Path,
    source_commit: str | None,
) -> dict[str, Any]:
    asset_ids = sorted(str(asset_id) for asset_id in asset_ids)
    rows = []
    for asset_id in asset_ids:
        data_source_id = f"partnet_mobility/{asset_id}"
        if data_source_id not in data_sources:
            raise KeyError(f"Missing installed ManiSkill DataSource: {data_source_id}")
        rows.append(asset_source_row(asset_id, data_source_id, data_sources[data_source_id]))

    drawer_source_ids = {row["data_source_id"] for row in rows}
    registered_dependencies = list(data_groups.get(DRAWER_ENV_ID, []))
    upstream_group_ids = set(data_groups.get(DRAWER_UPSTREAM_GROUP, []))
    missing_from_upstream_group = sorted(drawer_source_ids - upstream_group_ids)
    if missing_from_upstream_group:
        raise ValueError(
            "Drawer metadata and installed cabinet group disagree: "
            + ", ".join(missing_from_upstream_group)
        )

    url_hosts = sorted(
        {
            urlparse(row["url"]).hostname
            for row in rows
            if row["url"] is not None
        }
    )
    return {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "scope": f"{DRAWER_ENV_ID} drawer assets only; no network operation",
        "installed_maniskill": {
            "version": package_version,
            "source_commit": source_commit,
            "package_dir": _json_path(package_dir),
            "package_asset_dir": _json_path(package_asset_dir),
            "drawer_metadata_path": _json_path(metadata_path),
            "drawer_metadata_sha256": hashlib.sha256(metadata_path.read_bytes()).hexdigest(),
        },
        "asset_root": _json_path(asset_root),
        "registered_environment_dependencies": registered_dependencies,
        "registered_upstream_group": DRAWER_UPSTREAM_GROUP,
        "registered_upstream_group_count": len(upstream_group_ids),
        "drawer_metadata_count": len(rows),
        "drawer_entries_in_upstream_group": len(drawer_source_ids & upstream_group_ids),
        "non_drawer_entries_in_upstream_group": len(upstream_group_ids - drawer_source_ids),
        "source_policy": {
            "registry_is_sole_source_of_remote_locations": True,
            "custom_or_third_party_mirror_allowed": False,
            "tls_verification_may_be_disabled": False,
            "official_alternatives_considered": ["hf_repo_id", "github_url"],
            "archive_checksum_required_when_published": True,
            "local_tree_sha256_is_not_archive_authenticity": True,
        },
        "summary": {
            "asset_count": len(rows),
            "url_hosts": url_hosts,
            "assets_with_hf_repo_id": sum(row["hf_repo_id"] is not None for row in rows),
            "assets_with_github_url": sum(row["github_url"] is not None for row in rows),
            "assets_with_archive_checksum": sum(row["checksum"] is not None for row in rows),
            "assets_with_official_alternative_source": sum(
                row["has_official_alternative_source"] for row in rows
            ),
            "all_sources_are_registered_https_or_hf": True,
            "offline_transfer_authenticatable_by_published_checksum": all(
                row["has_archive_checksum"] for row in rows
            ),
        },
        "assets": rows,
    }


def _source_commit(mani_skill_module: Any) -> str | None:
    try:
        info = mani_skill_module.get_commit_info()
    except Exception:
        return None
    return None if not info else info.get("commit_id")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", default=".maniskill")
    parser.add_argument(
        "--audit-output",
        default="runs/r1_ms/preflight/asset_source_audit.json",
        help="Write installed ManiSkill DataSource metadata here before downloading.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Audit metadata without performing any network operation.",
    )
    args = parser.parse_args()
    asset_root = Path(args.asset_root).resolve()
    os.environ["MS_ASSET_DIR"] = str(asset_root)
    os.environ["MS_SKIP_ASSET_DOWNLOAD_PROMPT"] = "1"

    import mani_skill
    import mani_skill.envs  # noqa: F401 - registers environment asset groups
    from mani_skill import PACKAGE_ASSET_DIR, PACKAGE_DIR
    from mani_skill.utils.assets.data import DATA_GROUPS, DATA_SOURCES

    metadata = (
        PACKAGE_ASSET_DIR
        / "partnet_mobility"
        / "meta"
        / "info_cabinet_drawer_train.json"
    )
    asset_ids = sorted(json.loads(metadata.read_text(encoding="utf-8")))
    audit = build_source_audit(
        asset_ids=asset_ids,
        data_sources=DATA_SOURCES,
        data_groups=DATA_GROUPS,
        package_version=mani_skill.__version__,
        package_dir=PACKAGE_DIR,
        package_asset_dir=PACKAGE_ASSET_DIR,
        metadata_path=metadata,
        asset_root=asset_root,
        source_commit=_source_commit(mani_skill),
    )
    _write_json(Path(args.audit_output), audit)
    print(json.dumps({key: value for key, value in audit.items() if key != "assets"}, indent=2))
    if args.audit_only:
        return

    # Importing the downloader is deliberately delayed until the immutable
    # registry audit has been persisted.
    from mani_skill.utils.download_asset import download

    dataset_root = asset_root / "data" / "partnet_mobility" / "dataset"
    rows = []
    for position, asset_id in enumerate(asset_ids, start=1):
        data_source_id = f"partnet_mobility/{asset_id}"
        data_source = DATA_SOURCES[data_source_id]
        validate_registered_source(data_source_id, data_source)
        target = dataset_root / asset_id
        if target.is_dir() and any(target.iterdir()):
            status = "reused"
        else:
            print(f"[{position}/{len(asset_ids)}] downloading drawer {asset_id}")
            download(data_source, verbose=True, non_interactive=True)
            status = "downloaded"
        digest, file_count, byte_count = tree_digest(target)
        rows.append(
            {
                "asset_id": asset_id,
                "data_source_id": data_source_id,
                "status": status,
                "files": file_count,
                "bytes": byte_count,
                "local_tree_sha256": digest,
                "archive_checksum": data_source.checksum,
                "archive_checksum_published": data_source.checksum is not None,
            }
        )

    manifest = {
        "scope": f"{DRAWER_ENV_ID} drawer assets only",
        "count": len(rows),
        "total_bytes": sum(row["bytes"] for row in rows),
        "assets": rows,
        "all_sources_from_installed_registry": True,
        "local_tree_sha256_is_not_archive_authenticity": True,
        "assets_with_published_archive_checksum": sum(
            row["archive_checksum_published"] for row in rows
        ),
        "source_audit": str(Path(args.audit_output).resolve()),
    }
    asset_root.mkdir(parents=True, exist_ok=True)
    _write_json(asset_root / "r1_ms_drawer_asset_manifest.json", manifest)
    print(json.dumps({key: value for key, value in manifest.items() if key != "assets"}, indent=2))


if __name__ == "__main__":
    main()
