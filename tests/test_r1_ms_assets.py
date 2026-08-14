from pathlib import Path

import pytest

from scripts.download_r1_ms_drawer_assets import (
    asset_source_row,
    build_source_audit,
    validate_registered_source,
)


class _Source:
    source_type = "objects"
    url = "https://official.example/drawer.zip"
    hf_repo_id = None
    github_url = None
    target_path = Path("partnet_mobility/dataset/1000")
    checksum = None
    zip_dirname = None
    filename = None
    output_dir = Path("/tmp/asset-root")


def test_source_row_preserves_registry_fields_without_inventing_alternative():
    row = asset_source_row("1000", "partnet_mobility/1000", _Source())
    assert row["url"] == _Source.url
    assert row["github_url"] is None
    assert row["hf_repo_id"] is None
    assert row["checksum"] is None
    assert not row["has_official_alternative_source"]
    assert not row["has_archive_checksum"]
    assert row["resolved_target_path"].endswith(
        "/tmp/asset-root/partnet_mobility/dataset/1000"
    )


def test_only_https_or_hf_registry_sources_are_accepted():
    insecure = replace_source(_Source(), url="http://official.example/drawer.zip")
    with pytest.raises(ValueError, match="non-HTTPS"):
        validate_registered_source("partnet_mobility/1000", insecure)

    missing = replace_source(_Source(), url=None)
    with pytest.raises(ValueError, match="no source"):
        validate_registered_source("partnet_mobility/1000", missing)


def test_audit_distinguishes_drawer_subset_from_broad_registered_group(tmp_path):
    metadata = tmp_path / "drawer.json"
    metadata.write_text('{"1000": {}}', encoding="utf-8")
    audit = build_source_audit(
        asset_ids=["1000"],
        data_sources={"partnet_mobility/1000": _Source()},
        data_groups={
            "OpenCabinetDrawer-v1": ["partnet_mobility_cabinet"],
            "partnet_mobility_cabinet": {
                "partnet_mobility/1000",
                "partnet_mobility/door-only",
            },
        },
        package_version="3.0.1",
        package_dir=tmp_path,
        package_asset_dir=tmp_path,
        metadata_path=metadata,
        asset_root=tmp_path / "assets",
        source_commit="abc",
    )
    assert audit["registered_upstream_group_count"] == 2
    assert audit["drawer_metadata_count"] == 1
    assert audit["non_drawer_entries_in_upstream_group"] == 1
    assert audit["summary"]["assets_with_official_alternative_source"] == 0
    assert not audit["summary"]["offline_transfer_authenticatable_by_published_checksum"]


def replace_source(source, **updates):
    values = {
        name: getattr(source, name)
        for name in (
            "source_type",
            "url",
            "hf_repo_id",
            "github_url",
            "target_path",
            "checksum",
            "zip_dirname",
            "filename",
            "output_dir",
        )
    }
    values.update(updates)
    return type("Source", (), values)()
