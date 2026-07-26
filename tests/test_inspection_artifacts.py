import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def test_owner_inspection_lab_is_complete_and_oracle_labeled():
    manifest = json.loads((DOCS / "inspection_manifest.json").read_text())
    assert manifest["schema"] == 4
    assert [case["stratum"] for case in manifest["factory_v2"]["cases"]] == [
        "solid",
        "mixed",
        "thin",
    ]
    assert all(
        case["core_fraction"] >= 0.55
        for case in manifest["factory_v2"]["cases"]
        if case["stratum"] == "solid"
    )
    assert len(manifest["ledger"]) == 10
    assert len(manifest["cases"]) == 8
    assert "never inputs" in manifest["oracle_warning"].lower()

    current_cases = {"extension_007", "extension_034", "s12_025"}
    legacy_cases = {
        "scene_72",
        "scene_75",
        "scene_114",
        "scene_122",
        "scene_147",
    }
    expected_cases = current_cases | legacy_cases
    assert {case["sid"] for case in manifest["cases"]} == expected_cases
    assert {
        case["sid"]
        for case in manifest["cases"]
        if case.get("current_v2")
    } == current_cases
    assert manifest["s12_summary"]["post_freeze_scene_count"] == 36
    assert manifest["s12_summary"]["post_freeze_fired_count"] == 1
    assert (
        manifest["s12_summary"]["post_freeze_all_partitions_nonregressing"]
        == 1
    )
    assert all(row["dg"] > 0 for row in manifest["ledger"])
    assert all(row["d_global_mae"] < 0 for row in manifest["ledger"])
    assert all(row["d_global_mse"] < 0 for row in manifest["ledger"])
    assert all(row["d_false_texture"] > 0 for row in manifest["ledger"])

    required_assets = {
        "frame0",
        "frame1",
        "base",
        "output",
        "gt",
        "estimated_alpha",
        "true_alpha",
        "application_mask",
        "owner_support",
        "protected",
        "edit_x8",
        "error_delta",
        "outcomes",
        "crop_edit",
        "crop_worse",
    }
    for case in manifest["cases"]:
        assert set(case["assets"]) >= required_assets
        assert case["report"]["fired"] is True
        assert case["metrics"]["d_ssim"] > 0
        assert case["metrics"]["d_mae"] < 0
        assert (
            case["metrics"]["owner_support_pixels"]
            == case["report"]["owner_support_pixels"]
        )
        if case.get("current_v2"):
            assert set(case["metrics"]["partitions"]) == {
                "complete_coverage_core",
                "inner_partial_occlusion",
                "outer_veil",
                "far_background",
            }
            assert "coverage_classes" in case["assets"]
            assert "ordered_visibility" in case["assets"]
            assert all(
                values["mae_base"] is None
                or values["mae_output"] <= values["mae_base"]
                for values in case["metrics"]["partitions"].values()
            )
        for relative_path in case["assets"].values():
            assert (DOCS / relative_path).is_file(), relative_path

    scene_114 = next(
        case for case in manifest["cases"] if case["sid"] == "scene_114"
    )
    assert scene_114["diagnostic_point"]["x"] == 187
    assert scene_114["diagnostic_point"]["y"] == 252
    assert scene_114["diagnostic_point"]["owner_support"] is True
    assert scene_114["diagnostic_point"]["output_error"] < (
        scene_114["diagnostic_point"]["base_error"]
    )
    assert "crop_reported" in scene_114["assets"]

    scene_122 = next(
        case for case in manifest["cases"] if case["sid"] == "scene_122"
    )
    assert scene_122["diagnostic_point"]["x"] == 804
    assert scene_122["diagnostic_point"]["y"] == 521
    assert scene_122["diagnostic_point"]["owner_support"] is True
    assert scene_122["diagnostic_point"]["output_error"] < (
        scene_122["diagnostic_point"]["base_error"]
    )
    assert "parent_silhouette" in scene_122["report"]["owner_support_kinds"]
    assert "crop_reported" in scene_122["assets"]

    html = (DOCS / "INSPECTION.html").read_text()
    assert "__INSPECTION_MANIFEST__" not in html
    assert "GT-only" in html
    assert "safety-disabled" in html
    assert "Current optical foundation" in html
    assert "Ordered visibility" in html
    assert "new post-freeze S12 fires" in html
    assert "Every image input" in html
    assert "Copy diagnostic note" in html
    assert "BASE</strong> is always left of the divider" in html
    assert "OUTPUT</strong> is always right of the divider" in html
    assert ".compare .after { clip-path: inset(0 0 0 50%);" in html
    assert "after.style.clipPath = `inset(0 0 0 ${v}%)`" in html
    assert "Select region on large image (optional)" in html
    assert "Owner-frame support · runtime" in html
    assert "probe-img" not in html
    assert "Click to record a native coordinate" not in html
    for scene_id in expected_cases:
        assert f'"sid":"{scene_id}"' in html
