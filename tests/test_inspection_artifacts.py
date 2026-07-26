import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def test_owner_inspection_lab_is_complete_and_oracle_labeled():
    manifest = json.loads((DOCS / "inspection_manifest.json").read_text())
    assert manifest["schema"] == 6
    assert [case["stratum"] for case in manifest["factory_v2"]["cases"]] == [
        "same-scene opaque-primary rerender",
        "solid",
        "mixed",
        "thin",
    ]
    formation_case = manifest["factory_v2"]["cases"][0]
    assert formation_case["sid"] == "extension_007_opaque_primary_r12"
    assert formation_case["core_fraction"] >= 0.75
    assert "coverage changes 0.726→1.000" in formation_case["description"]
    assert all(
        case["core_fraction"] >= 0.55
        for case in manifest["factory_v2"]["cases"]
        if case["stratum"] == "solid"
    )
    assert len(manifest["cases"]) == 7
    assert len(manifest["normal_cases"]) == 6
    assert "never enter" in manifest["oracle_warning"].lower()

    current_cases = {
        "extension_007",
        "extension_034",
        "s12_025",
        "s16_034",
        "s19_000",
        "s19_012",
        "s19_013",
    }
    assert {case["sid"] for case in manifest["cases"]} == current_cases
    assert all(case.get("current_v2") for case in manifest["cases"])
    normal_cases = {
        "standard_c01",
        "standard_c05",
        "standard_c10",
        "standard_c20",
        "mobile_kitchen",
        "mobile_smallmotion",
    }
    assert {
        case["sid"] for case in manifest["normal_cases"]
    } == normal_cases
    assert manifest["s12_summary"]["post_freeze_scene_count"] == 36
    assert manifest["s12_summary"]["post_freeze_fired_count"] == 1
    assert (
        manifest["s12_summary"]["post_freeze_all_partitions_nonregressing"]
        == 1
    )
    assert manifest["s12_summary"]["post_refinement_scene_count"] == 36
    assert manifest["s12_summary"]["post_refinement_fired_count"] == 1
    assert (
        manifest["s12_summary"][
            "post_refinement_all_partitions_nonregressing"
        ]
        == 1
    )
    assert manifest["s12_summary"]["post_final_scene_count"] == 72
    assert manifest["s12_summary"]["post_final_fired_count"] == 3
    assert (
        manifest["s12_summary"]["post_final_all_partitions_nonregressing"]
        == 3
    )
    assert manifest["s12_summary"]["normal_case_count"] == 6

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
        assert set(case["metrics"]["partitions"]) == {
            "complete_coverage_core",
            "inner_partial_occlusion",
            "outer_veil",
            "far_background",
        }
        assert "coverage_classes" in case["assets"]
        assert "ordered_visibility" in case["assets"]
        assert "front_reconstruction" in case["assets"]
        assert (
            case["metrics"]["front_reconstruction_pixels"]
            == case["report"]["owner_front_reconstruction_pixels"]
        )
        assert all(
            values["mae_base"] is None
            or values["mae_output"] <= values["mae_base"]
            for values in case["metrics"]["partitions"].values()
        )
        for relative_path in case["assets"].values():
            assert (DOCS / relative_path).is_file(), relative_path

    required_normal_assets = {
        "inputs",
        "aligned",
        "first",
        "middle",
        "last",
        "base",
        "output",
        "selection",
        "edit_x8",
    }
    for case in manifest["normal_cases"]:
        assert set(case["assets"]) == required_normal_assets
        assert case["frame_count"] >= 2
        assert len(case["metrics"]["winner_shares"]) == case["frame_count"]
        assert abs(sum(case["metrics"]["winner_shares"]) - 1.0) < 1e-6
        assert case["ground_truth"] is False
        for relative_path in case["assets"].values():
            assert (DOCS / relative_path).is_file(), relative_path

    extension_007 = next(
        case for case in manifest["cases"]
        if case["sid"] == "extension_007"
    )
    assert extension_007["diagnostic_point"]["x"] == 1048
    assert extension_007["diagnostic_point"]["y"] == 216
    assert extension_007["diagnostic_point"]["front_reconstruction"] is True
    assert extension_007["diagnostic_point"]["output_error"] < (
        extension_007["diagnostic_point"]["base_error"]
    )
    assert "crop_reported" in extension_007["assets"]

    s16_034 = next(
        case for case in manifest["cases"] if case["sid"] == "s16_034"
    )
    assert s16_034["report"]["owner_refinement_fired"] is True
    assert s16_034["report"]["owner_refinement_forward_improvement"] > 0.01
    assert s16_034["metrics"]["partitions"]["complete_coverage_core"][
        "mae_output"
    ] == s16_034["metrics"]["partitions"]["complete_coverage_core"][
        "mae_base"
    ]

    assert all(
        case["report"]["veil_disabled_safety"] is True
        for case in manifest["normal_cases"]
    )

    html = (DOCS / "INSPECTION.html").read_text()
    assert "__INSPECTION_MANIFEST__" not in html
    assert "GT-only" in html
    assert "safety-disabled" in html
    assert "Current optical foundation" in html
    assert "Ordered visibility" in html
    assert "genuinely post-final S19 fires" in html
    assert "Every original input frame" in html
    assert "Aligned + exposure-normalized" in html
    assert "no legacy reproductions" in html.lower()
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
    for scene_id in current_cases | normal_cases:
        assert f'"sid":"{scene_id}"' in html
