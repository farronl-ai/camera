import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def test_owner_inspection_lab_is_complete_and_oracle_labeled():
    manifest = json.loads((DOCS / "inspection_manifest.json").read_text())
    assert manifest["schema"] == 7
    assert [case["sid"] for case in manifest["factory_v2"]["cases"]] == [
        "s29_000",
        "s29_001",
        "s29_002",
    ]
    assert all(
        case["core_fraction"] == 1.0
        for case in manifest["factory_v2"]["cases"]
    )
    assert all(
        (DOCS / case["asset"]).is_file()
        for case in manifest["factory_v2"]["cases"]
    )
    assert len(manifest["cases"]) == 7
    assert len(manifest["normal_cases"]) == 6
    assert "never enter" in manifest["oracle_warning"].lower()

    current_cases = {
        "s29_000",
        "s29_002",
        "s29_005",
        "s29_007",
        "s29_009",
        "s29_010",
        "s29_011",
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
    summary = manifest["s12_summary"]
    assert summary["auto_enabled"] is True
    assert summary["current_s29_scene_count"] == 12
    assert summary["current_s29_fired_count"] == 11
    assert summary["current_s29_all_partitions_nonregressing"] == 11
    assert summary["current_s29_protected_rear_overlap"] == 0
    assert summary["formation_owned_invariant"] is True
    assert summary["formation_outer_veil_retained"] is True
    assert summary["normal_case_count"] == 6

    required_assets = {
        "frame0",
        "frame1",
        "base",
        "output",
        "gt",
        "estimated_alpha",
        "true_alpha",
        "hard_ownership",
        "application_mask",
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
        assert case["metrics"]["d_mae"] < 0
        assert case["metrics"]["d_mse"] < 0
        assert case["report"]["candidate_rank"] == -1
        assert case["report"]["one_sided_geometry_fired"] is True
        assert (
            case["metrics"]["owner_support_pixels"]
            == case["report"]["owner_support_pixels"]
        )
        assert set(case["metrics"]["partitions"]) == {
            "hard_owned_foreground",
            "owned_foreground_core",
            "hard_owned_soft_edge",
            "foreground_boundary",
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

    assert all(case["split"] == "s29" for case in manifest["cases"])
    assert all(
        case["metrics"]["partitions"]["far_background"]["mae_output"]
        == case["metrics"]["partitions"]["far_background"]["mae_base"]
        for case in manifest["cases"]
    )

    assert all(
        case["report"]["veil_disabled_safety"] is False
        for case in manifest["normal_cases"]
    )
    assert sum(
        case["report"]["veil_reason"] == "licensed_consensus"
        for case in manifest["normal_cases"]
    ) == 2

    html = (DOCS / "INSPECTION.html").read_text()
    assert "__INSPECTION_MANIFEST__" not in html
    assert "GT-only" in html
    assert "validated two-frame auto path" in html
    assert "Current optical foundation" in html
    assert "Final rear-application weight" in html
    assert "fresh S29 licensed fires" in html
    assert "Every original input frame" in html
    assert "Aligned + exposure-normalized" in html
    assert "no legacy inputs or results" in html.lower()
    assert "Every image input" in html
    assert "Copy diagnostic note" in html
    assert "BASE</strong> is always left of the divider" in html
    assert "OUTPUT</strong> is always right of the divider" in html
    assert ".compare .after { clip-path: inset(0 0 0 50%);" in html
    assert "after.style.clipPath = `inset(0 0 0 ${v}%)`" in html
    assert "Select region on large image (optional)" in html
    assert "True hard ownership · GT-only" in html
    assert "region selection is active only in this window" in html
    assert "probe-img" not in html
    assert "Click to record a native coordinate" not in html
    assert '"sid":"s23_' not in html
    for scene_id in current_cases | normal_cases:
        assert f'"sid":"{scene_id}"' in html
