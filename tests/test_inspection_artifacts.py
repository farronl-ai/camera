import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "docs"


def test_owner_inspection_lab_is_complete_and_oracle_labeled():
    manifest = json.loads((DOCS / "inspection_manifest.json").read_text())
    assert len(manifest["ledger"]) == 10
    assert len(manifest["cases"]) == 5
    assert "never inputs" in manifest["oracle_warning"].lower()

    expected_cases = {"scene_72", "scene_75", "scene_114", "scene_122", "scene_147"}
    assert {case["sid"] for case in manifest["cases"]} == expected_cases
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
        "protected",
        "edit_x8",
        "error_delta",
        "outcomes",
        "crop_edit",
        "crop_worse",
    }
    for case in manifest["cases"]:
        assert set(case["assets"]) == required_assets
        assert case["report"]["fired"] is True
        assert case["metrics"]["d_ssim"] > 0
        assert case["metrics"]["d_mae"] < 0
        for relative_path in case["assets"].values():
            assert (DOCS / relative_path).is_file(), relative_path

    html = (DOCS / "INSPECTION.html").read_text()
    assert "__INSPECTION_MANIFEST__" not in html
    assert "GT-only" in html
    assert "Every image input" in html
    assert "Copy diagnostic note" in html
    assert "BASE</strong> is always left of the divider" in html
    assert "OUTPUT</strong> is always right of the divider" in html
    assert ".compare .after { clip-path: inset(0 0 0 50%);" in html
    assert "after.style.clipPath = `inset(0 0 0 ${v}%)`" in html
    assert "Select region on large image (optional)" in html
    assert "probe-img" not in html
    assert "Click to record a native coordinate" not in html
    for scene_id in expected_cases:
        assert f'"sid":"{scene_id}"' in html
