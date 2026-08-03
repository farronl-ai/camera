"""The two-frame route: when it engages, when it declines, and the ported gate.

The route is F101's doctrine applied a second time — two architectures that win
on opposite scenes are ROUTED, never merged — so the tests that matter are about
the DECISION and about non-regression, not about fusion quality:

  * a scene with nothing stranded must be byte-identical with the route on or off;
  * a scene with a stranded object the two-frame path can place must engage;
  * a scene whose stranded object needs re-registration must decline (large-motion
    is the real case; here it is reproduced synthetically by moving the object
    further than the architecture's displacement licence);
  * and the ported validity gate must still catch a wrong layer fit, because a
    port is a new instrument (DEVSTYLE §12.1) — `global_stage` is checked against
    the shipped global stage the same way.
"""

import cv2
import numpy as np

from focusstack.align import align_stack
from focusstack.pipeline import run
from focusstack.fusion import fuse_coherent
from focusstack.twoframe import (GATE_TOL, SURFACE_POOL, EdgeEvidence, gate_shift,
                                 global_stage, pooled_majority, same_surface,
                                 surface_agreement, twoframe_fullres,
                                 twoframe_stack)


def _textured(seed, h, w):
    rng = np.random.default_rng(seed)
    img = np.full((h, w), 90, np.float32)
    for _ in range(160):
        x0, y0 = rng.integers(0, w - 30), rng.integers(0, h - 14)
        shade = float(rng.integers(0, 255))
        cv2.rectangle(img, (int(x0), int(y0)),
                      (int(x0 + rng.integers(6, 28)), int(y0 + rng.integers(3, 12))),
                      shade, -1)
    return img


def _disk_blur(image, radius):
    """Real defocus is a disk, not a Gaussian (PLAYBOOK §0)."""
    r = int(round(radius))
    if r < 1:
        return image
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = (xx * xx + yy * yy <= r * r).astype(np.float32)
    return cv2.filter2D(image, -1, kernel / kernel.sum())


def _stranded_object_stack(step_px, frames=5, h=360, w=460, square_focus=None,
                           background_focus=0):
    """A textured square translating over a static background, with a focal sweep.

    The moving-square fixture of `test_motion_groups`, given the one thing a
    focus stack must have: each layer is sharp in its own frame. `square_focus`
    defaults to the reference, which is the kitchen's regime — the stranded
    object is sharpest where it needs no correction, and that is exactly when the
    two-frame path can serve it.
    """
    ref = frames // 2
    if square_focus is None:
        square_focus = ref
    background = _textured(3, h, w)
    square = _textured(7, 150, 150)
    stack = []
    for k in range(frames):
        sharp = background.copy()
        x0 = 150 + int(round(step_px * (k - ref)))
        sharp[100:250, x0:x0 + 150] = square
        mask = np.zeros((h, w), bool)
        mask[100:250, x0:x0 + 150] = True
        near = _disk_blur(sharp, 1.6 * abs(k - square_focus))
        far = _disk_blur(sharp, 1.6 * abs(k - background_focus))
        frame = np.where(mask, near, far)
        stack.append(cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8),
                                  cv2.COLOR_GRAY2BGR))
    return stack, ref


def _still_stack(frames=4, h=240, w=320):
    """Frames that differ only by sensor noise: nothing is stranded."""
    rng = np.random.default_rng(11)
    base = _textured(5, h, w)
    return [cv2.cvtColor(np.clip(base + rng.normal(0, 2, base.shape), 0, 255)
                         .astype(np.uint8), cv2.COLOR_GRAY2BGR)
            for _ in range(frames)]


def _write(stack, directory):
    paths = []
    for i, frame in enumerate(stack):
        path = str(directory / f"f{i:02d}.png")
        cv2.imwrite(path, frame)
        paths.append(path)
    return paths


def test_route_declines_and_is_byte_identical_on_a_still_stack(tmp_path, capsys):
    """Non-regression by construction: nothing stranded, nothing changed."""
    paths = _write(_still_stack(), tmp_path)
    routed = run(paths, str(tmp_path / "on.png"), twoframe_route=True, verbose=True)
    log = capsys.readouterr().out
    plain = run(paths, str(tmp_path / "off.png"), twoframe_route=False)

    assert "fusion path: shipped depth-bin" in log
    assert "TWO-FRAME" not in log
    assert np.array_equal(routed, plain)


def test_route_engages_on_a_stranded_object_the_two_frame_path_can_place(tmp_path,
                                                                         capsys):
    """The kitchen's regime in miniature: an object the depth path cannot see,
    sharpest where it needs no correction."""
    stack, _ref = _stranded_object_stack(step_px=6.0)
    paths = _write(stack, tmp_path)
    routed = run(paths, str(tmp_path / "on.png"), twoframe_route=True, verbose=True)
    log = capsys.readouterr().out
    plain = run(paths, str(tmp_path / "off.png"), twoframe_route=False)

    assert "fusion path: TWO-FRAME" in log
    assert not np.array_equal(routed, plain)


def test_route_declines_when_the_object_needs_re_registration(tmp_path, capsys):
    """Large-motion's regime: the stranded object is sharpest far from the
    reference, so serving it means re-registering by more than the architecture's
    displacement licence. The composite is built, measured, and discarded."""
    stack, _ref = _stranded_object_stack(step_px=8.0, square_focus=0)
    paths = _write(stack, tmp_path)
    routed = run(paths, str(tmp_path / "on.png"), twoframe_route=True, verbose=True)
    log = capsys.readouterr().out
    plain = run(paths, str(tmp_path / "off.png"), twoframe_route=False)

    assert "DECLINED" in log
    assert "fusion path: shipped depth-bin" in log
    assert np.array_equal(routed, plain)


def test_ported_global_stage_still_matches_the_shipped_one():
    """A port is a new instrument. This one claims to be a VIEW of the shipped
    global stage that also returns its matrices; identical pixels is the claim."""
    stack, ref = _stranded_object_stack(step_px=4.0)
    coarse, warps, _valid = global_stage(stack, ref)
    shipped = align_stack(stack, ref_index=ref, depth_bins=0, crop_valid=False)
    for ours, theirs in zip(coarse, shipped):
        assert np.array_equal(ours, theirs)
    assert warps[ref] is None


def test_the_validity_gate_catches_an_injected_layer_error():
    """The promotion blocker the gate exists for: a wrong-but-plausible layer
    shift is invisible to `max_shift`, and F110 measured one taking the factory
    to 0.668 with nothing objecting. Inject +8 px and require the gate to recover
    the clean composite.

    ROUND 4 scoping: every run here has the same-surface precondition OFF, so the
    subject is the GATE. With it on, the injected member is refused a second time
    on appearance (8 px out of place is not the same surface) and the ungated
    composite is barely damaged at all — 0.47 against 4.35 — which would make the
    gate's own test vacuous. The second instrument is asserted separately below
    rather than allowed to stand in for the first (§12.2).
    """
    stack, ref = _stranded_object_stack(step_px=6.0)
    clean, info = twoframe_stack(stack, ref, surface=False)
    assert len(info["pairs"]) >= 1

    # Into a real two-member pair: a degenerate one-frame region has no second
    # observation, so refusing it falls back to the unwarped reference and the
    # test would be measuring the fallback rather than the gate.
    pair_index = next(i for i, d in enumerate(info["diagnostics"])
                      if len(d["frames"]) == 2)
    error = {(pair_index, 0): (8.0, 0.0)}
    ungated, ungated_info = twoframe_stack(stack, ref, inject=error, gate=False,
                                           surface=False)
    gated, gated_info = twoframe_stack(stack, ref, inject=error, gate=True,
                                       surface=False)

    def compare(a, a_info, b, b_info):
        """Mean |difference| on the region both outputs actually observed.

        A wrong layer shift moves the common footprint, so the crops differ; each
        output is mapped back into REFERENCE coordinates before differencing. The
        research harness lost a run to exactly this trap (F110 H4).
        """
        ax0, ay0, ax1, ay1 = a_info["crop"]
        bx0, by0, bx1, by1 = b_info["crop"]
        x0, y0 = max(ax0, bx0), max(ay0, by0)
        x1, y1 = min(ax1, bx1), min(ay1, by1)
        left = a[y0 - ay0:y1 - ay0, x0 - ax0:x1 - ax0].astype(float)
        right = b[y0 - by0:y1 - by0, x0 - bx0:x1 - bx0].astype(float)
        return float(np.abs(left - right).mean())

    damage = compare(ungated, ungated_info, clean, info)
    residual = compare(gated, gated_info, clean, info)
    assert damage > 1.0, damage                 # the error really is damaging
    assert residual < 0.25 * damage, (residual, damage)

    # ROUND 4, the second instrument, stated rather than relied on: with the
    # precondition ON and the gate still OFF, the same injected error is refused
    # on APPEARANCE — a layer 8 px out of place is not observing the surface the
    # reference does. Measured against the same clean composite, so the two
    # instruments are compared on one scale.
    guarded, guarded_info = twoframe_stack(stack, ref, inject=error, gate=False,
                                           surface=True)
    intact, intact_info = twoframe_stack(stack, ref, surface=True)
    surface_only = compare(guarded, guarded_info, intact, intact_info)
    assert surface_only < 0.5 * damage, (surface_only, damage)

    # And the error must not have been applied silently: the entry it was
    # injected into is no longer carrying it.
    injected = gated_info["diagnostics"][pair_index]["shifts"][0]
    clean_shift = info["diagnostics"][pair_index]["shifts"][0]
    if injected is not None and clean_shift is not None:
        assert abs(injected[0] - clean_shift[0]) < 2.0, (injected, clean_shift)


def test_the_ported_gate_separates_a_correct_fit_from_a_wrong_one():
    """The gate as an instrument, on a known answer (F110 KAT 6 re-run here).

    `warpAffine` moves content by -t and every estimator in this module returns
    the warp taking REFERENCE coordinates to MOVING ones, so the correct answer
    for a +12 px content move is -12. A correct fit must verify; an 8 px error
    must be contradicted, and the statistic must read the error's own size,
    because that reading is what the repair step uses as its correction.
    """
    grey = _textured(21, 320, 420) / 255.0
    h, w = grey.shape
    moved = cv2.warpAffine(grey.astype(np.float32), np.float32([[1, 0, -12], [0, 1, 0]]),
                           (w, h), borderMode=cv2.BORDER_REFLECT)
    evidence = EdgeEvidence([grey.astype(np.float32), moved], 0,
                            np.zeros((h, w), np.float32), np.ones((h, w), bool),
                            np.ones((h, w), np.float32))
    mask = np.zeros((h, w), bool)
    mask[60:260, 80:340] = True
    indices = evidence.indices_in(mask)
    assert len(indices) >= 20

    status, statistic, _why = gate_shift(evidence, 1, indices, (-12.0, 0.0))
    assert status == "verified", (status, statistic)
    assert statistic < GATE_TOL

    status, statistic, _why = gate_shift(evidence, 1, indices, (-12.0 + 8.0, 0.0))
    assert status == "contradicted", (status, statistic)
    assert abs(statistic - 8.0) < 1.0, statistic


def test_the_full_resolution_transfer_agrees_with_its_own_analysis():
    """Above the working width the geometry is carried to native pixels by matrix
    conjugation (F107). The KAT feeds a 2x upscale as the 'natives' and asks
    whether the native result, brought back down, matches the working-scale run
    it came from — measured against this KAT's own resample floor, because no
    transfer can score below the 2x-up/area-down round trip itself.
    """
    stack, ref = _stranded_object_stack(step_px=6.0)
    h, w = stack[0].shape[:2]
    natives = [cv2.resize(f, (2 * w, 2 * h), interpolation=cv2.INTER_CUBIC)
               for f in stack]
    native, info = twoframe_fullres(natives, working_width=w, ref=ref)
    working = info["working_fused"]
    scale = info["scale"]
    assert abs(scale - 2.0) < 1e-6

    wx0, wy0, wx1, wy1 = info["working_crop"]
    nx0, ny0, nx1, ny1 = info["crop"]
    x0 = int(np.ceil(max(wx0, nx0 / scale))) + 4
    y0 = int(np.ceil(max(wy0, ny0 / scale))) + 4
    x1 = int(min(wx1, nx1 / scale)) - 4
    y1 = int(min(wy1, ny1 / scale)) - 4
    left = working[y0 - wy0:y1 - wy0, x0 - wx0:x1 - wx0].astype(np.float32)
    right = native[int(y0 * scale) - ny0:int(y1 * scale) - ny0,
                   int(x0 * scale) - nx0:int(x1 * scale) - nx0]
    down = cv2.resize(right, (left.shape[1], left.shape[0]),
                      interpolation=cv2.INTER_AREA).astype(np.float32)
    transfer = float(np.abs(down - left).max(axis=2).mean())

    floor = cv2.resize(natives[ref], (w, h), interpolation=cv2.INTER_AREA)
    floor = float(np.abs(floor[y0:y1, x0:x1].astype(np.float32)
                         - stack[ref][y0:y1, x0:x1].astype(np.float32))
                  .max(axis=2).mean())
    control = float(np.abs(stack[ref][y0:y1, x0:x1].astype(np.float32) - left)
                    .max(axis=2).mean())
    assert transfer < max(4.0 * floor, control), (transfer, floor, control)


# ---------------------------------------------------------------------------
# ROUND 3 — the same-surface precondition of the focus contest.
# ---------------------------------------------------------------------------
def _disk(image, radius):
    r = int(round(radius))
    if r < 1:
        return image
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    kernel = (xx * xx + yy * yy <= r * r).astype(np.float32)
    return cv2.filter2D(image.astype(np.float32), -1,
                        kernel / kernel.sum()).astype(np.uint8)


def _kat_scene():
    """The committed round-3 fixture, unchanged, so the two versions of the test
    are directly comparable on the four questions F112/R3 asked."""
    rng = np.random.default_rng(4)
    texture = cv2.GaussianBlur(rng.integers(0, 255, (300, 400)).astype(np.uint8),
                               (0, 0), 2)
    scene = cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR).astype(np.float32)
    inner = np.zeros(scene.shape[:2], bool)
    inner[40:-40, 40:-40] = True
    return scene, inner


def test_same_surface_is_blind_to_defocus_and_sees_a_displaced_occluder():
    """KAT for the instrument (DEVSTYLE §12.1), on known answers.

    `same_surface` decides whether two frames are looking at the same thing, and
    the round-3 fix rests on it, so it is tested against answers that are known
    by construction BEFORE it is believed on the kitchen:

      * real defocus is a DISK (PLAYBOOK §0) and must not trip it — that is the
        entire premise, and ROUND 4 replaced the global low-pass that stood in
        for it with the physics: two observations of one surface satisfy
        `m (x) disk(R_r) == r (x) disk(R_m)`, so each is convolved with the
        OTHER's disk and the defocus difference is REMOVED rather than exceeded.
        The rows that prove the retirement are 8 and 12 px, where the global
        sigma=4 version read 0.931 and 0.759 and this one reads 1.000;
      * a residual displacement inside the module's own `GATE_TOL` must not trip
        it, because the geometry already declares such a fit verified;
      * `normalize_exposure`'s residual gain (measured at most 1.9% on the
        kitchen) must not trip it;
      * an occluder that MOVED must trip it, over the strip it vacated and the
        strip it now covers, and nowhere else.

    Run with the pooling window OFF, because these are questions about the
    MATCHING; `test_same_surface_pools_its_verdict` covers the shipped verdict.
    Every number below reproduces `research/scene_model.py kat` to three decimals,
    which is the port's own known-answer test.
    """
    scene, inner = _kat_scene()
    zero = np.zeros(scene.shape[:2], np.float32)
    ones = np.ones(scene.shape[:2], np.float32)

    def agree(member, reference, r_m, r_r):
        return same_surface(member, reference, r_m, r_r, pool=1)

    # Defocus, at every radius — including the two the global low-pass failed.
    for radius in (1, 2, 4, 6, 8, 12):
        value = agree(_disk(scene, radius), scene, ones * radius, zero)[inner].mean()
        assert value > 0.97, (radius, value)

    # A displacement the geometry already tolerates, and one it does not.
    for shift in (0.5, 1.0, GATE_TOL):
        moved = cv2.warpAffine(scene, np.float32([[1, 0, shift], [0, 1, 0]]),
                               (scene.shape[1], scene.shape[0]),
                               borderMode=cv2.BORDER_REFLECT)
        assert agree(moved, scene, zero, zero)[inner].mean() > 0.97, shift
    far = cv2.warpAffine(scene, np.float32([[1, 0, 4.0], [0, 1, 0]]),
                         (scene.shape[1], scene.shape[0]),
                         borderMode=cv2.BORDER_REFLECT)
    assert agree(far, scene, zero, zero)[inner].mean() < 0.80

    # The exposure residual.
    gained = np.clip(scene * 1.019, 0, 255)
    assert agree(gained, scene, zero, zero)[inner].mean() > 0.97

    # And the thing it exists for: an occluder standing somewhere it is not.
    for step in (4, 8, 20):
        here, there = scene.copy(), scene.copy()
        here[100:200, 120:220] = (240, 240, 240)
        there[100:200, 120 + step:220 + step] = (240, 240, 240)
        agreement = agree(there, here, zero, zero)
        strip = np.zeros(scene.shape[:2], bool)
        strip[100:200, 120:120 + step] = True        # vacated
        strip[100:200, 220:220 + step] = True        # newly covered
        band = np.zeros(scene.shape[:2], bool)
        band[90:210, 110:230 + step] = True
        assert agreement[strip].mean() < 0.05, step
        assert agreement[inner & ~band].mean() > 0.95, step


def test_same_surface_pools_its_verdict_over_the_focus_operators_own_window():
    """The half of the low-pass the cross-convolution does NOT retire.

    A per-pixel level agreement is not evidence about a SURFACE: a textured
    intruder crosses the occluded surface's level at scattered pixels, and
    matching the defocus exactly makes the test see those coincidences (measured
    on the kitchen: agreement inside the F112 box 1 rose 8.9% -> 14.3% for the
    member that renders the pot in front of the bottle, and the defect came
    back). So the verdict is pooled — unanimously — over the window the FOCUS
    CONTEST itself is decided on, `focus.content_aware_energies`'s `smooth_ksize`.

    Two known answers: pooling may not disturb a verdict that is uniform over its
    window (pure defocus still reads 1.000), and it must remove an agreement that
    cannot be supported over one (isolated matching pixels inside a region that
    disagrees).
    """
    scene, inner = _kat_scene()
    zero = np.zeros(scene.shape[:2], np.float32)
    ones = np.ones(scene.shape[:2], np.float32)
    assert SURFACE_POOL >= 3

    # Uniform verdicts survive pooling untouched.
    for radius in (2, 8):
        assert same_surface(_disk(scene, radius), scene,
                            ones * radius, zero)[inner].mean() > 0.97, radius

    # An intruder made of the scene's own texture, level-shifted so that its
    # levels still CROSS the surface it replaced: unpooled, the crossings are
    # admitted; pooled, none of them is supported by its own neighbourhood.
    here = scene.copy()
    there = scene.copy()
    there[100:200, 120:220] = np.roll(scene, 137, axis=1)[100:200, 120:220]
    patch = np.zeros(scene.shape[:2], bool)
    patch[100:200, 120:220] = True
    loose = same_surface(there, here, zero, zero, pool=1)[patch].mean()
    pooled = same_surface(there, here, zero, zero)[patch].mean()
    assert loose > 0.05, loose
    assert pooled < 0.2 * loose, (pooled, loose)


def test_a_withdrawn_sharp_member_falls_back_to_a_present_one_not_the_reference():
    """F111's designated repair, on a known answer (ROUND 4).

    The fallback chain is trinary: the sharp member, then a PRESENT member the
    appearance evidence licenses, and only then the reference's defocused
    stand-in. The synthetic case puts all three in one frame: the reference is
    the blurriest observation of the surface, member A is the sharpest, member B
    is in between, and F82's refusal is forced to withdraw A over the whole
    field. B must win the vacated pixels, not the reference.

    Scored where it can only be answered one way — the deviation from the SHARP
    source, which A and B share and the reference does not.
    """
    rng = np.random.default_rng(31)
    scene = cv2.GaussianBlur(rng.integers(0, 255, (200, 260)).astype(np.float32),
                             (0, 0), 1.2)
    colour = lambda g: cv2.cvtColor(np.clip(g, 0, 255).astype(np.uint8),
                                    cv2.COLOR_GRAY2BGR)
    sharp = colour(scene)
    middle = colour(_disk_blur(scene, 2))
    reference = colour(_disk_blur(scene, 6))
    zero = np.zeros(scene.shape, np.float32)
    radius_ref = np.full(scene.shape, 6.0, np.float32)

    # The evidence: both members observe the reference's surface once the
    # defocus is matched, and the sharper one is the sharper one.
    vote_a = pooled_majority(surface_agreement(sharp, reference, zero, radius_ref))
    vote_b = pooled_majority(surface_agreement(middle, reference,
                                               np.full(scene.shape, 2.0, np.float32),
                                               radius_ref))
    inner = np.zeros(scene.shape, bool)
    inner[30:-30, 30:-30] = True
    assert vote_a[inner].mean() > 0.9, vote_a[inner].mean()
    assert vote_b[inner].mean() > 0.9, vote_b[inner].mean()

    # And the preference the chain must express: with A withdrawn, the pixel is
    # served by B (present, defocused, correctly placed) rather than by the
    # reference — which is what the fused result must resemble.
    # A member that is present but shows a DIFFERENT SURFACE must not be
    # licensed by the same gate — the clause that stops a defocused wrong
    # surface entering. (A displaced copy of the SAME fine texture is a
    # different question and reads 0.61 here: an appearance test cannot
    # separate two surfaces that look alike, which is the recorded limit of
    # this instrument and why F82's geometric check is not removed.)
    intruder = middle.copy()
    intruder[70:130, 40:200] = 240
    vote_bad = pooled_majority(surface_agreement(
        intruder, reference, np.full(scene.shape, 2.0, np.float32), radius_ref))
    patch = np.zeros(scene.shape, bool)
    patch[70:130, 40:200] = True
    assert vote_bad[patch].mean() < 0.05, vote_bad[patch].mean()

    withdrawn = np.zeros(scene.shape, bool)
    withdrawn[60:140, :] = True                  # F82 refuses A on this band
    usable = [~withdrawn, withdrawn & vote_b]
    usable.append(~usable[0] & ~usable[1])
    fused = fuse_coherent([sharp, middle, reference], harden=0.5, usable=usable)
    band = withdrawn & inner
    to_middle = float(np.abs(fused[band].astype(float)
                             - middle[band].astype(float)).mean())
    to_reference = float(np.abs(fused[band].astype(float)
                                - reference[band].astype(float)).mean())
    assert to_middle < to_reference, (to_middle, to_reference)


def _occluded_pair_stack(step_px=18.0, frames=5, h=300, w=420):
    """A near square sharp AT THE REFERENCE, over a background sharp at the end.

    This is the kitchen's own geometry in miniature and it reproduces the user's
    flaw class directly: the last frame is the only one that can supply the sharp
    background, and in it the near square has swung `step_px` sideways, so its
    background is EXPOSED where the reference says the square is, and its own
    copy of the square stands where the reference says background is. A fusion
    that soft-mixes, or that lets the focus contest decide at an occlusion, puts
    background inside the square and a second square outside it.
    """
    ref = frames // 2
    rng = np.random.default_rng(17)
    background = cv2.GaussianBlur(
        rng.integers(0, 255, (h, w)).astype(np.float32), (0, 0), 1.5)
    square = cv2.GaussianBlur(
        rng.integers(0, 255, (120, 120)).astype(np.float32), (0, 0), 1.0) * 0.4 + 150
    stack, masks = [], []
    for k in range(frames):
        sharp = background.copy()
        x0 = 150 + int(round(step_px * (k - ref)))
        sharp[90:210, x0:x0 + 120] = square
        mask = np.zeros((h, w), bool)
        mask[90:210, x0:x0 + 120] = True
        near = _disk_blur(sharp, 2.2 * abs(k - ref))            # square: sharp at ref
        far = _disk_blur(sharp, 2.2 * abs(k - (frames - 1)))    # background: sharp last
        frame = np.where(mask, near, far)
        stack.append(cv2.cvtColor(np.clip(frame, 0, 255).astype(np.uint8),
                                  cv2.COLOR_GRAY2BGR))
        masks.append(mask)
    return stack, ref, masks


def test_a_member_misregistered_for_a_region_does_not_alias_into_the_fused_pair():
    """The user's round-3 flaw class, synthesized: an occluder that moved.

    Two things must hold inside the pair, and BOTH failed before this round:
      1. ORDERING — where the reference shows the near square, the composite must
         show the square, not the background the other member sees through it.
      2. NO SECOND COPY — where the reference shows background, the composite must
         show background, not the other member's displaced copy of the square.
    Scored against the reference frame, which is sharp on the square and is the
    authority on what is VISIBLE (it is the composite's own geometry).
    """
    stack, ref, masks = _occluded_pair_stack()
    fused, info = twoframe_stack(stack, ref)
    x0, y0, x1, y1 = info["crop"]
    reference = stack[ref][y0:y1, x0:x1].astype(np.float32)
    square = masks[ref][y0:y1, x0:x1]
    ghost = masks[-1][y0:y1, x0:x1] & ~square      # where the LAST frame's copy is
    delta = np.abs(fused.astype(np.float32) - reference).max(axis=2)

    without, _ = twoframe_stack(stack, ref, surface=False)
    loose = np.abs(without.astype(np.float32) - reference).max(axis=2)

    # The bar is the DEFECT'S OWN SIZE, measured on the same fixture with the
    # precondition off: an absolute threshold would be measuring the fixture's
    # blur, not the flaw (§12.2). Round 3 scoped these clauses against the
    # composite's ORDINARY deviation from the reference instead, and round 4
    # retired that scope because it went degenerate — the physical test refuses
    # enough that the composite sits ~1.6 levels from the reference everywhere,
    # so "ordinary" and "defect-free" became the same number (1.63 vs 1.72) and
    # the comparison stopped being able to fail for the right reason.
    ordinary = float(delta[~square & ~ghost].mean())
    # 1. the square's own interior is not showing the background behind it
    assert delta[square].mean() < 0.25 * loose[square].mean(), (
        delta[square].mean(), loose[square].mean())
    # 2. no second copy of it outside its own silhouette — and there the bar is
    #    still ordinary, because a ghost is content that does not belong at all.
    assert delta[ghost].mean() < 1.5 * ordinary, (delta[ghost].mean(), ordinary)
    # and the precondition is what buys it: without it the ghost is gross
    assert loose[ghost].mean() > 3.0 * delta[ghost].mean(), (loose[ghost].mean(),
                                                             delta[ghost].mean())
    assert loose[square].mean() > 1.5 * delta[square].mean()
