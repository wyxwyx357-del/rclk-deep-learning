"""Read formal per-frame geometry signals with frozen topology masks.

The 35 Hz grid is a computational representation. Interpolated values are
tagged and never counted as new source observations.
"""

from pathlib import Path
import numpy as np

TARGET_HZ = 35
POINTS = np.linspace(0, 1, 8, dtype=np.float64)
EDGES = (POINTS[:-1] + POINTS[1:]) / 2


def _one(folder, pattern):
    paths = list(folder.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f'Expected one {pattern} in {folder}, found {len(paths)}')
    return paths[0]


def _spatial(values, source_x, target_x):
    """Interpolate only between adjacent finite source positions."""
    source_x = np.asarray(source_x, dtype=np.float64)
    values = np.asarray(values, dtype=np.float32)
    assert values.ndim == 3 and values.shape[2] == len(source_x)
    out = np.full(values.shape[:2] + (len(target_x),), np.nan, dtype=np.float32)
    observed = np.zeros(out.shape, dtype=bool)
    left = np.full(len(target_x), -1, dtype=np.int32)
    right = np.full(len(target_x), -1, dtype=np.int32)
    weight = np.full(len(target_x), np.nan, dtype=np.float32)
    for j, x in enumerate(target_x):
        i = int(np.searchsorted(source_x, x))
        exact = i < len(source_x) and abs(source_x[i] - x) < 1e-6
        if not exact and i > 0 and abs(source_x[i - 1] - x) < 1e-6:
            i -= 1
            exact = True
        if exact:
            out[:, :, j] = values[:, :, i]
            observed[:, :, j] = np.isfinite(values[:, :, i])
            left[j] = right[j] = i
            weight[j] = 0
        elif 0 < i < len(source_x):
            a, b = values[:, :, i - 1], values[:, :, i]
            good = np.isfinite(a) & np.isfinite(b)
            w = float((x - source_x[i - 1]) / (source_x[i] - source_x[i - 1]))
            out[:, :, j] = np.where(good, (1 - w) * a + w * b, np.nan)
            left[j], right[j], weight[j] = i - 1, i, w
    return out, observed, {'left': left, 'right': right, 'right_weight': weight}


def _timeline(frame_count, fps):
    source_pos = np.arange(int(np.floor((frame_count - 1) * TARGET_HZ / fps + 1e-7)) + 1,
                           dtype=np.float64) * fps / TARGET_HZ
    nearest = np.rint(source_pos).astype(np.int32)
    exact = np.abs(source_pos - nearest) < 1e-6
    left = np.where(exact, nearest, np.floor(source_pos).astype(np.int32))
    right = np.where(exact, nearest, left + 1)
    assert left.min() >= 0 and right.max() < frame_count
    weight = np.where(exact, 0, source_pos - left).astype(np.float32)
    assert np.all((weight >= 0) & (weight < 1))
    return source_pos / fps, left, right, weight, exact


def _temporal(spatial, spatial_observed, left, right, weight, exact):
    a, b = spatial[left], spatial[right]
    w = weight[:, None, None]
    good = np.isfinite(a) & np.isfinite(b)
    out = np.where(good, (1 - w) * a + w * b, np.nan).astype(np.float32)
    observed = spatial_observed[left] & exact[:, None, None]
    return out, np.isfinite(out), observed


def read_case(case_id, root, include_k=False):
    """Return 35 Hz arrays, masks, and reproducible source coordinates.

    Array order is R(2x8), C(1x8), L(2x7), then optional K(2x8).
    No clinical outcome is read here. Missing measurements remain NaN.
    """
    patient = Path(root) / case_id
    core = patient / '02_追踪质控与形变'
    with np.load(_one(core / '10_rsr_fourway', '*_rsr_fourway_v1.npz'), allow_pickle=False) as z:
        r = z['current_qc_radial_strain_rate_s'].copy()
        x = z['normalized_cervix_to_fundus_position'].astype(np.float64)
        fps = float(z['fps'])
        lineage = str(z['formal_qc_lineage_id'])
    frames, walls, positions = r.shape
    assert walls == 2 and len(x) == positions and np.all(np.diff(x) > 0)
    assert np.isfinite(fps) and 0 < fps <= TARGET_HZ
    with np.load(_one(core / '11_anatomical_deformation', '*_anatomical_deformation_v1.npz'), allow_pickle=False) as z:
        c = z['qc_cavity_width_strain_rate_s'].copy()
        l = z['qc_longitudinal_wall_strain_rate_s'].copy()
        assert str(z['formal_qc_lineage_id']) == lineage
        assert float(z['fps']) == fps
        assert np.array_equal(z['normalized_cervix_to_fundus_position'], x)
    assert c.shape == (frames, positions) and l.shape == (frames, 2, positions - 1)
    with np.load(_one(core / '05_radial_tracking_corrected', '*_p3_image_boundary_evidence_v2.npz'), allow_pickle=False) as z:
        risk = z['p3_topology_risk_frame'].copy()
    with np.load(_one(core / '07_artifact_qc', '*_artifact_qc_v2_1.npz'), allow_pickle=False) as z:
        assert np.array_equal(risk, z['p3_topology_risk'])
        assert np.array_equal(r, z['qc_radial_strain_rate_grade3_blank_s'], equal_nan=True)
    assert risk.shape == (frames,)
    native = {'R': r, 'C': c[:, None, :], 'L': l}
    source_x = {'R': x, 'C': x, 'L': (x[:-1] + x[1:]) / 2}
    target_x = {'R': POINTS, 'C': POINTS, 'L': EDGES}
    if include_k:
        with np.load(_one(patient / '03_DICOM物理标定', '*_dicom_curvature_v1.npz'), allow_pickle=False) as z:
            k = z['qc_wall_curvature_change_rate_mm_inv_s'].copy()
            assert k.shape == r.shape and str(z['formal_qc_lineage_id']) == lineage
        assert np.isfinite(k).any(), case_id
        native['K'], source_x['K'], target_x['K'] = k, x, POINTS
    times, left, right, weight, exact = _timeline(frames, fps)
    values, valid, observed, spatial_provenance, formal_medians = {}, {}, {}, {}, {}
    for name, arr in native.items():
        arr[risk] = np.nan
        formal_medians[name] = float(np.nanmedian(np.abs(arr)))
        spatial, spatial_seen, spatial_provenance[name] = _spatial(arr, source_x[name], target_x[name])
        values[name], valid[name], observed[name] = _temporal(
            spatial, spatial_seen, left, right, weight, exact)
    return {
        'case_id': case_id, 'fps': fps, 'native_frames': frames,
        'native_positions': positions, 'topology_risk': risk,
        'formal_medians': formal_medians, 'time_s': times,
        'time_left_frame': left, 'time_right_frame': right,
        'time_right_weight': weight, 'time_observed': exact,
        'spatial_provenance': spatial_provenance,
        'values': values, 'valid': valid, 'observed': observed,
        'formal_qc_lineage_id': lineage,
    }


def model_channels(record, include_k=False):
    """Flatten anatomical dimensions into channels without hiding missingness."""
    names = ('R', 'C', 'L', 'K') if include_k else ('R', 'C', 'L')
    arrays = [record['values'][name].reshape(len(record['time_s']), -1) for name in names]
    masks = [record['valid'][name].reshape(len(record['time_s']), -1) for name in names]
    value = np.concatenate(arrays, axis=1)
    mask = np.concatenate(masks, axis=1)
    assert value.shape[1] == (54 if include_k else 38)
    return value, mask
