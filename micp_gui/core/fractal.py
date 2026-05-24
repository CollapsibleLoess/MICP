import numpy as np
from scipy.stats import linregress

from .models import MICPResult, MICPParams


def calc_fractal(params: MICPParams, result: MICPResult):
    if len(result.cum_intrusion) < 3 or len(result.intrusion_pressure_mpa) < 3:
        return

    cum_max = np.max(result.cum_intrusion)
    if cum_max <= 0:
        return

    result.logP = np.log10(result.intrusion_pressure_mpa)
    sat_ratio = 1 - result.cum_intrusion / cum_max
    sat_ratio = np.clip(sat_ratio, 1e-10, 1)
    result.log1_S = np.log10(sat_ratio)

    valid = np.isfinite(result.logP) & np.isfinite(result.log1_S) & (result.intrusion_pressure_mpa > 0)
    if np.sum(valid) < 3:
        return

    result.fractal_slopes = []
    result.fractal_dimensions = []

    log_p_valid = result.logP[valid]
    log_1s_valid = result.log1_S[valid]

    bp = params.frac_seg_breakpoint if params.frac_seg_breakpoint > 0 else 2.0

    mask1 = log_p_valid <= bp
    mask2 = log_p_valid > bp

    sort_idx = np.argsort(log_p_valid)
    result._frac_seg_data = {
        'breakpoint': float(bp),
        'logP_sorted': log_p_valid[sort_idx].tolist(),
        'log1_S_sorted': log_1s_valid[sort_idx].tolist(),
        'seg_masks': [mask1.tolist(), mask2.tolist()],
    }

    for mask in (mask1, mask2):
        if np.sum(mask) >= 2:
            slope, _, _, _, _ = linregress(log_p_valid[mask], log_1s_valid[mask])
            result.fractal_slopes.append(slope)
            result.fractal_dimensions.append(3 + slope)
        else:
            result.fractal_slopes.append(0)
            result.fractal_dimensions.append(0)
