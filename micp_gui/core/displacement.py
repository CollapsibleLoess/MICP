import numpy as np
from scipy.stats import linregress

from .models import MICPParams, MICPResult


def calc_displacement_pressure(params: MICPParams, result: MICPResult):
    if len(result.cum_intrusion) < 2 or len(result.intrusion_pressure_mpa) < 2:
        return

    cum_max = np.max(result.cum_intrusion)
    if cum_max <= 0:
        return

    saturation = result.cum_intrusion / cum_max * 100
    log_p = np.log10(result.intrusion_pressure_mpa)

    valid = np.isfinite(log_p) & (result.intrusion_pressure_mpa > 0) & (saturation >= 0)
    if np.sum(valid) < 2:
        result.displacement_pressure = float(result.intrusion_pressure_mpa[0])
        return

    sat_valid = saturation[valid]
    log_p_valid = log_p[valid]

    sort_idx = np.argsort(sat_valid)
    sat_sorted = sat_valid[sort_idx]
    log_p_sorted = log_p_valid[sort_idx]

    sat_low = params.disp_p_sat_min if params.disp_p_sat_min > 0 else max(float(sat_sorted[0]), 0.1)
    sat_high = params.disp_p_sat_max if params.disp_p_sat_max > 0 else min(float(sat_sorted[-1]), 25.0)
    if sat_high <= sat_low:
        sat_low = max(float(sat_sorted[0]), 0.1)
        sat_high = min(float(sat_sorted[-1]), 25.0)

    fit_mask = (sat_sorted >= sat_low) & (sat_sorted <= sat_high)
    if np.sum(fit_mask) < 2:
        sat_low = max(float(sat_sorted[0]), 0.1)
        sat_high = min(float(sat_sorted[-1]), 25.0)
        fit_mask = (sat_sorted >= sat_low) & (sat_sorted <= sat_high)
        if np.sum(fit_mask) < 2:
            result.displacement_pressure = float(result.intrusion_pressure_mpa[0])
            return

    sat_fit = sat_sorted[fit_mask]
    log_p_fit = log_p_sorted[fit_mask]

    slope, intercept, r_value, _, _ = linregress(sat_fit, log_p_fit)

    result._disp_p_fit_data = {
        'sat_fit': sat_fit.tolist(),
        'log_p_fit': log_p_fit.tolist(),
        'sat_low': float(sat_low),
        'sat_high': float(sat_high),
        'slope': float(slope),
        'intercept': float(intercept),
        'r_squared': float(r_value ** 2),
        'sat_interp': np.linspace(0, sat_high, 100).tolist(),
    }
    result._disp_p_fit_data['log_p_interp'] = (slope * np.array(result._disp_p_fit_data['sat_interp']) + intercept).tolist()

    result.displacement_pressure = 10 ** intercept if slope > 0 else float(result.intrusion_pressure_mpa[0])

    sigma = params.surface_tension
    theta = np.radians(params.contact_angle)
    result.max_pore_diameter_um = -2 * sigma * np.cos(theta) / result.displacement_pressure * 2 if result.displacement_pressure > 0 else 0

    cal_porosity_pct = result.cal_porosity if result.cal_porosity > 0 else result._he_porosity
    he_por = params.he_porosity_override if params.he_porosity_override > 0 else (result._he_porosity if result._he_porosity > 0 else result.cal_porosity)
    if he_por > 0:
        result.intrusion_saturation = cal_porosity_pct / he_por * 100

    if len(result.cum_extrusion) > 0 and cum_max > 0:
        valid_ext = np.isfinite(result.cum_extrusion)
        if np.any(valid_ext):
            min_remaining = float(np.min(result.cum_extrusion[valid_ext]))
            if 0 <= min_remaining < cum_max:
                result.efficiency = (1 - min_remaining / cum_max) * 100
