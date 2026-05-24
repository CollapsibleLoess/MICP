import numpy as np

from .models import MICPParams, MICPResult


def calc_pore_throat_diameter(params: MICPParams, result: MICPResult):
    if len(result.intrusion_pressure_mpa) == 0:
        return

    sigma = params.surface_tension
    theta = np.radians(params.contact_angle)

    result.pore_throat_diameter_nm = -2 * sigma * np.cos(theta) / result.intrusion_pressure_mpa * 1000
    result.pore_throat_diameter_um = result.pore_throat_diameter_nm * 0.001


def calc_pore_throat_ratio(params: MICPParams, result: MICPResult):
    if len(result.cum_intrusion) < 3 or len(result.cum_extrusion) < 3:
        return
    if len(result.intrusion_pressure_mpa) < 2 or len(result.extrusion_pressure_mpa) < 2:
        return

    cum_max = np.max(result.cum_intrusion)
    if cum_max <= 0:
        return

    sigma = params.surface_tension
    theta = np.radians(params.contact_angle)
    const = -4 * sigma * np.cos(theta) * 1000

    int_sat = result.cum_intrusion / cum_max * 100
    int_p = result.intrusion_pressure_mpa
    ext_sat = result.cum_extrusion / cum_max * 100
    ext_p = result.extrusion_pressure_mpa

    sat_min = max(float(np.min(ext_sat)), 0.5)
    sat_max = min(float(np.max(ext_sat)), 99.5)
    if sat_max <= sat_min:
        return

    sat_targets = np.linspace(sat_min, sat_max, 50)

    int_p_interp = np.interp(sat_targets, int_sat, int_p)
    ext_p_interp = np.interp(sat_targets, ext_sat, ext_p)

    d_int_nm = np.where(int_p_interp > 0, const / int_p_interp, np.nan)
    d_ext_nm = np.where(ext_p_interp > 0, const / ext_p_interp, np.nan)

    valid = np.isfinite(d_int_nm) & np.isfinite(d_ext_nm) & (d_ext_nm > 0)
    if np.sum(valid) < 2:
        return

    ratios = d_ext_nm[valid] / d_int_nm[valid]
    sat_valid = sat_targets[valid]

    result.pore_throat_ratio_data = {
        'threshold': sat_valid.tolist(),
        'pore_throat_ratio': ratios.tolist(),
    }
