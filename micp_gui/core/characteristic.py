import numpy as np

from .models import MICPParams, MICPResult


def calc_characteristic_params(params: MICPParams, result: MICPResult):
    if len(result.cum_intrusion) < 2 or len(result.pore_throat_diameter_nm) < 2:
        return

    cum_max = np.max(result.cum_intrusion)
    if cum_max <= 0:
        return

    saturation = result.cum_intrusion / cum_max * 100
    d_nm = result.pore_throat_diameter_nm
    d_um = d_nm * 0.001

    d_ref = 1e6
    psi = -np.log2(np.maximum(d_nm, 1e-30) / d_ref)

    def interp_psi(sat_target):
        return np.interp(sat_target, saturation, psi) if len(saturation) >= 2 else 0

    psi_84 = interp_psi(84)
    psi_16 = interp_psi(16)
    psi_95 = interp_psi(95)
    psi_5 = interp_psi(5)
    psi_50 = interp_psi(50)
    psi_75 = interp_psi(75)
    psi_25 = interp_psi(25)

    result.sorting_coefficient = (psi_84 - psi_16) / 4 + (psi_95 - psi_5) / 6.6

    denom1 = 2 * (psi_84 - psi_16) if abs(psi_84 - psi_16) > 1e-10 else 1e-10
    denom2 = 2 * (psi_95 - psi_5) if abs(psi_95 - psi_5) > 1e-10 else 1e-10
    result.skewness = (psi_84 + psi_16 - 2 * psi_50) / denom1 + (psi_95 + psi_5 - 2 * psi_50) / denom2

    denom3 = 2.44 * (psi_75 - psi_25) if abs(psi_75 - psi_25) > 1e-10 else 1e-10
    result.kurtosis = (psi_95 - psi_5) / denom3

    result.mean_radius = (psi_16 + psi_84 + psi_50) / 3
    result.structure_coefficient = 2 ** result.mean_radius if abs(result.sorting_coefficient) > 1e-10 else 0
    result.relative_sorting_coeff = result.sorting_coefficient / result.mean_radius if abs(result.mean_radius) > 1e-10 else 0

    result.median_pressure = np.interp(0.5, result.cum_intrusion / cum_max, result.intrusion_pressure_mpa)
    sigma = params.surface_tension
    theta_rad = np.radians(params.contact_angle)
    result.median_diameter_um = -2 * sigma * np.cos(theta_rad) / result.median_pressure * 2 * 1000 * 0.001 if result.median_pressure > 0 else 0

    result.pore_volume = float(cum_max)

    _calc_median_pore_diameters(result, params)
    _calc_avg_pore_diameter(result)
    _calc_conductivity_tortuosity(result, params, cum_max)
    _calc_breakthrough_ratio(result, cum_max)


def _calc_median_pore_diameters(result: MICPResult, params: MICPParams):
    if len(result.pore_throat_diameter_nm) < 2:
        return

    d_nm = result.pore_throat_diameter_nm
    sort_idx = np.argsort(d_nm)[::-1]
    d_sorted = d_nm[sort_idx]
    cum_sorted = result.cum_intrusion[sort_idx]

    cum_max = np.max(cum_sorted)
    if cum_max <= 0:
        return

    result.median_pore_diameter_volume_nm = float(np.interp(0.5, cum_sorted / cum_max, d_sorted))

    if hasattr(result, 'incremental_pore_area') and len(result.incremental_pore_area) > 0:
        inc_area = result.incremental_pore_area
        if len(inc_area) >= 2:
            if len(inc_area) == len(d_nm) - 1:
                inc_area_full = np.zeros(len(d_nm))
                inc_area_full[1:] = np.maximum(inc_area, 0)
            else:
                inc_area_full = np.maximum(inc_area, 0)

            if len(inc_area_full) == len(d_nm):
                inc_area_sorted = inc_area_full[sort_idx]
                cum_area = np.cumsum(inc_area_sorted)
                area_max = cum_area[-1] if cum_area[-1] > 0 else 0
                if area_max > 0:
                    result.median_pore_diameter_area_nm = float(np.interp(0.5, cum_area / area_max, d_sorted))
                    result.total_pore_area = float(area_max)
                elif result.specific_surface_area > 0:
                    result.total_pore_area = result.specific_surface_area


def _calc_avg_pore_diameter(result: MICPResult):
    if result.pore_volume > 0 and result.total_pore_area > 0:
        result.avg_pore_diameter_nm = 4 * result.pore_volume / result.total_pore_area * 1000
    elif result.pore_volume > 0 and result.specific_surface_area > 0:
        result.avg_pore_diameter_nm = 4 * result.pore_volume / result.specific_surface_area * 1000


def _calc_conductivity_tortuosity(result: MICPResult, params: MICPParams, cum_max: float):
    if result.max_pore_diameter_um <= 0 or result.median_diameter_um <= 0:
        return

    d_max_nm = result.max_pore_diameter_um * 1000.0
    d_med_nm = result.median_diameter_um * 1000.0

    result.characteristic_length_nm = d_max_nm

    if d_max_nm > 0:
        result.conductivity_formation_factor = (d_med_nm / d_max_nm) ** 2

    if result.conductivity_formation_factor > 0:
        result.tortuosity_factor = 1.0 / (result.conductivity_formation_factor ** 0.5)
        porosity_frac = result.cal_porosity / 100.0 if result.cal_porosity > 0 else result._he_porosity / 100.0
        if porosity_frac > 0:
            result.tortuosity = result.tortuosity_factor / porosity_frac

    if hasattr(result, 'fractal_dimensions') and len(result.fractal_dimensions) > 0:
        result.percolation_fractal_dimension = result.fractal_dimensions[0]


def _calc_breakthrough_ratio(result: MICPResult, cum_max: float):
    if cum_max <= 0:
        return

    avg_p = np.mean(result.intrusion_pressure_mpa) if len(result.intrusion_pressure_mpa) > 0 else 0
    if result.displacement_pressure > 0 and avg_p > 0:
        result.breakthrough_pressure_ratio = avg_p / result.displacement_pressure
