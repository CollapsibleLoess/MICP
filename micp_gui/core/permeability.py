import numpy as np

from .models import MICPResult


def calc_permeability(result: MICPResult):
    if len(result.pore_throat_diameter_nm) < 3 or len(result.cum_intrusion) < 3:
        return

    cum_max = np.max(result.cum_intrusion)
    if cum_max <= 0:
        return

    d_nm = result.pore_throat_diameter_nm
    saturation = result.cum_intrusion / cum_max

    _swanson(result, d_nm, saturation, 1, 10, target='permeability_10')
    _swanson(result, d_nm, saturation, 10, 100, target='permeability_413')


def _swanson(result: MICPResult, d_nm, saturation, d_min_um, d_max_um, target: str):
    d_um = d_nm * 0.001
    mask = (d_um >= d_min_um) & (d_um <= d_max_um)

    if np.sum(mask) < 2:
        setattr(result, target, 0.0)
        return

    d_f = d_um[mask]
    s_f = saturation[mask]

    product = s_f * d_f ** 3
    max_idx = np.argmax(product)

    lc = d_f[max_idx]
    vc = s_f[max_idx]
    vtot = float(np.max(s_f))
    lmax = float(np.max(d_f))

    cal_porosity_pct = result.cal_porosity if result.cal_porosity > 0 else result._he_porosity

    if lc > 0 and vtot > 0:
        setattr(result, target, (1 / 89) * (lmax) ** 2 * (lmax / lc) * cal_porosity_pct * (vc / vtot) * 1e-16)
    else:
        setattr(result, target, 0.0)
