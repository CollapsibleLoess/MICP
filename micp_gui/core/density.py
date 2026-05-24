import numpy as np

from .models import MICPData, MICPParams, MICPResult


def calc_density_porosity(data: MICPData, result: MICPResult):
    m = data.sample_mass
    mp = data.penetrometer_mass
    ma = data.assembly_mass
    vp = data.penetrometer_volume

    if m <= 0 or vp <= 0:
        return

    denom_base = vp - (ma - mp - m) / 13.5939
    if denom_base <= 0:
        return

    result.bulk_density = m / denom_base

    result.max_intrusion_vol = float(np.max(result.cum_intrusion)) if len(result.cum_intrusion) > 0 else 0
    intrusion_vol_cm3 = result.max_intrusion_vol * m

    denom_skeletal = denom_base - intrusion_vol_cm3
    if denom_skeletal > 0:
        result.skeletal_density = m / denom_skeletal
    else:
        result.skeletal_density = result.bulk_density

    if result.bulk_density > 0 and result.max_intrusion_vol > 0:
        result.cal_porosity = round(result.max_intrusion_vol * result.bulk_density * 100, 4)
