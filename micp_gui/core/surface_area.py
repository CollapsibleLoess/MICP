import numpy as np

from .models import MICPResult


def calc_specific_surface_area(result: MICPResult):
    if len(result.pore_throat_diameter_nm) < 3 or len(result.incremental_intrusion) < 3:
        return

    d_nm = result.pore_throat_diameter_nm
    inc = result.incremental_intrusion

    d_avg = (d_nm[:-1] + d_nm[1:]) / 2
    result.incremental_pore_area = 2 * inc[1:] / (d_avg * 0.001)

    positive = result.incremental_pore_area[result.incremental_pore_area > 0]
    result.specific_surface_area = float(np.sum(positive)) if len(positive) > 0 else 0.0
