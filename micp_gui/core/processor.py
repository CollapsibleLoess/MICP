import numpy as np
from scipy.ndimage import uniform_filter1d

from .models import MICPData, MICPParams, MICPResult
from .density import calc_density_porosity
from .pore_throat import calc_pore_throat_diameter, calc_pore_throat_ratio
from .psd import calc_incremental, calc_psd
from .characteristic import calc_characteristic_params
from .displacement import calc_displacement_pressure
from .fractal import calc_fractal
from .multifractal import calc_multifractal
from .permeability import calc_permeability
from .surface_area import calc_specific_surface_area
from .interpolate import interpolate_fixed_intrusion, interpolate_fixed_extrusion
from .correction import correct_micp_data


class MICPProcessor:
    def __init__(self):
        self.data = MICPData()
        self.params = MICPParams()
        self.result = MICPResult()

    def load(self, file_path: str) -> 'MICPProcessor':
        from .loader import load_from_excel, load_contact_angle
        self.data = load_from_excel(file_path)
        self.params.contact_angle = load_contact_angle(file_path)
        return self

    def process(self) -> MICPResult:
        psia_to_mpa = 1.0 / 145.036

        self.result = MICPResult()
        self.result.attach_metadata(self.data)

        enforce_mono = self.params.enforce_monotonic

        p_int, cum_int, p_ext, cum_ext = correct_micp_data(self.data, enforce_mono)

        self.result.intrusion_pressure_mpa = p_int * psia_to_mpa
        self.result.cum_intrusion = cum_int.copy()

        if self.params.use_fixed_interpolation:
            p_min_psia = self.params.interp_p_min if self.params.interp_p_min > 0 else None
            p_max_psia = self.params.interp_p_max if self.params.interp_p_max > 0 else None
            self.result.intrusion_pressure_mpa, self.result.cum_intrusion = interpolate_fixed_intrusion(
                self.result.intrusion_pressure_mpa,
                self.result.cum_intrusion,
                enforce_mono=enforce_mono,
                p_min_psia=p_min_psia,
                p_max_psia=p_max_psia
            )

        w = self.params.smoothing_window
        if w > 1 and len(self.result.cum_intrusion) >= w:
            self.result.cum_intrusion = uniform_filter1d(
                self.result.cum_intrusion, size=w, mode='nearest'
            )
            self.result.cum_intrusion = np.maximum(self.result.cum_intrusion, 0)

        self.result.n_intrusion_points = len(self.result.intrusion_pressure_mpa)

        if len(p_ext) > 0:
            self.result.extrusion_pressure_mpa = p_ext * psia_to_mpa
            self.result.cum_extrusion = cum_ext.copy()

            if self.params.use_fixed_interpolation:
                self.result.extrusion_pressure_mpa, self.result.cum_extrusion = interpolate_fixed_extrusion(
                    self.result.extrusion_pressure_mpa,
                    self.result.cum_extrusion,
                    enforce_mono=enforce_mono
                )

            if w > 1 and len(self.result.cum_extrusion) >= w:
                self.result.cum_extrusion = uniform_filter1d(
                    self.result.cum_extrusion, size=w, mode='nearest'
                )
                self.result.cum_extrusion = np.maximum(self.result.cum_extrusion, 0)

            self.result.n_withdrawal_points = len(self.result.extrusion_pressure_mpa)

        calc_density_porosity(self.data, self.result)
        calc_pore_throat_diameter(self.params, self.result)
        calc_incremental(self.result)
        calc_psd(self.result)
        calc_displacement_pressure(self.params, self.result)
        calc_fractal(self.params, self.result)
        calc_multifractal(self.result)
        calc_permeability(self.result)
        calc_specific_surface_area(self.result)
        calc_characteristic_params(self.params, self.result)
        calc_pore_throat_ratio(self.params, self.result)

        return self.result
