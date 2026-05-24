from dataclasses import dataclass, field
import numpy as np


@dataclass
class MICPParams:
    contact_angle: float = 130.0
    surface_tension: float = 0.475
    mercury_injection_pressure: float = 0.0
    mercury_end_pressure: float = 0.0
    smoothing_window: int = 1
    enforce_monotonic: bool = False
    use_fixed_interpolation: bool = True
    interp_p_min: float = 0.5
    interp_p_max: float = 60000.0
    disp_p_sat_min: float = 0.0
    disp_p_sat_max: float = 0.0
    frac_seg_breakpoint: float = 0.0
    he_porosity_override: float = 0.0


@dataclass
class MICPData:
    sample_name: str = ""
    sample_mass: float = 0.0
    penetrometer_mass: float = 0.0
    assembly_mass: float = 0.0
    penetrometer_volume: float = 0.0
    porosity: float = 0.0
    lp_analysis_time: str = ""
    hp_analysis_time: str = ""

    intrusion_pressure_psia: np.ndarray = field(default_factory=lambda: np.array([]))
    intrusion_volume_mlg: np.ndarray = field(default_factory=lambda: np.array([]))
    intrusion_cumulative_mlg: np.ndarray = field(default_factory=lambda: np.array([]))
    extrusion_pressure_psia: np.ndarray = field(default_factory=lambda: np.array([]))
    extrusion_volume_mlg: np.ndarray = field(default_factory=lambda: np.array([]))
    extrusion_cumulative_mlg: np.ndarray = field(default_factory=lambda: np.array([]))


@dataclass
class MICPResult:
    intrusion_pressure_mpa: np.ndarray = field(default_factory=lambda: np.array([]))
    cum_intrusion: np.ndarray = field(default_factory=lambda: np.array([]))
    extrusion_pressure_mpa: np.ndarray = field(default_factory=lambda: np.array([]))
    cum_extrusion: np.ndarray = field(default_factory=lambda: np.array([]))

    n_intrusion_points: int = 0
    n_withdrawal_points: int = 0

    bulk_density: float = 0.0
    skeletal_density: float = 0.0
    cal_porosity: float = 0.0
    max_intrusion_vol: float = 0.0

    pore_throat_diameter_nm: np.ndarray = field(default_factory=lambda: np.array([]))
    pore_throat_diameter_um: np.ndarray = field(default_factory=lambda: np.array([]))

    incremental_intrusion: np.ndarray = field(default_factory=lambda: np.array([]))
    smoothed_incremental: np.ndarray = field(default_factory=lambda: np.array([]))
    dv_dD: np.ndarray = field(default_factory=lambda: np.array([]))
    dv_dlogD: np.ndarray = field(default_factory=lambda: np.array([]))
    pct_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    pore_throat_bins: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_intrusion: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_area: np.ndarray = field(default_factory=lambda: np.array([]))
    bin_area_pct: np.ndarray = field(default_factory=lambda: np.array([]))
    _bin_edges: np.ndarray = field(default_factory=lambda: np.array([]), repr=False)

    displacement_pressure: float = 0.0
    _disp_p_fit_data: dict = field(default_factory=dict, repr=False, init=False)
    max_pore_diameter_um: float = 0.0
    efficiency: float = 0.0
    intrusion_saturation: float = 100.0

    median_pressure: float = 0.0
    median_diameter_um: float = 0.0
    median_pore_diameter_volume_nm: float = 0.0
    median_pore_diameter_area_nm: float = 0.0
    avg_pore_diameter_nm: float = 0.0
    avg_diameter_um: float = 0.0
    pore_volume: float = 0.0
    total_pore_area: float = 0.0

    sorting_coefficient: float = 0.0
    skewness: float = 0.0
    kurtosis: float = 0.0
    mean_radius: float = 0.0
    structure_coefficient: float = 0.0
    relative_sorting_coeff: float = 0.0

    characteristic_length_nm: float = 0.0
    conductivity_formation_factor: float = 0.0
    tortuosity_factor: float = 0.0
    tortuosity: float = 0.0
    percolation_fractal_dimension: float = 0.0
    breakthrough_pressure_ratio: float = 0.0

    logP: np.ndarray = field(default_factory=lambda: np.array([]))
    log1_S: np.ndarray = field(default_factory=lambda: np.array([]))
    fractal_slopes: list = field(default_factory=list)
    fractal_dimensions: list = field(default_factory=list)

    pore_throat_ratio_data: dict = field(default_factory=dict)

    permeability_10: float = 0.0
    permeability_413: float = 0.0

    specific_surface_area: float = 0.0
    incremental_pore_area: np.ndarray = field(default_factory=lambda: np.array([]))

    # ---- 多重分形 (Multifractal) ----
    mf_q: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_falpha: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_Dq: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_tau_q: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_D0: float = 0.0
    mf_D1: float = 0.0
    mf_D2: float = 0.0
    mf_D_neg10: float = 0.0
    mf_D_10: float = 0.0
    mf_delta_alpha: float = 0.0
    mf_delta_f: float = 0.0
    mf_D_neg10_minus_D_10: float = 0.0
    mf_D_neg10_minus_D0: float = 0.0
    mf_D0_minus_D10: float = 0.0
    mf_H: float = 0.0
    mf_a_min: float = 0.0
    mf_a_max: float = 0.0
    mf_D_a: float = 0.0
    mf_R_d: float = 0.0
    mf_F_max: float = 0.0
    mf_F_min: float = 0.0
    mf_D_Fa: float = 0.0
    mf_R2_alpha: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_R2_falpha: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_R2_Dq: np.ndarray = field(default_factory=lambda: np.array([]))
    mf_partition_data: dict = field(default_factory=dict)

    def to_report_dict(self) -> dict:
        return {
            'sample_name': self._sample_name,
            'test_time': self._test_time,
            'he_porosity': self._he_porosity,
            'micp_porosity': self.cal_porosity,
            'bulk_density': self.bulk_density,
            'skeletal_density': self.skeletal_density,
            'specific_surface_area': self.specific_surface_area,
            'intrusion_saturation': self.intrusion_saturation,
            'efficiency': self.efficiency,
            'displacement_pressure': self.displacement_pressure,
            'max_pore_diameter_um': self.max_pore_diameter_um,
            'matrix_permeability': self.permeability_413,
            'fracture_permeability': self.permeability_10,
            'pore_volume': self.pore_volume,
            'total_pore_area': self.total_pore_area,
            'median_pressure': self.median_pressure,
            'median_diameter_um': self.median_diameter_um,
            'median_pore_diameter_volume_nm': self.median_pore_diameter_volume_nm,
            'median_pore_diameter_area_nm': self.median_pore_diameter_area_nm,
            'avg_pore_diameter_nm': self.avg_pore_diameter_nm,
            'avg_diameter_um': self.avg_diameter_um,
            'pore_throat_peak_um': float(np.min(self.pore_throat_diameter_nm) * 0.001) if len(self.pore_throat_diameter_nm) > 0 else 0,
            'pore_throat_peak_pct': float(np.max(self.bin_pct)) if len(self.bin_pct) > 0 else 0,
            'sorting_coefficient': self.sorting_coefficient,
            'skewness': self.skewness,
            'kurtosis': self.kurtosis,
            'mean_radius': self.mean_radius,
            'structure_coefficient': self.structure_coefficient,
            'relative_sorting_coeff': self.relative_sorting_coeff,
            'characteristic_length_nm': self.characteristic_length_nm,
            'conductivity_formation_factor': self.conductivity_formation_factor,
            'tortuosity_factor': self.tortuosity_factor,
            'tortuosity': self.tortuosity,
            'percolation_fractal_dimension': self.percolation_fractal_dimension,
            'breakthrough_pressure_ratio': self.breakthrough_pressure_ratio,
            'fractal_dimension': self.fractal_dimensions[0] if len(self.fractal_dimensions) > 0 else 0,
            'mf_D0': self.mf_D0,
            'mf_D1': self.mf_D1,
            'mf_D2': self.mf_D2,
            'mf_D_neg10': self.mf_D_neg10,
            'mf_D_10': self.mf_D_10,
            'mf_D_neg10_minus_D_10': self.mf_D_neg10_minus_D_10,
            'mf_H': self.mf_H,
            'mf_delta_alpha': self.mf_delta_alpha,
            'mf_delta_f': self.mf_delta_f,
            'mf_D_a': self.mf_D_a,
            'mf_R_d': self.mf_R_d,
            'mf_D_Fa': self.mf_D_Fa,
        }

    _sample_name: str = field(default="", repr=False, init=False)
    _test_time: str = field(default="", repr=False, init=False)
    _he_porosity: float = field(default=0.0, repr=False, init=False)

    def attach_metadata(self, data: MICPData):
        self._sample_name = data.sample_name
        self._test_time = data.hp_analysis_time
        self._he_porosity = data.porosity
