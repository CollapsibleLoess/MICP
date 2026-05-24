import numpy as np
from scipy import interpolate


PSIA_TO_MPA = 1.0 / 145.036

DEFAULT_INTRUSION_PSIA_MIN = 0.5
DEFAULT_INTRUSION_PSIA_MAX = 60000.0
FIXED_EXTRUSION_PSIA_MIN = 15.0
FIXED_EXTRUSION_PSIA_MAX = 60000.0

N_INTRUSION_POINTS = 200
N_EXTRUSION_POINTS = 100


def enforce_monotonic(y: np.ndarray, increasing: bool = True) -> np.ndarray:
    if len(y) < 2:
        return y
    result = y.copy()
    if increasing:
        for i in range(1, len(result)):
            if result[i] < result[i - 1]:
                result[i] = result[i - 1]
    else:
        for i in range(1, len(result)):
            if result[i] > result[i - 1]:
                result[i] = result[i - 1]
    return result


def _safe_spline_fit(x_data, x_interp, sorted_x, sorted_y, method='akima'):
    """
    安全插值：优先使用指定方法，失败则回退到线性插值。

    method: 'akima' (Akima1DInterpolator, 单调保形) | 'pchip' | 'linear'
    返回 y_interp 数组（只填充 in_range 部分）。
    """
    in_range = (x_interp >= np.min(sorted_x) - 1e-12) & (x_interp <= np.max(sorted_x) + 1e-12)
    if np.sum(in_range) < 2:
        return np.full(len(x_interp), np.nan), in_range

    xi = x_interp[in_range]
    yi = None

    if method in ('akima', 'pchip'):
        try:
            cls = interpolate.Akima1DInterpolator if method == 'akima' else interpolate.PchipInterpolator
            interp_fn = cls(sorted_x, sorted_y, axis=0)
            yi = interp_fn(xi)
            yi = np.maximum(yi, 0)
        except Exception:
            pass

    if yi is None:
        yi = np.interp(xi, sorted_x, sorted_y)
        yi = np.maximum(yi, 0)

    y_out = np.full(len(x_interp), np.nan, dtype=float)
    y_out[in_range] = yi
    return y_out, in_range


def interpolate_fixed_intrusion(pressure_mpa: np.ndarray, cumulative: np.ndarray,
                                 enforce_mono: bool = True,
                                 p_min_psia: float = None, p_max_psia: float = None) -> tuple:
    """
    固定进汞压力区间，对数均匀分布，Akima 样条插值（安全无振荡）。

    参数:
        p_min_psia: 起始压力 (psia)，默认 0.5
        p_max_psia: 结束压力 (psia)，默认 60000

    外拓处理（左右两侧超出原始范围）:
        - 左侧 → 累积量 = 0（进汞未开始）
        - 右侧 → 累积量 = 末值（进汞已完成）
    """
    n_points = N_INTRUSION_POINTS
    if p_min_psia is None:
        p_min_psia = DEFAULT_INTRUSION_PSIA_MIN
    if p_max_psia is None:
        p_max_psia = DEFAULT_INTRUSION_PSIA_MAX

    x_min_mpa = p_min_psia * PSIA_TO_MPA
    x_max_mpa = p_max_psia * PSIA_TO_MPA

    valid = np.isfinite(pressure_mpa) & np.isfinite(cumulative) & (pressure_mpa > 0)
    x_valid = pressure_mpa[valid]
    y_valid = cumulative[valid]

    if len(x_valid) < 3:
        return pressure_mpa, cumulative

    data_x_min = float(np.min(x_valid))
    data_x_max = float(np.max(x_valid))

    x_min = max(x_min_mpa, data_x_min * 0.999)
    x_max = x_max_mpa
    if x_min <= 0 or x_min >= x_max:
        return pressure_mpa, cumulative

    log_x_interp = np.linspace(np.log10(x_min), np.log10(x_max), n_points)
    x_interp = 10.0 ** log_x_interp

    sort_idx = np.argsort(x_valid)
    x_sorted = x_valid[sort_idx]
    y_sorted = y_valid[sort_idx]

    y_interp, _ = _safe_spline_fit(x_valid, x_interp, x_sorted, y_sorted, method='akima')

    y_interp[x_interp < data_x_min] = 0.0
    y_interp[x_interp > data_x_max] = float(y_sorted[-1])

    if enforce_mono:
        y_interp = enforce_monotonic(y_interp, increasing=True)

    return x_interp, y_interp


def interpolate_fixed_extrusion(pressure_mpa: np.ndarray, cumulative: np.ndarray,
                                 enforce_mono: bool = True) -> tuple:
    """
    固定退汞压力区间 60000 → 15 psia（大气压），对数均匀分布（从高到低）。
    Pchip 保单调样条插值，外拓安全处理。
    """
    n_points = N_EXTRUSION_POINTS
    x_max_mpa = FIXED_EXTRUSION_PSIA_MAX * PSIA_TO_MPA
    x_min_mpa = FIXED_EXTRUSION_PSIA_MIN * PSIA_TO_MPA

    valid = np.isfinite(pressure_mpa) & np.isfinite(cumulative) & (pressure_mpa > 0)
    x_valid = pressure_mpa[valid]
    y_valid = cumulative[valid]

    if len(x_valid) < 3:
        return pressure_mpa, cumulative

    data_x_min = float(np.min(x_valid))
    data_x_max = float(np.max(x_valid))

    x_min = x_min_mpa
    x_max = min(x_max_mpa, data_x_max * 1.001)
    if x_min <= 0 or x_min >= x_max:
        return pressure_mpa, cumulative

    log_x_interp = np.linspace(np.log10(x_max), np.log10(x_min), n_points)
    x_interp = 10.0 ** log_x_interp

    sort_idx = np.argsort(x_valid)
    x_sorted = x_valid[sort_idx]
    y_sorted = y_valid[sort_idx]

    y_interp, _ = _safe_spline_fit(x_valid, x_interp, x_sorted, y_sorted, method='pchip')

    y_interp[x_interp > data_x_max] = float(y_sorted[-1])
    y_interp[x_interp < data_x_min] = float(y_sorted[0])

    if enforce_mono:
        y_interp = enforce_monotonic(y_interp, increasing=False)

    return x_interp, y_interp
