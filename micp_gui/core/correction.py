import numpy as np


def correct_intrusion_cum(cum: np.ndarray) -> np.ndarray:
    """
    差分法修正进汞累积曲线：
    1. 计算差分增量 diff[i] = cum[i] - cum[i-1]
    2. 负增量清零（压力增大时体积不应减小）
    3. 将修正后的增量重新累加
    """
    if len(cum) < 2:
        return np.maximum(cum, 0.0)

    cum = np.asarray(cum, dtype=float)
    diff = np.diff(cum)
    diff = np.maximum(diff, 0.0)

    fixed = np.zeros(len(cum))
    fixed[0] = max(cum[0], 0.0)
    fixed[1:] = fixed[0] + np.cumsum(diff)

    return np.maximum(fixed, 0.0)


def correct_micp_data(data, enforce_mono=True):
    """
    对导入的 MICP 原始数据进行预处理修正。

    进汞：差分法 — 负增量清零后重新累加
    退汞：从进汞峰值开始，逐步减去退汞增量构建累计曲线
         差分法：正增量（体积增大，物理不允许）清零后重新累减

    修正在插值处理之前完成。不修改原始 data 对象。

    返回:
        (intrusion_p, intrusion_cum, extrusion_p, extrusion_cum)
    """
    p_int = np.asarray(data.intrusion_pressure_psia, dtype=float).copy()
    cum_int = np.asarray(data.intrusion_cumulative_mlg, dtype=float).copy()
    p_ext = np.asarray(data.extrusion_pressure_psia, dtype=float).copy()

    if enforce_mono:
        cum_int = correct_intrusion_cum(cum_int)

    peak_vol = cum_int[-1] if len(cum_int) > 0 else 0.0

    if len(p_ext) > 0:
        if enforce_mono:
            inc_ext = np.asarray(data.extrusion_volume_mlg, dtype=float).copy()
            inc_ext = np.maximum(inc_ext, 0.0)

            cum_ext = np.zeros(len(inc_ext))
            cum_ext[0] = max(peak_vol - inc_ext[0], 0.0)
            for i in range(1, len(cum_ext)):
                cum_ext[i] = max(cum_ext[i - 1] - inc_ext[i], 0.0)
        else:
            cum_ext = np.asarray(data.extrusion_cumulative_mlg, dtype=float).copy()
    else:
        cum_ext = np.array([])

    return p_int, cum_int, p_ext, cum_ext
