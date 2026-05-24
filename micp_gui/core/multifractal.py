"""
多重分形分析模块 (Multifractal Analysis for MICP Pore Size Distribution)

基于多尺度分箱的多重分形分析方法。
使用 box_size → 0 多项式外推法求取标度指数的极限值。

算法流程：
  1. 从 MICP 孔径分布构建归一化概率测度
  2. 横坐标对数变换 + 线性化到 [0,1] 区间
  3. 多尺度等间距分箱降采样 (2, 4, 8, 16, ..., 256)
  4. 对每个 q 值，通过 box_size → 0 外推求取 τ(q), α(q)
  5. 计算 D(q), f(α) 及派生特征参数

参考特征参数：
  - D(0), D(1), D(2) — 容量维、信息维、关联维
  - D(-10), D(10) — 极端 q 值下的广义维数
  - D(-10)-D(10) — 谱宽度指标，反映非均质性
  - H = (D(2)+1)/2 — Hurst 指数，>0.5 正持久性，<0.5 反持久性
  - Δα = D_a = α_max - α_min — 奇异性谱宽度
  - R_d — 谱不对称性 (R_d > 0 小测度主导，R_d < 0 大测度主导)
  - Δf = f(α_min) - f(α_max) — 谱不对称性 (>0 大测度占优)
  - D_Fa = f_min - f_max — f(α) 谱宽度

参考文献:
  - Chhabra A, Jensen R. Direct determination of the f(α) singularity spectrum.
    Physical Review Letters, 1989.
  - 冯光俊等. 基于多重分形理论的低阶煤孔隙结构非均质性及影响因素研究.
    现代地质, 2025.
"""

import numpy as np
from numpy import polyfit
from scipy.interpolate import interp1d

from .models import MICPResult


def generate_q_values(q_min=-10, q_max=10):
    ql0 = np.linspace(-4, 4, 160)
    ql1 = np.linspace(q_min, -1, 80)
    ql2 = np.linspace(1, q_max, 80)
    return np.around(np.unique(np.append(np.append(ql1, ql2), np.append([0, 1, 2], ql0))), 2)


def calc_multifractal(result: MICPResult):
    measure_data = _build_input_data(result)
    if measure_data is None:
        return

    x_raw, y_raw = measure_data
    preproc = _preprocess_distribution(x_raw, y_raw)
    if preproc is None:
        return

    box_probabilities = _multi_scale_box_count(preproc)
    if len(box_probabilities) < 3:
        return

    q_values = generate_q_values()
    box_counts = np.array([len(p) for p in box_probabilities])
    box_size = 1.0 / box_counts

    n_q = len(q_values)
    Dq_arr = np.full(n_q, np.nan)
    aq_arr = np.full(n_q, np.nan)
    tq_arr = np.full(n_q, np.nan)
    fq_arr = np.full(n_q, np.nan)

    moment_curves = []

    for iq, q in enumerate(q_values):
        Zq_list = []
        a_numer_list = []
        d_numer_list = []

        for prob in box_probabilities:
            p = np.asarray(prob, dtype=float)
            p_pos = p[p > 0]
            if len(p_pos) < 2:
                continue

            if abs(q) < 1e-10:
                Zq = float(len(p_pos))
            else:
                Zq = float(np.sum(p_pos ** q))
            if Zq <= 0 or not np.isfinite(Zq):
                continue

            if abs(q) < 1e-10:
                mu = np.ones(len(p_pos)) / len(p_pos)
            else:
                mu = (p_pos ** q) / Zq

            a_numer = float(np.sum(mu * np.log(np.maximum(p_pos, 1e-30))))
            Zq_list.append(Zq)
            a_numer_list.append(a_numer)

            if abs(q - 1.0) < 1e-10:
                d_numer = float(np.sum(p_pos * np.log(np.maximum(p_pos, 1e-30))))
                d_numer_list.append(d_numer)

        if len(Zq_list) < 3:
            continue

        bs_arr = box_size[:len(Zq_list)]
        Zq_arr = np.array(Zq_list)
        log_bs = np.log(bs_arr)
        log_Zq = np.log(Zq_arr)

        tau_q = polyfit(bs_arr, log_Zq / log_bs, 1)[1]

        a_numer_arr = np.array(a_numer_list)
        alpha_q = polyfit(bs_arr, a_numer_arr / log_bs, 1)[1]

        if abs(q - 1.0) < 1e-10 and len(d_numer_list) >= 3:
            d_numer_arr = np.array(d_numer_list)
            dim_q = polyfit(bs_arr, d_numer_arr / log_bs, 1)[1]
        elif abs(q - 1.0) > 1e-10:
            dim_q = tau_q / (q - 1)
        else:
            dim_q = np.nan

        f_alpha = q * alpha_q - tau_q

        tq_arr[iq] = tau_q
        aq_arr[iq] = alpha_q
        Dq_arr[iq] = dim_q
        fq_arr[iq] = f_alpha

        moment_curves.append({
            'log_box_size': log_bs.tolist(),
            'log_moment': log_Zq.tolist(),
        })

    result.mf_q = q_values
    result.mf_alpha = aq_arr
    result.mf_falpha = fq_arr
    result.mf_Dq = Dq_arr
    result.mf_tau_q = tq_arr
    result.mf_partition_data = {'moment_curves': moment_curves}

    _extract_key_params(result, q_values, aq_arr, fq_arr, Dq_arr, tq_arr)


def _build_input_data(result: MICPResult):
    if len(result.incremental_intrusion) > 10 and len(result.pore_throat_diameter_nm) > 10:
        d_nm = result.pore_throat_diameter_nm
        inc = result.incremental_intrusion
        valid = np.isfinite(d_nm) & np.isfinite(inc) & (d_nm > 0) & (inc > 0)
        x = d_nm[valid]
        y = inc[valid]
        if len(x) >= 10:
            return x, y

    if len(result.dv_dlogD) > 10:
        d_nm = result.pore_throat_diameter_nm
        dv = result.dv_dlogD
        valid = np.isfinite(d_nm) & np.isfinite(dv) & (d_nm > 0) & (dv >= 0)
        x = d_nm[valid]
        y = dv[valid]
        if len(x) >= 10:
            return x, y

    if len(result.bin_pct) >= 8:
        bins = result.pore_throat_bins
        pct = result.bin_pct
        valid = np.isfinite(bins) & np.isfinite(pct) & (bins > 0) & (pct > 0)
        x = bins[valid]
        y = pct[valid]
        if len(x) >= 8:
            return x, y

    return None


def _preprocess_distribution(x, y):
    mask = y > 0
    x = np.asarray(x[mask], dtype=float)
    y = np.asarray(y[mask], dtype=float)
    if len(x) < 3:
        return None

    log_x = np.log10(np.maximum(x, 1e-30))
    log_min, log_max = np.min(log_x), np.max(log_x)
    if log_max <= log_min:
        return None

    linear_x = (log_x - log_min) / (log_max - log_min)

    n_interp = 1024
    x_new = np.linspace(linear_x[0], linear_x[-1], n_interp)
    f = interp1d(linear_x, y, kind='quadratic', bounds_error=False, fill_value='extrapolate')
    y_new = np.abs(f(x_new))
    y_new = np.maximum(y_new, 0)

    total = np.sum(y_new)
    if total <= 0:
        return None
    return y_new / total


def _multi_scale_box_count(measure):
    n = len(measure)
    max_pow = int(np.floor(np.log2(n)))
    if max_pow < 1:
        return []

    probabilities = []
    for k in range(1, min(max_pow + 1, 9)):
        n_boxes = 2 ** k
        bin_size = n // n_boxes
        p = np.zeros(n_boxes)
        for b in range(n_boxes):
            start = b * bin_size
            end = (b + 1) * bin_size if b < n_boxes - 1 else n
            p[b] = np.sum(measure[start:end])

        total = np.sum(p)
        if total > 0:
            p = p / total
        p = p[p > 0]
        if len(p) >= 2:
            probabilities.append(p)

    return probabilities


def _extract_key_params(result, q_values, alpha, falpha, Dq, tau_q):
    idx0 = np.argmin(np.abs(q_values - 0.0))
    idx1 = np.argmin(np.abs(q_values - 1.0))
    idx2 = np.argmin(np.abs(q_values - 2.0))
    idx_n10 = np.argmin(np.abs(q_values + 10.0))
    idx_10 = np.argmin(np.abs(q_values - 10.0))

    result.mf_D0 = float(Dq[idx0]) if not np.isnan(Dq[idx0]) else 0.0
    result.mf_D1 = float(Dq[idx1]) if not np.isnan(Dq[idx1]) else 0.0
    result.mf_D2 = float(Dq[idx2]) if not np.isnan(Dq[idx2]) else 0.0
    result.mf_D_neg10 = float(Dq[idx_n10]) if not np.isnan(Dq[idx_n10]) else 0.0
    result.mf_D_10 = float(Dq[idx_10]) if not np.isnan(Dq[idx_10]) else 0.0

    result.mf_D_neg10_minus_D_10 = result.mf_D_neg10 - result.mf_D_10
    result.mf_D_neg10_minus_D0 = result.mf_D_neg10 - result.mf_D0
    result.mf_D0_minus_D10 = result.mf_D0 - result.mf_D_10

    result.mf_H = (result.mf_D2 + 1.0) / 2.0 if result.mf_D2 > 0 else 0.0

    valid_alpha = alpha[np.isfinite(alpha)]
    valid_falpha = falpha[np.isfinite(falpha)]
    if len(valid_alpha) >= 2:
        a_min, a_max = float(np.min(valid_alpha)), float(np.max(valid_alpha))
        result.mf_a_min = a_min
        result.mf_a_max = a_max
        result.mf_delta_alpha = a_max - a_min
        result.mf_D_a = a_max - a_min

        a_mid = float(valid_alpha[len(valid_alpha) // 2])
        result.mf_R_d = (a_mid - a_max) - (a_min - a_mid)

        f_min, f_max = float(np.min(valid_falpha)), float(np.max(valid_falpha))
        result.mf_F_min = f_min
        result.mf_F_max = f_max
        result.mf_D_Fa = f_min - f_max

        idx_amin = np.argmin(valid_alpha)
        idx_amax = np.argmax(valid_alpha)
        result.mf_delta_f = float(valid_falpha[idx_amin] - valid_falpha[idx_amax])
    else:
        result.mf_delta_alpha = 0.0
        result.mf_delta_f = 0.0
