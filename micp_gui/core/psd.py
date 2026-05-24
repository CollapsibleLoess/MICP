import numpy as np

from .models import MICPParams, MICPResult


def calc_incremental(result: MICPResult):
    if len(result.cum_intrusion) < 2:
        result.incremental_intrusion = np.zeros_like(result.cum_intrusion)
        return

    raw_incremental = np.diff(result.cum_intrusion, prepend=0)
    result.incremental_intrusion = np.where(raw_incremental > 0, raw_incremental, 0)

    if len(result.pore_throat_diameter_nm) < 2:
        return

    d = result.pore_throat_diameter_nm

    d_diff = np.diff(d)
    result.dv_dD = np.zeros(len(d))
    valid = np.abs(d_diff) > 1e-15
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > 0:
        result.dv_dD[valid_idx + 1] = np.where(
            result.incremental_intrusion[valid_idx + 1] / np.abs(d_diff[valid_idx]) > 0,
            result.incremental_intrusion[valid_idx + 1] / np.abs(d_diff[valid_idx]), 0)

    result.dv_dlogD = np.zeros(len(d))
    for i in range(1, len(d)):
        log_diff = abs(np.log10(max(d[i - 1], 1e-30)) - np.log10(max(d[i], 1e-30)))
        if log_diff > 1e-15 and result.incremental_intrusion[i] > 0:
            result.dv_dlogD[i] = result.incremental_intrusion[i] / log_diff

    total = np.sum(result.incremental_intrusion)
    result.pct_distribution = result.incremental_intrusion / total * 100 if total > 0 else np.zeros_like(result.incremental_intrusion)


def calc_psd(result: MICPResult):
    if len(result.pore_throat_diameter_nm) < 2:
        return

    d_nm = result.pore_throat_diameter_nm

    valid = np.isfinite(d_nm) & (d_nm > 0)
    if np.sum(valid) < 2:
        return
    d_valid = d_nm[valid]
    inc_valid = result.incremental_intrusion[valid]

    d_min = float(np.min(d_valid))
    d_max = float(np.max(d_valid))
    if d_min <= 0 or d_max <= 0 or d_min >= d_max:
        return

    n_bins = 25
    log_edges = np.linspace(np.log(d_min), np.log(d_max), n_bins + 1)
    bin_edges = np.exp(log_edges)
    result.pore_throat_bins = np.exp((np.log(bin_edges[:-1]) + np.log(bin_edges[1:])) / 2)
    result._bin_edges = bin_edges

    result.bin_intrusion = np.zeros(n_bins)
    for i in range(n_bins):
        mask = (d_valid >= bin_edges[i]) & (d_valid < bin_edges[i + 1])
        result.bin_intrusion[i] = np.sum(inc_valid[mask])

    total = np.sum(result.bin_intrusion)
    result.bin_pct = result.bin_intrusion / total * 100 if total > 0 else np.zeros(n_bins)

    result.bin_area = result.pore_throat_bins ** 2 * result.bin_intrusion
    total_area = np.sum(result.bin_area)
    result.bin_area_pct = result.bin_area / total_area * 100 if total_area > 0 else np.zeros(n_bins)
