import numpy as np
import pandas as pd

from .models import MICPData, MICPParams


class LoadError(Exception):
    pass


def _parse_number(s) -> float:
    if s is None or pd.isna(s):
        return 0.0
    s = str(s).strip()
    s = s.replace('g', '').replace('°', '').replace('%', '').replace('mL', '').strip()
    try:
        return float(s)
    except ValueError:
        try:
            return float(''.join(c for c in s if c.isdigit() or c == '.' or c == '-'))
        except:
            return 0.0


def _safe_val(df, row, col):
    try:
        if row < len(df) and col < len(df.columns):
            val = df.iloc[row, col]
            if isinstance(val, float) and np.isnan(val):
                return None
            return val
    except:
        pass
    return None


def _find_label(df, label, search_cols=None, search_rows=None):
    if search_cols is None:
        search_cols = range(min(10, len(df.columns)))
    if search_rows is None:
        search_rows = range(len(df))
    for i in search_rows:
        for j in search_cols:
            if j < len(df.columns):
                v = df.iloc[i, j]
                if pd.notna(v) and label in str(v):
                    val_col = j + 1
                    if val_col < len(df.columns):
                        val = df.iloc[i, val_col]
                        if pd.notna(val):
                            return i, j, val
                    return i, j, None
    return None, None, None


def _find_tabular_header_row(df):
    for i in range(20, min(50, len(df))):
        for j in range(5, min(15, len(df.columns))):
            v = df.iloc[i, j]
            if pd.notna(v) and 'Pressure' in str(v) and 'psia' in str(v):
                return i
    return None


def _find_column_by_header(df, header_row, label_fragment):
    if header_row is None:
        return None
    for r in range(header_row, min(header_row + 3, len(df))):
        for c in range(5, min(15, len(df.columns))):
            v = df.iloc[r, c]
            if pd.notna(v) and label_fragment in str(v):
                return c
    return None


def _find_data_start_row(df, header_row, pressure_col):
    if header_row is not None:
        for i in range(header_row + 1, min(header_row + 10, len(df))):
            v = df.iloc[i, pressure_col] if pressure_col < len(df.columns) else None
            if pd.notna(v):
                try:
                    float(v)
                    return i
                except (ValueError, TypeError):
                    pass
    return 29


def load_from_excel(file_path: str) -> MICPData:
    data = MICPData()
    df = pd.read_excel(file_path, sheet_name=0, header=None, engine=None)

    _, _, sample_name_val = _find_label(df, 'Sample:', search_cols=[0], search_rows=range(5, 8))
    if sample_name_val is not None:
        data.sample_name = str(sample_name_val).strip()

    _, _, mass_val = _find_label(df, 'Sample Mass:', search_cols=range(0, 5), search_rows=range(8, 15))
    if mass_val is not None:
        data.sample_mass = _parse_number(mass_val)

    _, _, lp_val = _find_label(df, 'LP Analysis Time:', search_cols=range(0, 5), search_rows=range(8, 15))
    if lp_val is not None:
        data.lp_analysis_time = str(lp_val)

    _, _, hp_val = _find_label(df, 'HP Analysis Time:', search_cols=range(0, 5), search_rows=range(8, 15))
    if hp_val is not None:
        data.hp_analysis_time = str(hp_val)

    _, _, asm_val = _find_label(df, 'Assembly Mass:', search_cols=range(0, 5), search_rows=range(14, 20))
    if asm_val is not None:
        data.assembly_mass = _parse_number(asm_val)

    _, _, pv_val = _find_label(df, 'Penetrometer Volume:', search_cols=range(0, 5), search_rows=range(14, 20))
    if pv_val is not None:
        data.penetrometer_volume = _parse_number(pv_val)

    _, _, pm_val = _find_label(df, 'Penetrometer Mass:', search_cols=range(0, 5), search_rows=range(14, 20))
    if pm_val is not None:
        data.penetrometer_mass = _parse_number(pm_val)

    _, _, por_val = _find_label(df, 'Porosity:', search_cols=range(0, 5), search_rows=range(25, 45))
    if por_val is not None:
        data.porosity = _parse_number(por_val)

    header_row = _find_tabular_header_row(df)
    if header_row is None:
        raise LoadError(
            f"文件 '{file_path}' 中未找到 Tabular Report 区域（需要包含 'Pressure (psia)' 表头）。\n"
            f"请确保原始数据包含 Pressure (psia) 和 Cumulative Pore Volume (mL/g) 两列。"
        )

    pressure_col = _find_column_by_header(df, header_row, 'Pressure')
    cum_vol_col = _find_column_by_header(df, header_row, 'Cumulative Pore Volume')

    if pressure_col is None or cum_vol_col is None:
        missing = []
        if pressure_col is None:
            missing.append('Pressure (psia)')
        if cum_vol_col is None:
            missing.append('Cumulative Pore Volume (mL/g)')
        raise LoadError(
            f"文件 '{file_path}' 中未找到必需的列：{', '.join(missing)}。\n"
            f"请确保原始数据包含 Pressure (psia) 和 Cumulative Pore Volume (mL/g) 两列。"
        )

    data_start_row = _find_data_start_row(df, header_row, pressure_col)

    all_pressures = []
    all_cum_vols = []
    for row_idx in range(data_start_row, len(df)):
        p_val = _safe_val(df, row_idx, pressure_col)
        cv_val = _safe_val(df, row_idx, cum_vol_col)
        if p_val is None or cv_val is None:
            continue
        try:
            p = float(p_val)
            v = float(cv_val)
            if p > 0:
                all_pressures.append(p)
                all_cum_vols.append(v)
        except (ValueError, TypeError):
            continue

    if len(all_pressures) == 0:
        raise LoadError(
            f"文件 '{file_path}' 中未找到有效的压力-累积体积数据。\n"
            f"请确保原始数据包含 Pressure (psia) 和 Cumulative Pore Volume (mL/g) 两列。"
        )

    all_pressures = np.array(all_pressures)
    all_cum_vols = np.array(all_cum_vols)

    peak_p_idx = int(np.argmax(all_pressures))

    intrusion_p = all_pressures[:peak_p_idx + 1]
    intrusion_cum = all_cum_vols[:peak_p_idx + 1]
    intrusion_v = np.zeros(len(intrusion_cum))
    if len(intrusion_cum) > 0:
        intrusion_v[0] = max(intrusion_cum[0], 0.0)
        for k in range(1, len(intrusion_cum)):
            diff = intrusion_cum[k] - intrusion_cum[k - 1]
            intrusion_v[k] = max(diff, 0.0)

    extrusion_p = all_pressures[peak_p_idx + 1:]
    extrusion_cum = all_cum_vols[peak_p_idx + 1:]
    if len(extrusion_cum) > 0:
        peak_vol = intrusion_cum[-1]
        extrusion_v = np.zeros(len(extrusion_cum))
        extrusion_v[0] = max(peak_vol - extrusion_cum[0], 0.0)
        for k in range(1, len(extrusion_cum)):
            diff = extrusion_cum[k - 1] - extrusion_cum[k]
            extrusion_v[k] = max(diff, 0.0)
    else:
        extrusion_v = np.array([])

    data.intrusion_pressure_psia = intrusion_p
    data.intrusion_volume_mlg = intrusion_v
    data.intrusion_cumulative_mlg = intrusion_cum
    data.extrusion_pressure_psia = extrusion_p
    data.extrusion_volume_mlg = extrusion_v
    data.extrusion_cumulative_mlg = extrusion_cum

    return data


def load_contact_angle(file_path: str) -> float:
    df = pd.read_excel(file_path, sheet_name=0, header=None, engine=None)
    _, _, val = _find_label(df, 'Adv. Contact Angle:', search_cols=range(0, 5), search_rows=range(14, 18))
    if val is not None:
        return _parse_number(val)
    return _parse_number(_safe_val(df, 15, 1))
