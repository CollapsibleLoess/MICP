import sys
import os
import warnings
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "data"))
import matplotlib
matplotlib.use('Qt5Agg')
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'PingFang SC', 'SimHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings('ignore', message='Glyph.*missing from font')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QMenuBar, QMenu, QAction,
    QVBoxLayout, QHBoxLayout, QGroupBox, QLabel, QLineEdit, QPushButton,
    QFileDialog, QMessageBox, QSplitter, QFormLayout,
    QScrollArea, QFrame, QDialog, QGridLayout, QRadioButton, QCheckBox,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QSpacerItem, QSizePolicy
)
from PyQt5.QtCore import Qt, pyqtSignal, QTimer
from PyQt5.QtGui import QFont, QKeySequence

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from core import MICPProcessor, LoadError

FONT_XS = 15
FONT_SM = 17
FONT_BASE = 18
FONT_MD = 20
FONT_LG = 21

LEFT_PANEL_WIDTH = 420

SPACING_XS = 6
SPACING_SM = 9
SPACING_MD = 12
SPACING_LG = 18
SPACING_XL = 24

CONTROL_HEIGHT_SM = 39
CONTROL_HEIGHT_MD = 45
CONTROL_HEIGHT_LG = 51

RADIUS_SM = 5
RADIUS_MD = 8
RADIUS_LG = 12

COLOR_PRIMARY = '#1976d2'
COLOR_PRIMARY_DARK = '#1565c0'
COLOR_PRIMARY_LIGHT = '#e3f2fd'
COLOR_SECONDARY = '#424242'
COLOR_TEXT = '#212121'
COLOR_TEXT_SECONDARY = '#616161'
COLOR_TEXT_HINT = '#9e9e9e'
COLOR_BORDER = '#e0e0e0'
COLOR_BG = '#fafafa'
COLOR_WHITE = '#ffffff'
COLOR_ALT_ROW = '#f5f9ff'
COLOR_ERROR = '#d32f2f'

CHART_PALETTE = [
    '#1976d2', '#e53935', '#43a047', '#f57c00', '#7e57c2',
    '#00acc1', '#689f38', '#3949ab', '#c0ca33', '#d81b60',
    '#00897b', '#ffb300', '#546e7a', '#8d6e63', '#78909c'
]

CHART_LINEWIDTH = 2.5
CHART_MARKERSIZE = 6.5
CHART_ALPHA = 0.85
CHART_GRID_ALPHA = 0.4
CHART_DPI = 100


def smart_format_number(value):
    try:
        val = float(value)
        if val == 0:
            return "0.00"
        abs_val = abs(val)
        int_part = int(abs_val)
        if int_part > 0 or (int_part == 0 and abs_val >= 1.0):
            return f"{val:.2f}"
        else:
            return f"{val:.5f}"
    except (ValueError, TypeError):
        return str(value)


class FontSettings:
    def __init__(self):
        self.xs = FONT_XS
        self.sm = FONT_SM
        self.base = FONT_BASE
        self.md = FONT_MD
        self.lg = FONT_LG

    def set_scale(self, scale):
        self.xs = int(FONT_XS * scale)
        self.sm = int(FONT_SM * scale)
        self.base = int(FONT_BASE * scale)
        self.md = int(FONT_MD * scale)
        self.lg = int(FONT_LG * scale)

font_settings = FontSettings()

CHARTS = [
    ("inj_ext", "进退汞曲线"),
    ("interpolated", "插值后进退汞曲线"),
    ("capillary", "毛细管压力曲线"),
    ("dvdlogD", "孔径分布(dV/dlogD)"),
    ("pct", "孔径分布(%)"),
    ("characteristic", "特征参数分布"),
    ("fractal", "单分形维数（分段）"),
    ("mf_spectrum", "多重分形谱f(α)"),
    ("mf_Dq", "广义维数D(q)"),
    ("mf_tau", "质量标度指数τ(q)"),
    ("ratio", "孔喉比"),
    ("swanson_matrix", "基质渗透率"),
    ("swanson_pore", "孔隙渗透率"),
]

CHART_PARAMS = {
    "inj_ext": [],
    "interpolated": [("n_int", "进汞插值点数"), ("n_ext", "退汞插值点数"),
                     ("smoothing", "平滑窗口大小")],
    "capillary": [("disp_p", "排驱压力 $P_d$ (MPa)"), ("max_d", "最大孔径 $d_{max}$ (μm)"),
                  ("med_p", "中值压力 $P_{c50}$ (MPa)"), ("med_d", "中值孔径 $d_{50}$ (μm)"),
                  ("inj_sat", "进汞饱和度 $S_{Hg}$ (%)"), ("eff", "退汞效率 $W_e$ (%)")],
    "dvdlogD": [],
    "pct": [("n_bins", "区间数量"), ("max_pct", "最大进汞量 (%)")],
    "characteristic": [("sp", "分选系数 $S_p$"), ("skp", "偏度 $S_{kp}$"),
                       ("kp", "峰态 $K_p$"), ("dm", "半径均值 $D_M$"),
                       ("phi", "结构系数 $\\phi$"), ("d_coeff", "相对分选 $D$"),
                       ("med_dvol", "体积中值孔径 $d_{50}^{vol}$ (nm)"), ("med_darea", "面积中值孔径 $d_{50}^{area}$ (nm)")],
    "fractal": [("frac_d", "分形维数 $D$")],
    "mf_spectrum": [("mf_delta_alpha", "谱宽度 $\\Delta\\alpha$"), ("mf_delta_f", "不对称性 $\\Delta f$"),
                    ("mf_D_a", "谱宽度 $D_a$"), ("mf_R_d", "不对称性 $R_d$"), ("mf_D_Fa", "谱宽度 $D_{Fa}$"),
                    ("mf_a_min", "最小奇异性 $\\alpha_{min}$"), ("mf_a_max", "最大奇异性 $\\alpha_{max}$")],
    "mf_Dq": [("mf_D0", "容量维数 $D(0)$"), ("mf_D1", "信息维数 $D(1)$"), ("mf_D2", "关联维数 $D(2)$"),
              ("mf_D_neg10", "极端维数 $D(-10)$"), ("mf_D_10", "极端维数 $D(10)$"),
              ("mf_H", "Hurst 指数 $H$"), ("mf_D_neg10_minus_D_10", "谱宽度 $D(-10)-D(10)$")],
    "mf_tau": [],
    "ratio": [("cl", "特征长度 $L_c$ (nm)"), ("cff", "地层因子 $F$"),
              ("tf", "迂曲度因子 $\\tau_F$"), ("tort", "迂曲度 $\\tau$"),
              ("bpr", "突破压力比 $BPR$")],
    "swanson_matrix": [("k413", "基质渗透率 $k_{413}$ (mD)")],
    "swanson_pore": [("k10", "孔隙渗透率 $k_{10}$ (mD)")],
}

SUMMARY_PARAMS = [
    ("he_por", "氦气孔隙度 $\\phi_{He}$ (%)"), ("micp_por", "压汞孔隙度 $\\phi_{Hg}$ (%)"),
    ("bulk_d", "体积密度 $\\rho_b$ (g/cm³)"), ("skel_d", "骨架密度 $\\rho_s$ (g/cm³)"),
    ("pore_v", "孔隙体积 $V_p$ (mL/g)"), ("total_pore_area", "总孔面积 $S_{total}$ (m²/g)"),
    ("ssa", "比表面积 $SSA$ (m²/g)"), ("avg_pd", "平均孔径 $d_{avg}$ (nm)"),
]

PARAM_DESCRIPTIONS = {
    'he_por': ("导入压汞孔隙度(%)",
        "<b>来源：原始Excel文件导入</b>（data.porosity）。代表样品的总孔隙度（氦气法测定）。"),
    'micp_por': ("计算压汞孔隙度(%)",
        "<b>计算：</b>cal_porosity = V_intrusion_max × ρ<sub>b</sub> × 100<br>"
        "其中 V_intrusion_max = max(cum_intrusion)（进汞数据最大累积量），"
        "ρ<sub>b</sub> = 体积密度（见下方）。"),
    'bulk_d': ("体积密度(g/cm³)",
        "<b>计算：</b>ρ<sub>b</sub> = m / D，D = vp - (ma - mp - m) / 13.5939<br>"
        "<b>源参数：</b>m=样品质量，mp=膨胀器质量，ma=注汞后质量，vp=膨胀器容积<br>（以上均从原始Excel导入，13.5939=汞密度 g/cm³）。"),
    'skel_d': ("骨架密度(g/cm³)",
        "<b>计算：</b>ρ<sub>s</sub> = m / (D - V_cm³)<br>"
        "V_cm³ = V_intrusion_max × m（将 mL/g 转为 cm³），D 同上。<br>若 D ≤ V_cm³ 则 ρ<sub>s</sub> = ρ<sub>b</sub>。"),
    'pore_v': ("孔隙体积(mL/g)",
        "<b>计算：</b>pore_volume = max(cum_intrusion)<br>进汞累积曲线的最大值。"),
    'total_pore_area': ("总孔面积(m²/g)",
        "<b>计算：</b>A_total = Σ ΔA_i（所有正孔面积增量之和）<br>"
        "ΔA_i = 2 × ΔV_i / (d_avg × 0.001)<br>"
        "d_avg = (d_i + d_{i+1}) / 2（相邻孔径均值 nm），ΔV_i = 进汞增量，×0.001=nm→μm 换算。"),
    'ssa': ("比表面积(m²/g)",
        "<b>计算：</b>ssa = Σ positive(ΔA_i)<br>总孔面积的正值部分，ΔA_i 计算方式同上。"),
    'avg_pd': ("平均孔径(nm)",
        "<b>计算：</b>d_avg = 4 × pore_volume / A_total × 1000<br>"
        "4V/A 假设（圆柱孔模型），×1000=μm→nm。若 A_total=0 则改用 ssa。"),
    'inj_sat': ("进汞饱和度(%)",
        "<b>计算：</b>= cal_porosity / he_porosity × 100<br>压汞孔隙度占总孔隙度的百分比。"),
    'eff': ("退汞效率(%)",
        "<b>基于绝对值计算：</b>= (1 - min(cum_extrusion) / max(cum_intrusion)) × 100<br>"
        "反映注入汞的绝对退出比例，<b>不受氦气孔隙度输入影响</b>。"),
    'disp_p': ("排驱压力Pd(MPa)",
        "<b>计算：</b>低饱和度区(0-25%) log₁₀(P) vs 饱和度线性回归，截距 10^intercept。<br>若回归失败则取最小进汞压力。"),
    'max_d': ("最大孔径(μm)",
        "<b>计算：</b>d_max = -2σ cosθ / Pd × 1000 × 0.001 (Washburn 方程)<br>"
        "σ=表面张力（Excel导入），θ=接触角（Excel导入）。"),
    'med_p': ("中值压力Pc50(MPa)",
        "<b>计算：</b>累积进汞饱和度 50% 处对应的毛细管压力。"),
    'med_d': ("中值孔径(d₅₀, μm)",
        "<b>计算：</b>d₅₀ = -2σ cosθ / Pc₅₀ × 2 × 1000 × 0.001 (Washburn 方程)"),
    'sp': ("分选系数Sp",
        "<b>由特征参数分布曲线（D-累积S）计算：</b><br>"
        "<b>公式：</b>Sp = (Ψ₈₄-Ψ₁₆)/4 + (Ψ₉₅-Ψ₅)/6.6<br>"
        "<b>Ψ = -log₂(d / 1,000,000)</b>——d=孔径(nm)，<b>Ψₓ</b>为累计饱和度 x% 处对应的 Ψ 值。<br>"
        "Sp 越小→分选越好。"),
    'skp': ("偏度Skp",
        "<b>由特征参数分布曲线（D-累积S）计算：</b><br>"
        "<b>公式：</b>Skp = (Ψ₈₄+Ψ₁₆-2Ψ₅₀)/2(Ψ₈₄-Ψ₁₆) + (Ψ₉₅+Ψ₅-2Ψ₅₀)/2(Ψ₉₅-Ψ₅)<br>"
        "Skp>0 偏细孔，Skp<0 偏粗孔。"),
    'kp': ("峰态Kp",
        "<b>由特征参数分布曲线（D-累积S）计算：</b><br>"
        "<b>公式：</b>Kp = (Ψ₉₅-Ψ₅) / [2.44×(Ψ₇₅-Ψ₂₅)]<br>"
        "Kp 越高→孔径越集中。"),
    'dm': ("半径均值DM",
        "<b>由特征参数分布曲线（D-累积S）计算：</b><br>"
        "<b>公式：</b>DM = (Ψ₁₆ + Ψ₈₄ + Ψ₅₀) / 3<br>"
        "Ψ 尺度上的平均孔径。"),
    'phi': ("结构系数φ",
        "<b>由特征参数分布曲线导出：</b><br>"
        "<b>公式：</b>φ = 2^DM<br>"
        "DM=半径均值（见上方），φ 反映孔隙结构的整体粗细。"),
    'd_coeff': ("相对分选系数D",
        "<b>由特征参数分布曲线导出：</b><br>"
        "<b>公式：</b>D = Sp / DM<br>"
        "Sp=分选系数，DM=半径均值。D 越小→相对分选越好。"),
    'med_dvol': ("中值孔径-体积(nm)",
        "<b>计算：</b>按孔径从大到小排序，累积进汞体积 50% 处对应的孔径。"),
    'med_darea': ("中值孔径-面积(nm)",
        "<b>计算：</b>按孔径从大到小排序，累积孔面积 50% 处对应的孔径。"),
    'frac_d': ("分形维数D",
        "<b>计算：</b>log₁₀(P)-log₁₀(1-S) 曲线斜率 + 3。D 越大→表面越粗糙。"),
    'perc_frac_d': ("渗流分形维数",
        "<b>计算：</b>等于分形维数 D（同上）。"),
    'cl': ("特征长度(nm)",
        "<b>计算：</b>Lc = d_max × 1000<br>最大连通喉道直径（μm→nm 换算）。"),
    'cff': ("导电地层因子",
        "<b>计算：</b>F = (d₅₀ / d_max)²<br>F 越小→导电性越好。"),
    'tf': ("迂曲度因子",
        "<b>计算：</b>τ_factor = 1 / √F"),
    'tort': ("迂曲度",
        "<b>计算：</b>Tortuosity = τ_factor / (cal_porosity / 100)<br>"
        "若 cal_porosity=0 则代入 he_porosity。"),
    'bpr': ("突破压力比",
        "<b>计算：</b>avg(intrusion_pressure) / Pd<br>平均进汞压力与排驱压力之比。"),
    'k413': ("基质渗透率(md)",
        "<b>计算：Swanson 方法</b><br>"
        "k = (1/89) × l_max² × (l_max/lc) × (cal_porosity/100) × (Vc/Vt) × 10⁻¹⁶<br>"
        "孔径范围 10-100 μm，l_max=S×D³ 峰值孔径，lc=特征长度。"),
    'k10': ("孔隙渗透率(md)",
        "<b>计算：同 Swanson 方法</b><br>公式同上，孔径范围 1-10 μm。"),
    'mf_D0': ("容量维数 $D(0)$",
        "<b>计算：</b>多重分形分析中 $q=0$ 对应的广义维数。$D(0)=-\\tau(0)$，反映测度支撑集的 Hausdorff 维数。<br>"
        "$D(0) \\approx 1$ 时分布接近一维直线，值越大表明孔隙分布越复杂。"),
    'mf_D1': ("信息维数 $D(1)$",
        "<b>计算：</b>$q \\to 1$ 时的广义维数极限值。通过 Shannon 熵的标度关系求得。<br>"
        "反映分布的均匀程度，$D(1)$ 越接近 $D(0)$ 分布越均匀，差异越大说明非均质性越强。"),
    'mf_D2': ("关联维数 $D(2)$",
        "<b>计算：</b>$q=2$ 时的广义维数。反映测度的空间关联程度。<br>"
        "与 Hurst 指数关系：$H = (D(2)+1)/2$，$H>0.5$ 正持久性，$H<0.5$ 反持久性。"),
    'mf_D_neg10': ("极端维数 $D(-10)$",
        "<b>计算：</b>$q=-10$ 时的广义维数。$q \\to -\\infty$ 时 $D(q)$ 趋近于最大测度区域的维数。<br>"
        "$D(-10)$ 反映高概率密度区域（大孔）的分形特征。"),
    'mf_D_10': ("极端维数 $D(10)$",
        "<b>计算：</b>$q=10$ 时的广义维数。$q \\to +\\infty$ 时 $D(q)$ 趋近于最小测度区域的维数。<br>"
        "$D(10)$ 反映低概率密度区域（小孔）的分形特征。"),
    'mf_delta_alpha': ("奇异谱宽度 $\\Delta\\alpha$",
        "<b>计算：</b>$\\Delta\\alpha = \\alpha_{max} - \\alpha_{min}$，奇异性指数 $\\alpha$ 的取值范围。<br>"
        "$\\Delta\\alpha$ 越大表示孔隙结构的非均质性越强，局部波动越剧烈。"),
    'mf_delta_f': ("谱不对称性 $\\Delta f$",
        "<b>计算：</b>$\\Delta f = f(\\alpha_{min}) - f(\\alpha_{max})$。<br>"
        "$\\Delta f > 0$ 表示大测度区域（高概率）占主导，$\\Delta f < 0$ 表示小测度区域占主导。"),
    'mf_D_a': ("谱宽度 $D_a$",
        "<b>计算：</b>$D_a = \\alpha_{max} - \\alpha_{min}$，与 $\\Delta\\alpha$ 相同。<br>"
        "衡量多重分形谱在 $\\alpha$ 轴上的展宽，值越大孔隙结构越复杂、非均质性越强。"),
    'mf_R_d': ("不对称性 $R_d$",
        "<b>计算：</b>$R_d = (\\alpha_{mid} - \\alpha_{max}) - (\\alpha_{min} - \\alpha_{mid})$。<br>"
        "$R_d > 0$ 表示小测度区域主导谱形，$R_d < 0$ 表示大测度区域主导。"),
    'mf_D_Fa': ("谱宽度 $D_{Fa}$",
        "<b>计算：</b>$D_{Fa} = f_{min} - f_{max}$，$f(\\alpha)$ 谱的垂直展宽。<br>"
        "反映多重分形谱的对称性和完整性，值越大谱形越不对称。"),
    'mf_H': ("Hurst 指数 $H$",
        "<b>计算：</b>$H = (D(2) + 1) / 2$。<br>"
        "$H > 0.5$ 正持久性（大孔倾向于聚集），$H < 0.5$ 反持久性（大小孔交替分布），$H = 0.5$ 随机分布。"),
    'mf_D_neg10_minus_D_10': ("谱宽度 $D(-10)-D(10)$",
        "<b>计算：</b>$D(-10) - D(10)$，极端 $q$ 值下的维数差。<br>"
        "反映从大孔区域到小孔区域分形特征的差异，值越大表示孔径分布越不均匀。"),
    'mf_a_min': ("最小奇异性 $\\alpha_{min}$",
        "<b>计算：</b>$\\alpha(q)$ 在 $q \\to +\\infty$ 时的极限，对应最大概率密度区域。<br>"
        "反映占主导地位（高概率）区域的局部分形维数。"),
    'mf_a_max': ("最大奇异性 $\\alpha_{max}$",
        "<b>计算：</b>$\\alpha(q)$ 在 $q \\to -\\infty$ 时的极限，对应最小概率密度区域。<br>"
        "反映稀疏分布（低概率）区域的局部分形维数。"),
    'cl': ("特征长度 $L_c$ (nm)",
        "<b>计算：</b>$L_c = d_{max} \\times 1000$，最大连通喉道直径（μm → nm 换算）。<br>"
        "用于 Swanson 渗透率模型中的喉道特征尺度，反映流体流动的主要通道尺寸。"),
    'cff': ("导电地层因子 $F$",
        "<b>计算：</b>$F = (d_{50} / d_{max})^2$。<br>"
        "$F$ 越小表示导电性越好，孔隙喉道分布越均匀。是评估储层导电能力的关键参数。"),
    'tf': ("迂曲度因子 $\\tau_F$",
        "<b>计算：</b>$\\tau_F = 1 / \\sqrt{F}$。<br>"
        "反映电流或流体在孔隙中的路径弯曲程度，值越大表示流动路径越长。"),
    'tort': ("迂曲度 $\\tau$",
        "<b>计算：</b>$\\tau = \\tau_F / (\\phi_{Hg} / 100)$。<br>"
        "消除孔隙度影响后的真实流动路径弯曲度。若压汞孔隙度为 0 则代入氦气孔隙度。<br>"
        "$\\tau$ 越大→流动阻力越大→渗透性越差。"),
    'bpr': ("突破压力比 $BPR$",
        "<b>计算：</b>$BPR = \\bar{P} / P_d$，其中 $\\bar{P}$ = 平均进汞压力，$P_d$ = 排驱压力。<br>"
        "反映整体进汞压力与排驱压力的比值，$BPR$ 越大表示大部分孔隙需要更高压力才能进入。"),
    'k413': ("基质渗透率 $k_{413}$ (mD)",
        "<b>计算：Swanson 方法</b><br>"
        "$k = \\frac{1}{89} \\times l_{max}^2 \\times \\frac{l_{max}}{L_c} \\times \\frac{\\phi_{Hg}}{100} \\times \\frac{V_c}{V_t}$<br>"
        "基于孔径 10–100 μm 范围的 Swanson 参数 ($S \\times D^3$) 峰值计算，适用于基质渗透率评估。"),
    'k10': ("孔隙渗透率 $k_{10}$ (mD)",
        "<b>计算：Swanson 方法</b><br>"
        "与 $k_{413}$ 相同公式，孔径范围 1–10 μm。<br>"
        "适用于裂缝/微孔隙渗透率评估，值越大表示小尺度孔隙的渗流能力越强。"),
    'n_int': ("进汞插值点数",
        "固定压力区间插值后得到的进汞数据点数量（若未启用插值则为原始数据点数）。"),
    'n_ext': ("退汞插值点数",
        "固定压力区间插值后得到的退汞数据点数量（若未启用插值则为原始数据点数）。"),
    'smoothing': ("平滑窗口大小",
        "进汞/退汞曲线上移动平均平滑的窗口大小（宽度）。值为 1 时不进行平滑。<br>"
        "窗口越大，曲线越平滑，但会损失细节特征。建议值 1–5。"),
    'n_bins': ("区间数量",
        "孔径分布 (%) 柱状图的分箱数量。对数均匀分箱，区间越多分布越精细。"),
    'max_pct': ("最大进汞量 (%)",
        "孔径分布 (%) 柱状图中各区间最大进汞占比，用于观察哪个孔径范围的进汞量最高。"),
}

RAW_PARAM_DESCRIPTIONS = {
    "sample_name": ("样品名称", "从原始Excel文件中读取的样品名称，用于标识当前测试样本。"),
    "sample_mass": ("样品质量 (g)", "从原始Excel文件中读取的样品质量，用于密度和体积计算。"),
    "penetrometer_mass": ("膨胀器质量 (g)", "从原始Excel文件中读取的空膨胀器（压汞仪）质量，用于计算汞体积校正。"),
    "assembly_mass": ("注汞后质量 (g)", "从原始Excel文件中读取的注汞后膨胀器+样品+汞的总质量。"),
    "penetrometer_volume": ("膨胀器容积 (mL)", "从原始Excel文件中读取的膨胀器内部容积，用于体积密度计算。"),
    "points": ("进汞/退汞点数", "原始数据中的进汞和退汞测量点数量，<b>插值前</b>的原始数据点数。"),
}

CALC_PARAM_DESCRIPTIONS = {
    "contact_angle": ("接触角 (°)", "汞与岩石表面的接触角，<b>从原始Excel文件读取</b>，默认130°。用于Washburn方程计算孔径。"),
    "surface_tension": ("表面张力 (N/m)", "汞的表面张力，<b>从原始Excel文件读取</b>，默认0.475 N/m。用于Washburn方程计算孔径。"),
    "smoothing": ("平滑窗口", "进汞/退汞曲线的移动平均平滑窗口大小。值为1时不进行平滑。"),
    "he_porosity": ("氦气孔隙度 (%)", "手动输入的氦气（总）孔隙度，用于计算进汞饱和度。<br><b>留空</b>则默认等于压汞孔隙度(100%饱和度)。"),
    "enforce_mono": ("差分法修正单调性", "勾选后强制进汞曲线单调递增、退汞曲线单调递减。"),
    "use_fixed_interpolation": ("固定压力区间插值", "勾选后使用固定的进汞压力区间进行<b>对数均匀插值</b>（Akima样条）。<br>进汞区间可由下方输入框自定义；退汞固定为 <b>60000→15 psia</b>（到大气压）。<br><b>取消勾选则不进行插值</b>，直接使用原始数据。"),
    "disp_p_range": ("排驱压力拟合范围 (%)", "低饱和度区间用于拟合排驱压力。<br><b>留空</b>则自动寻找最佳范围（默认为0~25%的低饱和度区）。<br>例如输入 0.1 和 20 表示使用饱和度 0.1%~20% 间的数据点进行线性回归。"),
    "frac_seg_breakpoint": ("分形分段断点 (logP)", "单分形维数分段拟合的断点位置。<br>数据点按 $\\log P$ 值以此断点为界分为两段，各自独立线性拟合求取分形维数。<br><b>留空</b>则默认使用 $\\log P = 2.0$。支持图表上拖动三角形标记交互调整。"),
}


def show_param_info(parent, title, detail):
    dlg = QDialog(parent)
    dlg.setWindowTitle("参数说明")
    dlg.setMinimumSize(480, 320)
    dlg.setMaximumSize(600, 480)
    dlg.setStyleSheet(f"background-color: {COLOR_BG};")
    layout = QVBoxLayout(dlg)
    layout.setContentsMargins(24, 20, 24, 20)
    layout.setSpacing(14)
    title_label = QLabel(f"<h3 style='color:{COLOR_PRIMARY};'>{_tex_to_html(title)}</h3>")
    title_label.setWordWrap(True)
    title_label.setTextFormat(Qt.RichText)
    layout.addWidget(title_label)
    sep = QFrame()
    sep.setFrameShape(QFrame.HLine)
    sep.setStyleSheet(f"background-color: {COLOR_BORDER};")
    sep.setFixedHeight(1)
    layout.addWidget(sep)
    detail_label = QLabel(_tex_to_html(detail))
    detail_label.setWordWrap(True)
    detail_label.setTextFormat(Qt.RichText)
    detail_label.setStyleSheet(f"color: {COLOR_TEXT}; font-size: {font_settings.sm}px; line-height: 1.6;")
    scroll = QScrollArea()
    scroll.setWidgetResizable(True)
    scroll.setFrameShape(QFrame.NoFrame)
    scroll.setWidget(detail_label)
    layout.addWidget(scroll, 1)
    btn_ok = QPushButton("确定")
    btn_ok.setMinimumHeight(CONTROL_HEIGHT_SM)
    btn_ok.setMinimumWidth(80)
    btn_ok.setStyleSheet(f"""
        QPushButton {{ background-color: {COLOR_PRIMARY}; color: white;
            font-size: {font_settings.sm}px; border: none; border-radius: {RADIUS_SM}px;
            padding: {SPACING_XS}px {SPACING_LG}px; }}
        QPushButton:hover {{ background-color: {COLOR_PRIMARY_DARK}; }}
    """)
    btn_ok.clicked.connect(dlg.accept)
    btn_layout = QHBoxLayout()
    btn_layout.addStretch()
    btn_layout.addWidget(btn_ok)
    layout.addLayout(btn_layout)
    dlg.exec_()


def _make_info_btn(parent, title, detail, small=False):
    size = f"min-width: 16px; max-width: 16px; min-height: 16px; max-height: 16px;" if small else "min-width: 20px; max-width: 20px; min-height: 20px; max-height: 20px;"
    sz = font_settings.xs if small else font_settings.xs
    btn = QPushButton("?")
    btn.setStyleSheet(f"""
        QPushButton {{
            background-color: transparent; color: {COLOR_PRIMARY};
            font-size: {sz}px; font-weight: 700;
            border: 1px solid {COLOR_BORDER}; border-radius: {RADIUS_SM}px;
            {size} padding: 0px;
        }}
        QPushButton:hover {{ background-color: {COLOR_PRIMARY_LIGHT}; border-color: {COLOR_PRIMARY}; }}
    """)
    btn.setCursor(Qt.PointingHandCursor)
    btn.clicked.connect(lambda: show_param_info(parent, title, detail))
    return btn


_TEX_MAP = {
    "\\phi": "φ", "\\tau": "τ", "\\rho": "ρ",
    "\\Delta": "Δ", "\\alpha": "α", "\\times": "×",
    "\\log": "log", "\\mu": "μm", "\\approx": "≈",
    "\\infty": "∞", "\\bar": "̄ ", "\\to": "→",
}


def _tex_to_html(text):
    import re
    t = text
    for macro, uni in _TEX_MAP.items():
        t = t.replace(macro, uni)
    t = re.sub(r'_\{(.*?)\}', r'<sub>\1</sub>', t)
    t = re.sub(r'\^\{([^}]*)\}', r'<sup>\1</sup>', t)
    t = t.replace("$", "")
    return t


class MplCanvas(FigureCanvas):
    MARGIN_TOP = 0.18
    MARGIN_BOTTOM = 0.20
    MARGIN_LEFT = 0.18
    MARGIN_RIGHT = 0.15

    def __init__(self, parent=None, width=10, height=6, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.fig.patch.set_facecolor('#ffffff')
        self.axes.set_facecolor('#fafafa')
        self.twin_x = None
        self.twin_y = None
        self.fig.subplots_adjust(
            top=1.0 - self.MARGIN_TOP, bottom=self.MARGIN_BOTTOM,
            left=self.MARGIN_LEFT, right=1.0 - self.MARGIN_RIGHT,
        )
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)

    def clear_plot(self):
        self.fig.clear()
        self.axes = self.fig.add_subplot(111)
        self.axes.set_facecolor('#fafafa')
        self.twin_x = None
        self.twin_y = None
        self._disp_low_marker = None
        self._disp_high_marker = None
        self.fig.subplots_adjust(
            top=1.0 - self.MARGIN_TOP, bottom=self.MARGIN_BOTTOM,
            left=self.MARGIN_LEFT, right=1.0 - self.MARGIN_RIGHT,
        )
        self.draw()

    def _add_pore_diam_twin_x(self, sigma, theta):
        theta_rad = np.radians(theta)
        const = -4 * sigma * np.cos(theta_rad) * 1000
        def p_to_d(p):
            a = np.asarray(p, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(a > 0, const / a, np.nan)
        def d_to_p(d):
            a = np.asarray(d, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(a > 0, const / a, np.nan)
        self.twin_x = self.axes.secondary_xaxis('top', functions=(p_to_d, d_to_p))
        self.twin_x.set_xlabel('d (nm)', color=COLOR_PRIMARY, fontsize=font_settings.md, fontweight=500, labelpad=10)
        self.twin_x.tick_params(colors=COLOR_PRIMARY, labelsize=font_settings.sm, length=5, width=1, pad=8)

    def _add_pore_diam_twin_y(self, sigma, theta):
        theta_rad = np.radians(theta)
        const = -4 * sigma * np.cos(theta_rad) * 1000
        def p_to_d(p):
            a = np.asarray(p, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(a > 0, const / a, np.nan)
        def d_to_p(d):
            a = np.asarray(d, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(a > 0, const / a, np.nan)
        self.twin_y = self.axes.secondary_yaxis('right', functions=(p_to_d, d_to_p))
        self.twin_y.set_ylabel('d (nm)', color=COLOR_PRIMARY, fontsize=font_settings.md, fontweight=500, labelpad=10)
        self.twin_y.tick_params(colors=COLOR_PRIMARY, labelsize=font_settings.sm, length=5, width=1, pad=8)

    def _add_pore_diam_twin_x_logp(self, sigma, theta):
        theta_rad = np.radians(theta)
        const = -4 * sigma * np.cos(theta_rad) * 1000
        def logp_to_d(logp):
            a = np.asarray(logp, dtype=float)
            p = 10.0 ** a
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(p > 0, const / p, np.nan)
        def d_to_logp(d):
            a = np.asarray(d, dtype=float)
            with np.errstate(divide='ignore', invalid='ignore'):
                return np.where(a > 0, np.log10(const / a), np.nan)
        self.twin_x = self.axes.secondary_xaxis('top', functions=(logp_to_d, d_to_logp))
        self.twin_x.set_xlabel('d (nm)', color=COLOR_PRIMARY, fontsize=font_settings.sm, fontweight=500, labelpad=6)
        self.twin_x.tick_params(colors=COLOR_PRIMARY, labelsize=font_settings.xs, length=4, width=1, pad=6)

    def update_axes(self, xlabel, ylabel, x_log=False, x_inv=False, y_log=False):
        self.axes.set_xlabel(xlabel, color=COLOR_SECONDARY, fontsize=font_settings.md, fontweight=500, labelpad=10)
        self.axes.set_ylabel(ylabel, color=COLOR_SECONDARY, fontsize=font_settings.md, fontweight=500, labelpad=10)
        self.axes.tick_params(colors='#616161', labelsize=font_settings.sm, length=5, width=1, pad=8)
        if x_log:
            self.axes.set_xscale('log')
        if x_inv:
            self.axes.invert_xaxis()
        if y_log:
            self.axes.set_yscale('log')
        self.axes.grid(True, color=COLOR_BORDER, alpha=CHART_GRID_ALPHA, linestyle='--', linewidth=0.8)
        for spine in self.axes.spines.values():
            spine.set_color('#bdbdbd')
            spine.set_linewidth(1.0)

    def draw_line(self, x, y, label, color, linewidth=CHART_LINEWIDTH, with_markers=True, marker='o'):
        if with_markers and len(x) <= 200:
            self.axes.plot(x, y, label=label, color=color, linewidth=linewidth,
                          marker=marker, markersize=CHART_MARKERSIZE, markeredgecolor=color,
                          markerfacecolor='white', markeredgewidth=1.2, alpha=CHART_ALPHA)
        elif with_markers and len(x) > 200:
            indices = np.linspace(0, len(x)-1, min(50, len(x)), dtype=int).tolist()
            x_markers = [x[i] for i in indices]
            y_markers = [y[i] for i in indices]
            self.axes.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=CHART_ALPHA)
            self.axes.scatter(x_markers, y_markers, color=color, s=CHART_MARKERSIZE**2 * 1.5,
                            edgecolors=color, facecolors='white', linewidths=1.2, zorder=5)
        else:
            self.axes.plot(x, y, label=label, color=color, linewidth=linewidth, alpha=CHART_ALPHA)

    def draw_scatter(self, x, y, label, color, markersize=30):
        self.axes.scatter(x, y, label=label, color=color, s=markersize,
                        edgecolors=color, facecolors='white', linewidths=1.2,
                        alpha=CHART_ALPHA, zorder=5)

    def draw_bar(self, x, y, label, color, width=0.8):
        bars = self.axes.bar(x, y, label=label, color=color, width=width,
                           alpha=CHART_ALPHA, edgecolor='white', linewidth=0.5)
        for bar in bars:
            height = bar.get_height()
            if height > 0 and len(x) <= 30:
                self.axes.text(bar.get_x() + bar.get_width()/2., height,
                             f'{height:.1f}', ha='center', va='bottom',
                             fontsize=font_settings.xs, color=COLOR_SECONDARY)

    def add_legend(self):
        lines, labels = self.axes.get_legend_handles_labels()
        if lines:
            legend = self.axes.legend(lines, labels, facecolor=COLOR_WHITE,
                                      edgecolor=COLOR_BORDER, labelcolor=COLOR_SECONDARY,
                                      fontsize=font_settings.xs, framealpha=0.95, loc='best',
                                      borderpad=0.8, fancybox=True, shadow=False, ncol=1)
            legend.get_frame().set_linewidth(1)
            legend.get_frame().set_boxstyle('round,pad=0.3')
        self.fig.subplots_adjust(
            top=1.0 - self.MARGIN_TOP, bottom=self.MARGIN_BOTTOM,
            left=self.MARGIN_LEFT, right=1.0 - self.MARGIN_RIGHT,
        )
        self.draw()


class ChartCard(QFrame):
    chart_clicked = pyqtSignal(str)

    def __init__(self, chart_id, chart_name, parent=None):
        super().__init__(parent)
        self.chart_id = chart_id
        self.chart_name = chart_name
        self.param_labels = {}
        self._selected = False
        self.disp_p_sat_min_input = None
        self.disp_p_sat_max_input = None
        self.frac_seg_bp_input = None
        self._dragging_handle = None
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(self._normal_style())
        layout = QVBoxLayout(self)
        layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        layout.setSpacing(SPACING_SM)

        title_label = QLabel(self.chart_name)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.lg + 4}px; font-weight: 700; padding: 0;")
        layout.addWidget(title_label)

        self.canvas = MplCanvas(self, width=6.5, height=10, dpi=CHART_DPI)
        self.canvas.setMinimumSize(400, int(8 * CHART_DPI * 0.7))
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        self.canvas.installEventFilter(self)
        layout.addWidget(self.canvas)

        sep = QFrame()
        sep.setFrameShape(QFrame.HLine)
        sep.setStyleSheet(f"background-color: {COLOR_BORDER};")
        sep.setFixedHeight(1)
        layout.addWidget(sep)

        params_widget = QWidget()
        self.params_layout = QFormLayout(params_widget)
        self.params_layout.setLabelAlignment(Qt.AlignLeft)
        self.params_layout.setFormAlignment(Qt.AlignLeft)
        self.params_layout.setSpacing(SPACING_XS)
        self.params_layout.setVerticalSpacing(SPACING_XS)
        self.params_layout.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM)

        params_config = CHART_PARAMS.get(self.chart_id, [])
        for key, label_text in params_config:
            lbl = QLabel(f"{_tex_to_html(label_text)}:")
            lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
            lbl.setTextFormat(Qt.RichText)
            val = QLabel("---")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            val.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.sm}px; font-weight: 600;")

            desc = PARAM_DESCRIPTIONS.get(key, (label_text, "暂无说明"))
            info_btn = _make_info_btn(self, desc[0], desc[1])

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(SPACING_XS)
            row_layout.addWidget(lbl)
            row_layout.addWidget(val)
            row_layout.addWidget(info_btn)
            row_layout.addStretch(1)
            self.param_labels[key] = val
            self.params_layout.addRow(row)

        if params_config:
            layout.addWidget(params_widget)

        if self.chart_id == "capillary":
            self._setup_disp_range_inputs(layout)

        if self.chart_id == "fractal":
            self._setup_frac_range_inputs(layout)

    def _setup_disp_range_inputs(self, layout):
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"background-color: {COLOR_BORDER};")
        sep2.setFixedHeight(1)
        layout.addWidget(sep2)

        disp_row = QWidget()
        disp_layout = QHBoxLayout(disp_row)
        disp_layout.setContentsMargins(SPACING_SM, SPACING_XS, SPACING_SM, SPACING_XS)
        disp_layout.setSpacing(SPACING_XS)
        disp_lbl = QLabel("排驱压力拟合范围 (%):")
        disp_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
        disp_layout.addWidget(disp_lbl)
        self.disp_p_sat_min_input = QLineEdit("")
        self.disp_p_sat_min_input.setPlaceholderText("下限")
        self.disp_p_sat_min_input.setMinimumHeight(CONTROL_HEIGHT_SM)
        self.disp_p_sat_min_input.setMinimumWidth(65)
        self.disp_p_sat_min_input.setMaximumWidth(85)
        self.disp_p_sat_min_input.setStyleSheet(LeftPanel._lineedit_style())
        self.disp_p_sat_min_input.textChanged.connect(self._on_disp_range_input_changed)
        disp_layout.addWidget(self.disp_p_sat_min_input)
        dash = QLabel("  —  ")
        dash.setStyleSheet(f"color: {COLOR_TEXT_HINT}; font-size: {font_settings.sm}px;")
        disp_layout.addWidget(dash)
        self.disp_p_sat_max_input = QLineEdit("")
        self.disp_p_sat_max_input.setPlaceholderText("上限")
        self.disp_p_sat_max_input.setMinimumHeight(CONTROL_HEIGHT_SM)
        self.disp_p_sat_max_input.setMinimumWidth(65)
        self.disp_p_sat_max_input.setMaximumWidth(85)
        self.disp_p_sat_max_input.setStyleSheet(LeftPanel._lineedit_style())
        self.disp_p_sat_max_input.textChanged.connect(self._on_disp_range_input_changed)
        disp_layout.addWidget(self.disp_p_sat_max_input)
        disp_desc = CALC_PARAM_DESCRIPTIONS.get("disp_p_range", ("排驱压力拟合范围 (%)", "暂无说明"))
        disp_info_btn = _make_info_btn(self, disp_desc[0], disp_desc[1])
        disp_layout.addWidget(disp_info_btn)
        disp_layout.addStretch(1)
        layout.addWidget(disp_row)

    def _on_disp_range_input_changed(self):
        self.update_disp_markers()

    def get_disp_p_sat_min(self):
        if self.disp_p_sat_min_input:
            try:
                return float(self.disp_p_sat_min_input.text().strip())
            except (ValueError, TypeError):
                pass
        return 0.0

    def get_disp_p_sat_max(self):
        if self.disp_p_sat_max_input:
            try:
                return float(self.disp_p_sat_max_input.text().strip())
            except (ValueError, TypeError):
                pass
        return 0.0

    def update_disp_markers(self):
        canvas = self.canvas
        if not hasattr(canvas, '_disp_low_marker') or canvas._disp_low_marker is None:
            return
        try:
            low_val = float(self.disp_p_sat_min_input.text().strip()) if self.disp_p_sat_min_input and self.disp_p_sat_min_input.text().strip() else None
            high_val = float(self.disp_p_sat_max_input.text().strip()) if self.disp_p_sat_max_input and self.disp_p_sat_max_input.text().strip() else None
        except (ValueError, TypeError):
            return
        sat_scale = getattr(canvas, '_cap_sat_scale', 1.0)
        if low_val is not None:
            canvas._disp_low_marker.set_offsets([[low_val * sat_scale, canvas._disp_marker_y]])
        if high_val is not None:
            canvas._disp_high_marker.set_offsets([[high_val * sat_scale, canvas._disp_marker_y]])
        canvas.draw_idle()

    def _setup_frac_range_inputs(self, layout):
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.HLine)
        sep2.setStyleSheet(f"background-color: {COLOR_BORDER};")
        sep2.setFixedHeight(1)
        layout.addWidget(sep2)

        frac_row = QWidget()
        frac_layout = QHBoxLayout(frac_row)
        frac_layout.setContentsMargins(SPACING_SM, SPACING_XS, SPACING_SM, SPACING_XS)
        frac_layout.setSpacing(SPACING_XS)
        frac_lbl = QLabel("分段拟合断点 (logP):")
        frac_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
        frac_layout.addWidget(frac_lbl)
        self.frac_seg_bp_input = QLineEdit("")
        self.frac_seg_bp_input.setPlaceholderText("默认 2.0")
        self.frac_seg_bp_input.setMinimumHeight(CONTROL_HEIGHT_SM)
        self.frac_seg_bp_input.setMinimumWidth(65)
        self.frac_seg_bp_input.setMaximumWidth(85)
        self.frac_seg_bp_input.setStyleSheet(LeftPanel._lineedit_style())
        frac_layout.addWidget(self.frac_seg_bp_input)
        frac_desc = CALC_PARAM_DESCRIPTIONS.get("frac_seg_breakpoint", ("分形分段断点 (logP)", "暂无说明"))
        frac_info_btn = _make_info_btn(self, frac_desc[0], frac_desc[1])
        frac_layout.addWidget(frac_info_btn)
        frac_layout.addStretch(1)
        layout.addWidget(frac_row)

    def get_frac_seg_breakpoint(self):
        if self.frac_seg_bp_input:
            try:
                return float(self.frac_seg_bp_input.text().strip())
            except (ValueError, TypeError):
                pass
        return 0.0

    def _normal_style(self):
        return f"ChartCard {{ background-color: {COLOR_WHITE}; border: 2px solid {COLOR_BORDER}; border-radius: {RADIUS_LG}px; }}"

    def set_selected(self, selected):
        self._selected = selected
        if selected:
            self.setStyleSheet(f"ChartCard {{ background-color: {COLOR_WHITE}; border: 3px solid {COLOR_PRIMARY}; border-radius: {RADIUS_LG}px; }}")
        else:
            self.setStyleSheet(self._normal_style())

    def mousePressEvent(self, event):
        self.chart_clicked.emit(self.chart_id)
        super().mousePressEvent(event)

    def eventFilter(self, obj, event):
        if obj is self.canvas and event.type() == event.MouseButtonPress:
            self.chart_clicked.emit(self.chart_id)
        return super().eventFilter(obj, event)

    def update_params(self, vals):
        for key, label in self.param_labels.items():
            if key in vals:
                label.setText(vals[key])
            else:
                label.setText("---")

    def _ensure_disp_drag_events(self):
        canvas = self.canvas
        if getattr(canvas, '_disp_drag_connected', False):
            return
        canvas._disp_drag_connected = True
        canvas._disp_dragging = None
        canvas.mpl_connect('pick_event', self._on_disp_marker_pick)
        canvas.mpl_connect('button_release_event', self._on_disp_marker_release)
        canvas.mpl_connect('motion_notify_event', self._on_disp_marker_motion)

    def _on_disp_marker_pick(self, event):
        canvas = self.canvas
        artist = event.artist
        if artist is getattr(canvas, '_disp_low_marker', None):
            canvas._disp_dragging = 'low'
        elif artist is getattr(canvas, '_disp_high_marker', None):
            canvas._disp_dragging = 'high'

    def _on_disp_marker_motion(self, event):
        canvas = self.canvas
        if getattr(canvas, '_disp_dragging', None) is None:
            return
        if event.xdata is None or event.ydata is None:
            return
        new_x = float(event.xdata)
        x_min, x_max = canvas.axes.get_xlim()
        if new_x < x_max or new_x > x_min:
            return
        if canvas._disp_dragging == 'low':
            canvas._disp_low_marker.set_offsets([[new_x, canvas._disp_marker_y]])
        else:
            canvas._disp_high_marker.set_offsets([[new_x, canvas._disp_marker_y]])
        canvas.draw_idle()

    def _on_disp_marker_release(self, event):
        canvas = self.canvas
        if getattr(canvas, '_disp_dragging', None) is None:
            return
        sat_scale = getattr(canvas, '_cap_sat_scale', 1.0)
        if canvas._disp_dragging == 'low':
            new_x = float(canvas._disp_low_marker.get_offsets()[0][0])
            if sat_scale > 0:
                new_x = new_x / sat_scale
            if self.disp_p_sat_min_input:
                self.disp_p_sat_min_input.blockSignals(True)
                self.disp_p_sat_min_input.setText(f"{new_x:.2f}")
                self.disp_p_sat_min_input.blockSignals(False)
        else:
            new_x = float(canvas._disp_high_marker.get_offsets()[0][0])
            if sat_scale > 0:
                new_x = new_x / sat_scale
            if self.disp_p_sat_max_input:
                self.disp_p_sat_max_input.blockSignals(True)
                self.disp_p_sat_max_input.setText(f"{new_x:.2f}")
                self.disp_p_sat_max_input.blockSignals(False)
        canvas._disp_dragging = None

    def _ensure_frac_drag_events(self):
        canvas = self.canvas
        if getattr(canvas, '_frac_drag_connected', False):
            return
        canvas._frac_drag_connected = True
        canvas._frac_dragging = False
        canvas.mpl_connect('pick_event', self._on_frac_marker_pick)
        canvas.mpl_connect('button_release_event', self._on_frac_marker_release)
        canvas.mpl_connect('motion_notify_event', self._on_frac_marker_motion)

    def _on_frac_marker_pick(self, event):
        canvas = self.canvas
        if event.artist is getattr(canvas, '_frac_bp_marker', None):
            canvas._frac_dragging = True

    def _on_frac_marker_motion(self, event):
        canvas = self.canvas
        if not getattr(canvas, '_frac_dragging', False):
            return
        if event.xdata is None:
            return
        new_x = float(event.xdata)
        x_min, x_max = canvas.axes.get_xlim()
        new_x = max(x_min, min(x_max, new_x))
        canvas._frac_bp = new_x
        canvas._frac_bp_marker.set_offsets([[new_x, canvas.axes.get_ylim()[1]]])
        canvas._frac_bp_label.set_position((new_x, canvas.axes.get_ylim()[1]))
        canvas._frac_bp_label.set_text(f"$\\log P$={new_x:.2f}")
        canvas.draw_idle()

    def _on_frac_marker_release(self, event):
        canvas = self.canvas
        if not getattr(canvas, '_frac_dragging', False):
            return
        canvas._frac_dragging = False
        new_x = float(canvas._frac_bp_marker.get_offsets()[0][0])
        if self.frac_seg_bp_input:
            self.frac_seg_bp_input.blockSignals(True)
            self.frac_seg_bp_input.setText(f"{new_x:.2f}")
            self.frac_seg_bp_input.blockSignals(False)


class SummaryReport(QFrame):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.param_labels = {}
        self.param_key_labels = {}
        self.init_ui()

    def init_ui(self):
        self.setStyleSheet(f"""
            SummaryReport {{
                background-color: {COLOR_WHITE};
                border: 2px solid {COLOR_PRIMARY};
                border-radius: {RADIUS_LG}px;
            }}
        """)
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG)
        main_layout.setSpacing(SPACING_MD)

        header_layout = QHBoxLayout()
        title = QLabel("Summary Report")
        title.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.lg + 4}px; font-weight: 700; padding: 0;")
        header_layout.addWidget(title)
        header_layout.addStretch(1)
        self.btn_copy = QPushButton("复制参数")
        self.btn_copy.setMinimumHeight(CONTROL_HEIGHT_SM)
        self.btn_copy.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY}; color: white;
                font-size: {font_settings.xs}px; font-weight: 500;
                border: none; border-radius: {RADIUS_SM}px;
                padding: 0 {SPACING_MD}px; min-height: {CONTROL_HEIGHT_SM}px;
            }}
            QPushButton:hover {{ background-color: {COLOR_PRIMARY_DARK}; }}
        """)
        self.btn_copy.setCursor(Qt.PointingHandCursor)
        self.btn_copy.clicked.connect(self._copy_all_params)
        header_layout.addWidget(self.btn_copy)
        main_layout.addLayout(header_layout)

        grid = QGridLayout()
        grid.setSpacing(SPACING_SM)
        grid.setContentsMargins(SPACING_SM, SPACING_SM, SPACING_SM, SPACING_SM)

        for i, (key, label_text) in enumerate(SUMMARY_PARAMS):
            row, col = i // 2, i % 2
            card = QFrame()
            card.setStyleSheet(f"""
                QFrame {{ background-color: {COLOR_PRIMARY_LIGHT}; border: 1px solid {COLOR_BORDER};
                    border-radius: {RADIUS_MD}px; padding: {SPACING_SM}px; }}
            """)
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(SPACING_MD, SPACING_SM, SPACING_MD, SPACING_SM)
            card_layout.setSpacing(2)
            header_row = QWidget()
            header_layout2 = QHBoxLayout(header_row)
            header_layout2.setContentsMargins(0, 0, 0, 0)
            header_layout2.setSpacing(SPACING_XS)
            lbl = QLabel(_tex_to_html(label_text))
            lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.xs}px; border: none;")
            lbl.setTextFormat(Qt.RichText)
            header_layout2.addWidget(lbl)
            desc = PARAM_DESCRIPTIONS.get(key, (label_text, "暂无说明"))
            info_btn = _make_info_btn(self, desc[0], desc[1], small=True)
            header_layout2.addWidget(info_btn)
            header_layout2.addStretch(1)
            card_layout.addWidget(header_row)
            val = QLabel("---")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            val.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.md}px; font-weight: 700; border: none;")
            self.param_labels[key] = val
            self.param_key_labels[key] = label_text
            card_layout.addWidget(val)
            grid.addWidget(card, row, col)

        main_layout.addLayout(grid)

    def _copy_all_params(self):
        lines = []
        for key, label_text in self.param_key_labels.items():
            val = self.param_labels[key].text()
            lines.append(f"{label_text}\t{val}")
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        parent = self.parent()
        while parent and not isinstance(parent, QMainWindow):
            parent = parent.parent()
        if parent:
            parent.statusBar().showMessage(f"已复制 {len(self.param_key_labels)} 个参数到剪贴板，可直接粘贴到 Excel", 3000)

    def update_summary(self, vals):
        for key, label in self.param_labels.items():
            if key in vals:
                label.setText(vals[key])
            else:
                label.setText("---")


class LeftPanel(QWidget):
    open_file = pyqtSignal()
    batch_process = pyqtSignal()
    recalculate = pyqtSignal()
    font_changed = pyqtSignal()
    he_porosity_changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.params = {}
        self.processor = None
        self.result = None
        self.file_label_text = "未加载文件"
        self.enforce_mono_check = None
        self._he_porosity_debounce = QTimer(self)
        self._he_porosity_debounce.setSingleShot(True)
        self._he_porosity_debounce.timeout.connect(self._emit_he_porosity_changed)
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG)
        layout.setSpacing(SPACING_XL)

        file_group = QGroupBox("文件操作")
        file_group.setStyleSheet(self._groupbox_style())
        file_layout = QHBoxLayout()
        file_layout.setSpacing(SPACING_MD)
        self.btn_open = QPushButton("打开文件")
        self.btn_open.setMinimumHeight(CONTROL_HEIGHT_MD)
        self.btn_open.setStyleSheet(self._button_style())
        self.btn_open.clicked.connect(self.open_file)
        self.btn_batch = QPushButton("批量处理")
        self.btn_batch.setMinimumHeight(CONTROL_HEIGHT_MD)
        self.btn_batch.setStyleSheet(self._button_style())
        self.btn_batch.clicked.connect(self.batch_process)
        file_layout.addWidget(self.btn_open)
        file_layout.addWidget(self.btn_batch)
        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        self.file_label = QLabel("未加载文件")
        self.file_label.setWordWrap(True)
        self.file_label.setStyleSheet(f"color: {COLOR_TEXT_HINT}; font-size: {font_settings.sm}px; padding: 0 2px;")
        layout.addWidget(self.file_label)

        layout.addWidget(self._separator())

        raw_group = QGroupBox("样品信息（原始数据）")
        raw_group.setStyleSheet(self._groupbox_style())
        raw_layout = QFormLayout()
        raw_layout.setSpacing(SPACING_SM)
        raw_layout.setVerticalSpacing(SPACING_MD)
        raw_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        raw_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.raw_labels = {}
        raw_configs = [
            ("sample_name", "样品名称"),
            ("sample_mass", "样品质量 (g)"),
            ("penetrometer_mass", "膨胀器质量 (g)"),
            ("assembly_mass", "注汞后质量 (g)"),
            ("penetrometer_volume", "膨胀器容积 (mL)"),
            ("points", "进汞/退汞点数"),
        ]
        for key, label_text in raw_configs:
            lbl = QLabel(label_text)
            lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
            val = QLabel("---")
            val.setTextInteractionFlags(Qt.TextSelectableByMouse)
            val.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.sm}px; font-weight: 600;")
            self.raw_labels[key] = val
            desc = RAW_PARAM_DESCRIPTIONS.get(key, (label_text, "暂无说明"))
            info_btn = _make_info_btn(self, desc[0], desc[1], small=True)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(SPACING_XS)
            row_layout.addWidget(lbl)
            row_layout.addWidget(val)
            row_layout.addWidget(info_btn)
            row_layout.addStretch(1)
            raw_layout.addRow(row)
        raw_group.setLayout(raw_layout)
        layout.addWidget(raw_group)

        layout.addWidget(self._separator())

        params_group = QGroupBox("计算参数")
        params_group.setStyleSheet(self._groupbox_style())
        params_layout = QFormLayout()
        params_layout.setSpacing(SPACING_SM)
        params_layout.setVerticalSpacing(SPACING_MD)
        params_layout.setLabelAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        params_layout.setFormAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.param_fields = {}
        param_configs = [
            ("contact_angle", "接触角 (°)", "130.0"),
            ("surface_tension", "表面张力 (N/m)", "0.475"),
            ("smoothing", "平滑窗口", "1"),
        ]
        for key, label_text, default in param_configs:
            lbl = QLabel(label_text)
            lbl.setFixedWidth(150)
            lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
            le = QLineEdit(default)
            le.setMinimumHeight(CONTROL_HEIGHT_SM)
            le.setStyleSheet(self._lineedit_style())
            self.param_fields[key] = le
            desc = CALC_PARAM_DESCRIPTIONS.get(key, (label_text, "暂无说明"))
            info_btn = _make_info_btn(self, desc[0], desc[1], small=True)
            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.setSpacing(SPACING_XS)
            row_layout.addWidget(lbl)
            row_layout.addWidget(le, 1)
            row_layout.addWidget(info_btn)
            params_layout.addRow(row)

        params_layout.addItem(QSpacerItem(1, SPACING_SM, QSizePolicy.Fixed, QSizePolicy.Fixed))
        enforce_lbl = QLabel("差分法修正单调性")
        enforce_lbl.setFixedWidth(150)
        enforce_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
        enforce_row = QWidget()
        enforce_layout = QHBoxLayout(enforce_row)
        enforce_layout.setContentsMargins(0, 0, 0, 0)
        enforce_layout.setSpacing(SPACING_XS)
        enforce_layout.addWidget(enforce_lbl)
        self.enforce_mono_check = QCheckBox()
        self.enforce_mono_check.setChecked(False)
        self.enforce_mono_check.setStyleSheet(f"spacing: 0px;")
        enforce_layout.addWidget(self.enforce_mono_check)
        en_label = QLabel("启用")
        en_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
        enforce_layout.addWidget(en_label)
        enforce_desc = CALC_PARAM_DESCRIPTIONS.get("enforce_mono", ("差分法修正单调性", "暂无说明"))
        enforce_info_btn = _make_info_btn(self, enforce_desc[0], enforce_desc[1], small=True)
        enforce_layout.addWidget(enforce_info_btn)
        enforce_layout.addStretch(1)
        params_layout.addRow(enforce_row)

        interp_lbl = QLabel("固定压力区间插值")
        interp_lbl.setFixedWidth(150)
        interp_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
        interp_row = QWidget()
        interp_layout = QHBoxLayout(interp_row)
        interp_layout.setContentsMargins(0, 0, 0, 0)
        interp_layout.setSpacing(SPACING_XS)
        interp_layout.addWidget(interp_lbl)
        self.interp_check = QCheckBox()
        self.interp_check.setChecked(True)
        self.interp_check.setStyleSheet(f"spacing: 0px;")
        interp_layout.addWidget(self.interp_check)
        il = QLabel("启用")
        il.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
        interp_layout.addWidget(il)
        interp_desc = CALC_PARAM_DESCRIPTIONS.get("use_fixed_interpolation", ("固定压力区间插值", "暂无说明"))
        interp_info_btn = _make_info_btn(self, interp_desc[0], interp_desc[1], small=True)
        interp_layout.addWidget(interp_info_btn)
        interp_layout.addStretch(1)
        params_layout.addRow(interp_row)

        self.enforce_mono_check.stateChanged.connect(self.on_enforce_mono_toggled)
        self.interp_check.stateChanged.connect(self.recalculate)

        range_row = QWidget()
        range_layout = QHBoxLayout(range_row)
        range_layout.setContentsMargins(0, 0, 0, 0)
        range_layout.setSpacing(SPACING_XS)
        range_lbl = QLabel("进汞压力范围 (psia):")
        range_lbl.setFixedWidth(150)
        range_lbl.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
        range_layout.addWidget(range_lbl)
        self.interp_p_min = QLineEdit("0.5")
        self.interp_p_min.setMinimumHeight(CONTROL_HEIGHT_SM)
        self.interp_p_min.setMaximumWidth(80)
        self.interp_p_min.setStyleSheet(self._lineedit_style())
        range_layout.addWidget(self.interp_p_min)
        dash = QLabel("  —  ")
        dash.setStyleSheet(f"color: {COLOR_TEXT_HINT}; font-size: {font_settings.sm}px;")
        range_layout.addWidget(dash)
        self.interp_p_max = QLineEdit("60000")
        self.interp_p_max.setMinimumHeight(CONTROL_HEIGHT_SM)
        self.interp_p_max.setMaximumWidth(90)
        self.interp_p_max.setStyleSheet(self._lineedit_style())
        range_layout.addWidget(self.interp_p_max)
        range_layout.addStretch(1)
        params_layout.addRow(range_row)

        he_por_label = QLabel("氦气孔隙度 (%):")
        he_por_label.setFixedWidth(150)
        he_por_label.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.sm}px;")
        self.he_porosity_input = QLineEdit()
        self.he_porosity_input.setPlaceholderText("留空则默认等于压汞孔隙度")
        self.he_porosity_input.setMinimumHeight(CONTROL_HEIGHT_SM)
        self.he_porosity_input.setStyleSheet(self._lineedit_style())
        self.he_porosity_input.textChanged.connect(self.on_he_porosity_changed)
        he_desc = CALC_PARAM_DESCRIPTIONS.get("he_porosity", ("氦气孔隙度 (%)", "暂无说明"))
        he_info_btn = _make_info_btn(self, he_desc[0], he_desc[1], small=True)
        he_row = QWidget()
        he_row_layout = QHBoxLayout(he_row)
        he_row_layout.setContentsMargins(0, 0, 0, 0)
        he_row_layout.setSpacing(SPACING_XS)
        he_row_layout.addWidget(he_por_label)
        he_row_layout.addWidget(self.he_porosity_input, 1)
        he_row_layout.addWidget(he_info_btn)
        params_layout.addRow(he_row)

        self.he_porosity_hint = QLabel("留空默认饱和度100%")
        self.he_porosity_hint.setWordWrap(True)
        self.he_porosity_hint.setMinimumHeight(28)
        self.he_porosity_hint.setStyleSheet(f"color: {COLOR_TEXT_HINT}; font-size: {font_settings.xs}px; padding-left: 2px;")
        params_layout.addRow(QLabel(""), self.he_porosity_hint)

        params_group.setLayout(params_layout)
        layout.addWidget(params_group)

        self.btn_recalc = QPushButton("重新计算")
        self.btn_recalc.setMinimumHeight(CONTROL_HEIGHT_MD)
        self.btn_recalc.setStyleSheet(self._button_style())
        self.btn_recalc.clicked.connect(self.recalculate)
        layout.addWidget(self.btn_recalc)

        layout.addWidget(self._separator())

        layout.addStretch(1)
        self.setLayout(layout)

    @staticmethod
    def _groupbox_style():
        return f"""
            QGroupBox {{
                color: {COLOR_PRIMARY}; font-weight: 600; font-size: {font_settings.lg}px;
                border: 1px solid {COLOR_BORDER}; border-radius: {RADIUS_LG}px;
                margin-top: 18px; padding-top: 10px;
                padding-left: {SPACING_LG}px; padding-right: {SPACING_LG}px;
                padding-bottom: {SPACING_MD}px; background-color: {COLOR_WHITE};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin; left: 14px;
                padding: 0 {SPACING_MD}px 0 {SPACING_MD}px; background-color: {COLOR_WHITE};
            }}
        """

    @staticmethod
    def _button_style():
        return f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY}; color: white;
                font-size: {font_settings.sm}px; font-weight: 500;
                border: none; border-radius: {RADIUS_MD}px;
                padding: 0 {SPACING_LG}px; min-height: {CONTROL_HEIGHT_MD}px;
            }}
            QPushButton:hover {{ background-color: {COLOR_PRIMARY_DARK}; }}
            QPushButton:pressed {{ background-color: #0d47a1; }}
            QPushButton:disabled {{ background-color: {COLOR_TEXT_HINT}; }}
        """

    @staticmethod
    def _lineedit_style():
        return f"""
            QLineEdit {{
                background-color: {COLOR_WHITE}; color: {COLOR_TEXT};
                font-size: {font_settings.sm}px; border: 1px solid {COLOR_BORDER};
                border-radius: {RADIUS_SM}px; padding: {SPACING_XS}px {SPACING_MD}px;
                min-height: {CONTROL_HEIGHT_SM}px;
            }}
            QLineEdit:focus {{ border-color: {COLOR_PRIMARY}; }}
        """

    @staticmethod
    def _separator():
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setFrameShadow(QFrame.Plain)
        line.setStyleSheet(f"background-color: {COLOR_BORDER};")
        line.setFixedHeight(1)
        return line

    def get_param(self, key):
        le = self.param_fields.get(key)
        if le:
            return le.text()
        return ""

    def set_file_label(self, text):
        self.file_label_text = text
        if hasattr(self, 'file_label') and self.file_label:
            self.file_label.setText(text)

    def update_sample_info(self, processor, result):
        self.processor = processor
        self.result = result
        d = processor.data
        r = result
        if hasattr(self, 'raw_labels') and self.raw_labels:
            self.raw_labels.get("sample_name", QLabel()).setText(str(d.sample_name) if d.sample_name else "---")
            self.raw_labels.get("sample_mass", QLabel()).setText(f"{d.sample_mass:.4f}" if d.sample_mass > 0 else "---")
            self.raw_labels.get("penetrometer_mass", QLabel()).setText(f"{d.penetrometer_mass:.4f}" if d.penetrometer_mass > 0 else "---")
            self.raw_labels.get("assembly_mass", QLabel()).setText(f"{d.assembly_mass:.4f}" if d.assembly_mass > 0 else "---")
            self.raw_labels.get("penetrometer_volume", QLabel()).setText(f"{d.penetrometer_volume:.4f}" if d.penetrometer_volume > 0 else "---")
            raw_int_pts = len(d.intrusion_pressure_psia)
            raw_ext_pts = len(d.extrusion_pressure_psia) if len(d.extrusion_pressure_psia) > 0 else 0
            self.raw_labels.get("points", QLabel()).setText(f"进汞 {raw_int_pts} / 退汞 {raw_ext_pts}")
        if hasattr(self, 'he_porosity_input') and self.he_porosity_input:
            self.he_porosity_input.blockSignals(True)
            self.he_porosity_input.setText(f"{r.cal_porosity:.4f}" if r.cal_porosity > 0 else "")
            self.he_porosity_input.blockSignals(False)
            self.on_he_porosity_changed(self.he_porosity_input.text())

    def on_he_porosity_changed(self, text):
        if not hasattr(self, 'he_porosity_input') or not self.he_porosity_input:
            return
        text = text.strip()
        if not text:
            self.he_porosity_input.setStyleSheet(self._lineedit_style())
            self.he_porosity_hint.setText("留空默认饱和度100%")
            self.he_porosity_hint.setStyleSheet(f"color: {COLOR_TEXT_HINT}; font-size: {font_settings.xs}px; padding-left: 2px;")
            if hasattr(self, 'processor') and self.processor:
                self._he_porosity_debounce.start(400)
            return
        try:
            val = float(text)
            if val <= 0:
                raise ValueError
        except (ValueError, TypeError):
            self.he_porosity_input.setStyleSheet(self._lineedit_error_style())
            self.he_porosity_hint.setText("请输入有效的正数")
            self.he_porosity_hint.setStyleSheet(f"color: {COLOR_ERROR}; font-size: {font_settings.xs}px; padding-left: 2px;")
            return
        if not hasattr(self, 'result') or not self.result:
            self.he_porosity_input.setStyleSheet(self._lineedit_style())
            self.he_porosity_hint.setText("输入值需大于压汞孔隙度")
            self.he_porosity_hint.setStyleSheet(f"color: {COLOR_TEXT_HINT}; font-size: {font_settings.xs}px; padding-left: 2px;")
            if hasattr(self, 'processor') and self.processor:
                self._he_porosity_debounce.start(400)
            return
        micp_por = self.result.cal_porosity
        if val < micp_por:
            self.he_porosity_input.setStyleSheet(self._lineedit_error_style())
            self.he_porosity_hint.setText(f"氦气孔隙度({val:.4f}%)须≥压汞孔隙度({micp_por:.4f}%)")
            self.he_porosity_hint.setStyleSheet(f"color: {COLOR_ERROR}; font-size: {font_settings.xs}px; padding-left: 2px;")
            return
        self.he_porosity_input.setStyleSheet(self._lineedit_style())
        if val == micp_por:
            self.he_porosity_hint.setText("进汞饱和度 = 100%")
            self.he_porosity_hint.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.xs}px; padding-left: 2px;")
        else:
            sat_adjusted = micp_por / val * 100
            self.he_porosity_hint.setText(f"进汞饱和度 = {sat_adjusted:.2f}%")
            self.he_porosity_hint.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.xs}px; padding-left: 2px;")
        if hasattr(self, 'processor') and self.processor:
            self._he_porosity_debounce.start(400)

    def _emit_he_porosity_changed(self):
        self.he_porosity_changed.emit()

    def get_he_porosity(self):
        if not hasattr(self, 'he_porosity_input') or not self.he_porosity_input:
            return None
        text = self.he_porosity_input.text().strip()
        if not text:
            return None
        try:
            val = float(text)
            if hasattr(self, 'result') and self.result and val >= self.result.cal_porosity:
                return val
        except (ValueError, TypeError):
            pass
        return None

    def get_enforce_monotonic(self):
        if self.enforce_mono_check:
            return self.enforce_mono_check.isChecked()
        return True

    def get_use_fixed_interpolation(self):
        if self.interp_check:
            return self.interp_check.isChecked()
        return True

    def get_interp_p_min(self):
        if hasattr(self, 'interp_p_min') and self.interp_p_min:
            try:
                return float(self.interp_p_min.text().strip())
            except (ValueError, TypeError):
                pass
        return 0.5

    def get_interp_p_max(self):
        if hasattr(self, 'interp_p_max') and self.interp_p_max:
            try:
                return float(self.interp_p_max.text().strip())
            except (ValueError, TypeError):
                pass
        return 60000.0

    def on_enforce_mono_toggled(self):
        self.recalculate.emit()

    def _lineedit_error_style(self):
        return f"""
            QLineEdit {{
                background-color: #fff5f5; color: {COLOR_TEXT};
                font-size: {font_settings.sm}px; border: 1px solid {COLOR_ERROR};
                border-radius: {RADIUS_SM}px; padding: {SPACING_XS}px {SPACING_MD}px;
                min-height: {CONTROL_HEIGHT_SM}px;
            }}
            QLineEdit:focus {{ border-color: {COLOR_ERROR}; }}
        """

    def refresh_fonts(self):
        self.init_ui()


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setFixedSize(400, 240)
        self.setStyleSheet(f"QDialog {{ background-color: {COLOR_BG}; }}")
        self.font_scale = 1.0
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(SPACING_XL, SPACING_XL, SPACING_XL, SPACING_XL)
        layout.setSpacing(SPACING_XL)
        font_group = QGroupBox("字体大小")
        font_group.setStyleSheet(LeftPanel._groupbox_style())
        grid_layout = QGridLayout()
        grid_layout.setSpacing(SPACING_LG)
        scales = [("小号", 0.85), ("中号", 1.0), ("大号", 1.15)]
        self.radio_buttons = []
        for i, (label, scale) in enumerate(scales):
            rb = QRadioButton(label)
            rb.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.base}px; spacing: 6px;")
            rb.scale = scale
            if scale == 1.0:
                rb.setChecked(True)
                self.font_scale = scale
            rb.toggled.connect(self.on_scale_change)
            self.radio_buttons.append(rb)
            grid_layout.addWidget(rb, 0, i)
        font_group.setLayout(grid_layout)
        layout.addWidget(font_group)
        button_layout = QHBoxLayout()
        button_layout.setSpacing(SPACING_MD)
        self.btn_ok = QPushButton("确定")
        self.btn_ok.setMinimumHeight(CONTROL_HEIGHT_MD)
        self.btn_ok.setStyleSheet(f"""
            QPushButton {{ background-color: {COLOR_PRIMARY}; color: white;
                font-size: {font_settings.base}px; font-weight: 500; border: none;
                border-radius: {RADIUS_MD}px; padding: 0 {SPACING_LG}px; min-width: 80px; }}
            QPushButton:hover {{ background-color: {COLOR_PRIMARY_DARK}; }}
        """)
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setMinimumHeight(CONTROL_HEIGHT_MD)
        self.btn_cancel.setStyleSheet(f"""
            QPushButton {{ background-color: {COLOR_PRIMARY}; color: white;
                font-size: {font_settings.base}px; font-weight: 500; border: none;
                border-radius: {RADIUS_MD}px; padding: 0 {SPACING_LG}px; min-width: 80px; }}
            QPushButton:hover {{ background-color: {COLOR_PRIMARY_DARK}; }}
        """)
        self.btn_cancel.clicked.connect(self.reject)
        button_layout.addStretch(1)
        button_layout.addWidget(self.btn_ok)
        button_layout.addWidget(self.btn_cancel)
        layout.addLayout(button_layout)
        self.setLayout(layout)

    def on_scale_change(self):
        for rb in self.radio_buttons:
            if rb.isChecked():
                self.font_scale = rb.scale
                break

    def get_font_scale(self):
        return self.font_scale


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.processor = None
        self.result = None
        self.batch_results = []
        self.current_chart = CHARTS[0][0]
        self.chart_cards = {}
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("MICP 压汞法数据处理系统")
        self.setMinimumSize(1280, 720)
        self.setStyleSheet(self._main_style())
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QHBoxLayout(self.central_widget)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        self.create_menu()
        self.create_layout()
        self.statusBar().setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.sm}px; background-color: {COLOR_PRIMARY_LIGHT}; padding: 2px 8px;")
        self.statusBar().hide()

    def _main_style(self):
        return f"""
            QMainWindow {{ background-color: {COLOR_BG}; }}
            QWidget {{ background-color: {COLOR_BG}; color: {COLOR_TEXT};
                font-family: "Microsoft YaHei", "PingFang SC", "Helvetica Neue", sans-serif;
                font-size: {font_settings.base}px; }}
            QScrollArea {{ border: none; }}
            QLabel {{ font-size: {font_settings.base}px; }}
            QMenuBar {{ background-color: {COLOR_WHITE}; border-bottom: 1px solid {COLOR_BORDER}; padding: 2px; }}
            QMenuBar::item {{ padding: {SPACING_SM}px {SPACING_LG}px; color: {COLOR_TEXT}; }}
            QMenuBar::item:selected {{ background-color: {COLOR_PRIMARY_LIGHT}; color: {COLOR_PRIMARY}; }}
            QMenu {{ background-color: {COLOR_WHITE}; border: 1px solid {COLOR_BORDER}; padding: {SPACING_XS}px; }}
            QMenu::item {{ padding: {SPACING_SM}px {SPACING_XL}px; color: {COLOR_TEXT}; }}
            QMenu::item:selected {{ background-color: {COLOR_PRIMARY_LIGHT}; color: {COLOR_PRIMARY}; }}
        """

    def create_menu(self):
        menubar = self.menuBar()
        file_menu = QMenu("文件", self)
        open_action = QAction("打开文件", self)
        open_action.triggered.connect(self.show_open_file)
        batch_action = QAction("批量处理", self)
        batch_action.triggered.connect(self.show_batch)
        exit_action = QAction("退出", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(open_action)
        file_menu.addAction(batch_action)
        file_menu.addSeparator()
        file_menu.addAction(exit_action)
        settings_menu = QMenu("设置", self)
        font_action = QAction("字体设置", self)
        font_action.triggered.connect(self.show_settings)
        settings_menu.addAction(font_action)
        menubar.addMenu(file_menu)
        menubar.addMenu(settings_menu)

    def create_layout(self):
        splitter = QSplitter(Qt.Horizontal)
        splitter.setStyleSheet("QSplitter::handle { background-color: #e0e0e0; width: 3px; }")
        self.main_layout.addWidget(splitter)
        self._build_left_panel(splitter)
        self._build_center_panel(splitter)
        self._build_right_panel(splitter)
        splitter.setStretchFactor(0, 12)
        splitter.setStretchFactor(1, 20)
        splitter.setStretchFactor(2, 16)

    def _build_left_panel(self, splitter):
        self.left_panel = LeftPanel()
        self._update_left_panel_width()
        self.left_panel.open_file.connect(self.show_open_file)
        self.left_panel.batch_process.connect(self.show_batch)
        self.left_panel.recalculate.connect(self.recalc)
        self.left_panel.he_porosity_changed.connect(self.update_all_charts)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(self.left_panel)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setStyleSheet("QScrollArea { border: none; background-color: #f5f5f5; }")
        self.left_scroll = left_scroll
        splitter.addWidget(left_scroll)

    def _update_left_panel_width(self):
        base_width = LEFT_PANEL_WIDTH
        new_width = int(base_width * font_settings.base / FONT_BASE)
        self.left_panel.setMinimumWidth(new_width)
        self.left_panel.setMaximumWidth(new_width + 80)
        if hasattr(self, 'left_scroll') and self.left_scroll:
            self.left_scroll.setMinimumWidth(new_width)
            self.left_scroll.setMaximumWidth(new_width + 80)

    def _build_center_panel(self, splitter):
        center_panel = QWidget()
        center_panel.setStyleSheet("background-color: #ffffff;")
        center_layout = QVBoxLayout(center_panel)
        center_layout.setContentsMargins(SPACING_LG, SPACING_LG, SPACING_LG, SPACING_LG)
        center_layout.setSpacing(SPACING_MD)

        header = QLabel("绘图数据")
        header.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.lg}px; font-weight: 700; padding: 0;")
        center_layout.addWidget(header)

        self.chart_data_label = QLabel("点击右侧图表查看对应数据")
        self.chart_data_label.setWordWrap(True)
        self.chart_data_label.setStyleSheet(f"color: {COLOR_TEXT_HINT}; font-size: {font_settings.sm}px; padding: 0 2px;")
        center_layout.addWidget(self.chart_data_label)

        data_group = QGroupBox("数据表格")
        data_group.setStyleSheet(LeftPanel._groupbox_style())
        data_group_layout = QVBoxLayout()
        data_group_layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        data_group_layout.setSpacing(SPACING_SM)

        self.data_table = QTableWidget()
        self.data_table.setStyleSheet(self._table_style())
        self.data_table.verticalHeader().setDefaultSectionSize(20)
        self.data_table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.data_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.data_table.setAlternatingRowColors(True)
        self.data_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.data_table.setWordWrap(True)
        self.data_table.horizontalHeader().setMinimumSectionSize(50)
        self.data_table.verticalHeader().setVisible(True)
        self.data_table.horizontalHeader().sectionClicked.connect(self._on_header_clicked_select_all)
        self.data_table.horizontalHeader().setContextMenuPolicy(Qt.CustomContextMenu)
        self.data_table.horizontalHeader().customContextMenuRequested.connect(self._header_context_menu)
        copy_shortcut = QKeySequence("Ctrl+C")
        from PyQt5.QtWidgets import QShortcut
        self._copy_shortcut = QShortcut(copy_shortcut, self.data_table)
        self._copy_shortcut.activated.connect(self._copy_selected_table_data)
        data_group_layout.addWidget(self.data_table, 1)
        btn_copy_table = QPushButton("复制数据")
        btn_copy_table.setMinimumHeight(CONTROL_HEIGHT_SM)
        btn_copy_table.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY}; color: white;
                font-size: {font_settings.xs}px; font-weight: 500;
                border: none; border-radius: {RADIUS_SM}px;
                padding: 0 {SPACING_MD}px; min-height: {CONTROL_HEIGHT_SM}px;
            }}
            QPushButton:hover {{ background-color: {COLOR_PRIMARY_DARK}; }}
        """)
        btn_copy_table.setCursor(Qt.PointingHandCursor)
        btn_copy_table.clicked.connect(self._copy_all_table_data)
        data_group_layout.addWidget(btn_copy_table)
        data_group.setLayout(data_group_layout)
        center_layout.addWidget(data_group, 1)

        splitter.addWidget(center_panel)

    def _build_right_panel(self, splitter):
        right_container = QWidget()
        right_container.setStyleSheet("background-color: #f5f5f5;")
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)

        right_scroll = QScrollArea()
        right_scroll.setWidgetResizable(True)
        right_scroll.setStyleSheet("QScrollArea { border: none; background-color: #f5f5f5; }")

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background-color: #f5f5f5;")
        self.right_scroll_layout = QVBoxLayout(scroll_content)
        self.right_scroll_layout.setContentsMargins(SPACING_MD, SPACING_MD, SPACING_MD, SPACING_MD)
        self.right_scroll_layout.setSpacing(SPACING_LG)

        self.summary_report = SummaryReport()
        self.right_scroll_layout.addWidget(self.summary_report)

        self.chart_cards = {}
        for chart_id, chart_name in CHARTS:
            divider = QFrame()
            divider.setFrameShape(QFrame.HLine)
            divider.setStyleSheet(f"background-color: {COLOR_PRIMARY}; max-height: 2px;")
            self.right_scroll_layout.addWidget(divider)

            card = ChartCard(chart_id, chart_name)
            card.chart_clicked.connect(self.on_chart_clicked)
            self.chart_cards[chart_id] = card
            self.right_scroll_layout.addWidget(card)

        self.right_scroll_layout.addStretch(1)
        right_scroll.setWidget(scroll_content)
        right_layout.addWidget(right_scroll)

        splitter.addWidget(right_container)

    def on_chart_clicked(self, chart_id):
        self.current_chart = chart_id
        for cid, card in self.chart_cards.items():
            card.set_selected(cid == chart_id)
        self.update_data_table()

        for cid, card in self.chart_cards.items():
            if cid == chart_id:
                card.ensurePolished()
                break

    def _safe(self, x, y, positive_x=False, nonneg_y=False):
        v = np.isfinite(x) & np.isfinite(y)
        if positive_x:
            v = v & (x > 0)
        if nonneg_y:
            v = v & (y >= 0)
        return x[v].tolist(), y[v].tolist()

    def chart_axes(self, tid):
        axes = {
            "capillary": ("S_Hg (%)", "P (MPa)", False, True),
            "dvdlogD": ("d (nm)", "dV/dlogD (mL/g)", True, True),
            "pct": ("d (nm)", "S_int (%)", True, True),
            "inj_ext": ("P (MPa)", "$V_{cum}$ (mL/g)", True, False),
            "interpolated": ("P (MPa)", "$V_{cum}$ (mL/g)", True, False),
            "characteristic": ("d ($\\mu$m)", "$S_{cum}$ (%)", True, True),
            "fractal": ("$\\log P$", "$\\log(1-S)$", False, False),
            "ratio": ("$S_{Hg}$ (%)", "孔喉比", False, False),
            "swanson_matrix": ("d ($\\mu$m)", "$S \\times D^3$", True, True),
            "swanson_pore": ("d ($\\mu$m)", "$S \\times D^3$", True, True),
        }
        return axes.get(tid, ("X", "Y", False, False))

    def get_display_sat_scale(self):
        he_por = self.left_panel.get_he_porosity()
        if he_por is not None and self.result and self.result.cal_porosity > 0:
            return self.result.cal_porosity / he_por
        return 1.0

    def get_chart_param_vals(self, r):
        vals = {
            'he_por': f"{r._he_porosity:.4f}",
            'micp_por': f"{r.cal_porosity:.4f}",
            'bulk_d': f"{r.bulk_density:.4f}",
            'skel_d': f"{r.skeletal_density:.4f}",
            'ssa': f"{r.specific_surface_area:.4f}",
            'pore_v': f"{r.pore_volume:.6f}",
            'total_pore_area': f"{r.total_pore_area:.4f}" if r.total_pore_area > 0 else "---",
            'avg_pd': f"{r.avg_pore_diameter_nm:.2f}" if r.avg_pore_diameter_nm > 0 else "---",
            'inj_sat': f"{r.intrusion_saturation:.2f}",
            'eff': f"{r.efficiency:.4f}",
            'disp_p': f"{r.displacement_pressure:.4f}",
            'max_d': f"{r.max_pore_diameter_um:.4f}",
            'med_p': f"{r.median_pressure:.4f}",
            'med_d': f"{r.median_diameter_um:.6f}",
            'med_dvol': f"{r.median_pore_diameter_volume_nm:.2f}" if r.median_pore_diameter_volume_nm > 0 else "---",
            'med_darea': f"{r.median_pore_diameter_area_nm:.2f}" if r.median_pore_diameter_area_nm > 0 else "---",
            'sp': f"{r.sorting_coefficient:.4f}",
            'skp': f"{r.skewness:.4f}",
            'kp': f"{r.kurtosis:.4f}",
            'dm': f"{r.mean_radius:.4f}",
            'phi': f"{r.structure_coefficient:.4f}",
            'd_coeff': f"{r.relative_sorting_coeff:.4f}",
            'frac_d': f"{r.fractal_dimensions[0]:.4f}" if r.fractal_dimensions else "---",
            'cl': f"{r.characteristic_length_nm:.2f}" if r.characteristic_length_nm > 0 else "---",
            'cff': f"{r.conductivity_formation_factor:.4f}" if r.conductivity_formation_factor > 0 else "---",
            'tf': f"{r.tortuosity_factor:.4f}" if r.tortuosity_factor > 0 else "---",
            'tort': f"{r.tortuosity:.4f}" if r.tortuosity > 0 else "---",
            'bpr': f"{r.breakthrough_pressure_ratio:.4f}" if r.breakthrough_pressure_ratio > 0 else "---",
            'k413': f"{r.permeability_413:.6e}",
            'k10': f"{r.permeability_10:.6e}",
            'n_int': str(r.n_intrusion_points),
            'n_ext': str(r.n_withdrawal_points),
            'smoothing': self.left_panel.get_param("smoothing"),
            'n_bins': str(len(r.pore_throat_bins)),
            'max_pct': f"{np.max(r.bin_pct):.2f}" if len(r.bin_pct) > 0 else "---",
            'mf_D0': f"{r.mf_D0:.4f}" if r.mf_D0 > 0 else "---",
            'mf_D1': f"{r.mf_D1:.4f}" if r.mf_D1 > 0 else "---",
            'mf_D2': f"{r.mf_D2:.4f}" if r.mf_D2 > 0 else "---",
            'mf_D_neg10': f"{r.mf_D_neg10:.4f}" if r.mf_D_neg10 != 0 else "---",
            'mf_D_10': f"{r.mf_D_10:.4f}" if r.mf_D_10 != 0 else "---",
            'mf_delta_alpha': f"{r.mf_delta_alpha:.4f}" if r.mf_delta_alpha > 0 else "---",
            'mf_delta_f': f"{r.mf_delta_f:.4f}" if r.mf_delta_f != 0 else "---",
            'mf_a_min': f"{r.mf_a_min:.4f}" if r.mf_a_min != 0 else "---",
            'mf_a_max': f"{r.mf_a_max:.4f}" if r.mf_a_max != 0 else "---",
            'mf_D_a': f"{r.mf_D_a:.4f}" if r.mf_D_a > 0 else "---",
            'mf_R_d': f"{r.mf_R_d:.4f}" if r.mf_R_d != 0 else "---",
            'mf_D_Fa': f"{r.mf_D_Fa:.4f}" if r.mf_D_Fa != 0 else "---",
            'mf_H': f"{r.mf_H:.4f}" if r.mf_H > 0 else "---",
            'mf_D_neg10_minus_D_10': f"{r.mf_D_neg10_minus_D_10:.4f}" if r.mf_D_neg10_minus_D_10 != 0 else "---",
        }
        return vals

    def update_all_charts(self):
        if not self.result:
            return
        r = self.result
        vals = self.get_chart_param_vals(r)
        self.summary_report.update_summary(vals)

        for chart_id, card in self.chart_cards.items():
            card.canvas.clear_plot()
            xlabel, ylabel, x_log, x_inv = self.chart_axes(chart_id)
            if chart_id == "capillary":
                card.canvas.update_axes("$S_{Hg}$ (%)", "", y_log=True)
            else:
                card.canvas.update_axes(xlabel, ylabel, x_log, x_inv)
            self.draw_on_canvas(chart_id, card.canvas)
            if chart_id == "characteristic":
                card.canvas.fig.subplots_adjust(top=0.92, bottom=0.16, left=0.15, right=0.88)
            else:
                card.canvas.add_legend()
            if chart_id == "fractal":
                card.canvas.fig.set_size_inches(6.5, 5.0)
                card.canvas.fig.subplots_adjust(top=0.72, bottom=0.16, left=0.15, right=0.88)
            card.update_params(vals)
            card.canvas.draw_idle()

        self.update_data_table()

    def draw_on_canvas(self, tid, canvas):
        r = self.result
        if not r or len(r.cum_intrusion) < 2:
            return
        dispatch = {
            "capillary": self.draw_capillary,
            "dvdlogD": self.draw_dvdlogD,
            "pct": self.draw_pct,
            "inj_ext": self.draw_inj_ext,
            "interpolated": self.draw_interpolated,
            "characteristic": self.draw_characteristic,
            "fractal": self.draw_fractal,
            "mf_spectrum": self.draw_mf_spectrum,
            "mf_Dq": self.draw_mf_Dq,
            "mf_tau": self.draw_mf_tau,
            "ratio": self.draw_ratio,
            "swanson_matrix": self.draw_swanson_matrix,
            "swanson_pore": self.draw_swanson_pore,
        }
        fn = dispatch.get(tid)
        if fn:
            fn(r, canvas)

    def draw_capillary(self, r, canvas):
        cum_max = np.max(r.cum_intrusion)
        sat_scale = self.get_display_sat_scale()
        sat = r.cum_intrusion / cum_max * 100.0 * sat_scale if cum_max > 0 else r.cum_intrusion
        pressure = r.intrusion_pressure_mpa
        x, y = self._safe(sat, pressure, nonneg_y=True)
        if x:
            canvas.draw_line(x, y, "$V_{in}$", CHART_PALETTE[0], marker='o')
        if len(r.extrusion_pressure_mpa) > 0 and len(r.cum_extrusion) > 0:
            ext_sat = r.cum_extrusion / cum_max * 100.0 * sat_scale if cum_max > 0 else r.cum_extrusion
            x2, y2 = self._safe(ext_sat, r.extrusion_pressure_mpa, nonneg_y=True)
            if x2:
                canvas.draw_line(x2, y2, "$V_{out}$", CHART_PALETTE[1], marker='s')
        canvas.axes.set_xlim(100, 0)
        canvas._cap_sat_scale = sat_scale
        canvas.axes.yaxis.set_visible(False)
        if hasattr(self, 'processor') and hasattr(self.processor, 'params'):
            sigma_p = self.processor.params.surface_tension
            theta_p = self.processor.params.contact_angle
            theta_rad = np.radians(theta_p)
            const = -2 * sigma_p * np.cos(theta_rad) * 1000
            def p_to_d_nm(p):
                a = np.asarray(p, dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(a > 0, const / a, np.nan)
            def d_nm_to_p(d):
                a = np.asarray(d, dtype=float)
                with np.errstate(divide='ignore', invalid='ignore'):
                    return np.where(a > 0, const / a, np.nan)
            twin_left = canvas.axes.secondary_yaxis('left', functions=(p_to_d_nm, d_nm_to_p))
            twin_left.set_ylabel('r (nm)', color=CHART_PALETTE[2], fontsize=font_settings.md, fontweight=500, labelpad=10)
            twin_left.tick_params(colors=CHART_PALETTE[2], labelsize=font_settings.sm, length=5, width=1, pad=8)
            canvas.twin_y = canvas.axes.secondary_yaxis('right', functions=(lambda x: x, lambda x: x))
            canvas.twin_y.set_ylabel('P (MPa)', color=COLOR_SECONDARY, fontsize=font_settings.md, fontweight=500, labelpad=10)
            canvas.twin_y.tick_params(colors=COLOR_TEXT_SECONDARY, labelsize=font_settings.sm, length=5, width=1, pad=8)
            fit = getattr(r, '_disp_p_fit_data', None)
            if fit and fit.get('sat_interp'):
                fit_x = np.array(fit['sat_interp']) * sat_scale
                fit_y = 10.0 ** np.array(fit['log_p_interp'])
                canvas.axes.plot(fit_x, fit_y, color=CHART_PALETTE[2], linestyle='--', linewidth=2.0, alpha=0.9,
                                label=f"$P_d$={r.displacement_pressure:.4f} MPa ($R^2$={fit['r_squared']:.4f})", zorder=4)
                if fit.get('sat_low') is not None and fit.get('sat_high') is not None:
                    sat_low_s = float(fit['sat_low']) * sat_scale
                    sat_high_s = float(fit['sat_high']) * sat_scale
                    canvas.axes.axvspan(sat_low_s, sat_high_s, alpha=0.08, color=CHART_PALETTE[2], zorder=1)
                    y_lim = canvas.axes.get_ylim()
                    y_mid = float(np.exp((np.log(max(y_lim[0], 1e-6)) + np.log(y_lim[1])) / 2))
                    canvas._disp_marker_y = y_mid
                    canvas.axes.plot([sat_low_s, sat_low_s], [y_lim[0], y_mid],
                                   color=CHART_PALETTE[2], linestyle='--', alpha=0.5, linewidth=1.2, zorder=9)
                    canvas.axes.plot([sat_high_s, sat_high_s], [y_lim[0], y_mid],
                                   color=CHART_PALETTE[3], linestyle='--', alpha=0.5, linewidth=1.2, zorder=9)
                    canvas._disp_low_marker = canvas.axes.scatter(
                        [sat_low_s], [y_mid], marker='v', s=200,
                        color=CHART_PALETTE[2], edgecolors='white', linewidths=1.5,
                        zorder=12, picker=10, label='_nolegend_')
                    canvas._disp_high_marker = canvas.axes.scatter(
                        [sat_high_s], [y_mid], marker='v', s=200,
                        color=CHART_PALETTE[3], edgecolors='white', linewidths=1.5,
                        zorder=12, picker=10, label='_nolegend_')
                    card = canvas.parent()
                    if isinstance(card, ChartCard):
                        card._ensure_disp_drag_events()

    def draw_dvdlogD(self, r, canvas):
        x, y = self._safe(r.pore_throat_diameter_nm, r.dv_dlogD, positive_x=True, nonneg_y=True)
        if x:
            canvas.draw_line(x, y, "$dV/d\\log D$", CHART_PALETTE[1], marker='v')

    def draw_pct(self, r, canvas):
        if len(r.pore_throat_bins) == 0 or len(r.bin_pct) == 0:
            return
        pct = r.bin_pct
        if not hasattr(r, '_bin_edges') or len(r._bin_edges) < 2:
            return
        edges = r._bin_edges
        centers = r.pore_throat_bins
        nonzero = pct > 0
        if not np.any(nonzero):
            return
        x = centers[nonzero]
        y = pct[nonzero]
        widths = np.diff(edges)[nonzero]
        canvas.axes.bar(x, y, width=widths, color=CHART_PALETTE[4],
                        alpha=CHART_ALPHA, edgecolor='white', linewidth=0.5,
                        label=f"$S_{{int}}$ (%) - {np.sum(nonzero)}区间")
        canvas.axes.set_xscale('log')

    def draw_inj_ext(self, r, canvas):
        if len(self.batch_results) > 1:
            self._draw_batch_inj_ext(canvas)
            return
        d = self.processor.data
        psia_to_mpa = 1.0 / 145.036
        int_p = d.intrusion_pressure_psia * psia_to_mpa
        int_cv = d.intrusion_cumulative_mlg
        x, y = self._safe(int_p, int_cv)
        if x:
            canvas.draw_line(x, y, "$V_{in}$", CHART_PALETTE[0], marker='o')
        if len(d.extrusion_pressure_psia) > 0 and len(d.extrusion_cumulative_mlg) > 0:
            ext_p = d.extrusion_pressure_psia * psia_to_mpa
            ext_cv = d.extrusion_cumulative_mlg
            x2, y2 = self._safe(ext_p, ext_cv)
            if x2:
                canvas.draw_line(x2, y2, "$V_{out}$", CHART_PALETTE[1], marker='s')
        canvas._add_pore_diam_twin_x(self.processor.params.surface_tension, self.processor.params.contact_angle)

    def draw_interpolated(self, r, canvas):
        psia_to_mpa = 1.0 / 145.036
        d = self.processor.data
        raw_p = d.intrusion_pressure_psia * psia_to_mpa
        raw_cv = d.intrusion_cumulative_mlg
        use_interp = self.processor.params.use_fixed_interpolation
        label_int = "插值" if use_interp else "$V_{in}$"
        x, y = self._safe(r.intrusion_pressure_mpa, r.cum_intrusion)
        if x:
            lw = CHART_LINEWIDTH + 1 if use_interp else CHART_LINEWIDTH
            canvas.axes.plot(x, y, label=f"{label_int} ({len(x)}pt)", color=CHART_PALETTE[0],
                           linewidth=lw, alpha=1.0, zorder=3)
        rx, ry = self._safe(raw_p, raw_cv)
        if rx:
            canvas.axes.scatter(rx, ry, label=f"原始 ({len(rx)}pt)", color=CHART_PALETTE[0],
                              s=CHART_MARKERSIZE**2, edgecolors=CHART_PALETTE[0],
                              facecolors='none', linewidths=0.8, alpha=0.55, zorder=2)
        if len(r.extrusion_pressure_mpa) > 0 and len(r.cum_extrusion) > 0:
            label_ext = "插值" if use_interp else "$V_{out}$"
            x2, y2 = self._safe(r.extrusion_pressure_mpa, r.cum_extrusion)
            if x2:
                lw2 = CHART_LINEWIDTH + 1 if use_interp else CHART_LINEWIDTH
                canvas.axes.plot(x2, y2, label=f"{label_ext} ({len(x2)}pt)", color=CHART_PALETTE[1],
                                linewidth=lw2, alpha=1.0, zorder=3)
        if len(d.extrusion_pressure_psia) > 0 and len(d.extrusion_cumulative_mlg) > 0:
            raw_ep = d.extrusion_pressure_psia * psia_to_mpa
            raw_ecv = d.extrusion_cumulative_mlg
            rx2, ry2 = self._safe(raw_ep, raw_ecv)
            if rx2:
                canvas.axes.scatter(rx2, ry2, label=f"原始 ({len(rx2)}pt)", color=CHART_PALETTE[1],
                                  s=CHART_MARKERSIZE**2, edgecolors=CHART_PALETTE[1],
                                  facecolors='none', linewidths=0.8, alpha=0.55, zorder=2)
        canvas._add_pore_diam_twin_x(self.processor.params.surface_tension, self.processor.params.contact_angle)

    def _draw_batch_inj_ext(self, canvas):
        markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h', '8', '<', '>', 'P', 'X', 'H', '+']
        psia_to_mpa = 1.0 / 145.036
        for idx, item in enumerate(self.batch_results):
            if 'error' in item:
                continue
            d = item['data']
            name = item['name']
            color = CHART_PALETTE[idx % len(CHART_PALETTE)]
            marker = markers[idx % len(markers)]
            int_p = d.intrusion_pressure_psia * psia_to_mpa
            int_cv = d.intrusion_cumulative_mlg
            x, y = self._safe(int_p, int_cv)
            if x:
                canvas.draw_line(x, y, f"{name} $V_{{in}}$", color, marker=marker, linewidth=CHART_LINEWIDTH)
            if len(d.extrusion_pressure_psia) > 0 and len(d.extrusion_cumulative_mlg) > 0:
                ext_p = d.extrusion_pressure_psia * psia_to_mpa
                ext_cv = d.extrusion_cumulative_mlg
                x2, y2 = self._safe(ext_p, ext_cv)
                if x2:
                    canvas.draw_line(x2, y2, f"{name} $V_{{out}}$", color, marker=marker, linewidth=CHART_LINEWIDTH * 0.7)
        canvas._add_pore_diam_twin_x(self.processor.params.surface_tension, self.processor.params.contact_angle)

    def draw_characteristic(self, r, canvas):
        cum_max = np.max(r.cum_intrusion)
        if cum_max <= 0:
            return
        sat = r.cum_intrusion / cum_max * 100
        d_um = r.pore_throat_diameter_um
        x, y = self._safe(d_um, sat, positive_x=True, nonneg_y=True)
        if x:
            canvas.axes.plot(x, y, label="$S_{cum}$", color=CHART_PALETTE[0],
                           linewidth=CHART_LINEWIDTH, marker='D',
                           markersize=max(CHART_MARKERSIZE, 4), markeredgecolor=CHART_PALETTE[0],
                           markerfacecolor='white', markeredgewidth=1.2, alpha=CHART_ALPHA)
        canvas.axes.set_ylim(0, 100)

        percentile_groups = [
            ([84, 16], CHART_PALETTE[1], "$\\Psi_{84},\\Psi_{16}$ ($S_p$)"),
            ([95, 5], CHART_PALETTE[2], "$\\Psi_{95},\\Psi_{5}$ ($S_p$)"),
        ]
        unlabeled_percentiles = [(50, CHART_PALETTE[3], "$\\Psi_{50}$ ($D_M$)"),
                                 (75, CHART_PALETTE[4]), (25, CHART_PALETTE[4])]
        d_valid = np.array(d_um)
        s_valid = np.array(sat)
        sort_idx = np.argsort(d_valid)
        d_sorted = d_valid[sort_idx]
        s_sorted = s_valid[sort_idx]

        for pcts, color, label in percentile_groups:
            added = False
            for pct in pcts:
                if np.min(s_sorted) <= pct <= np.max(s_sorted):
                    d_at_pct = float(np.interp(pct, s_sorted, d_sorted))
                    if d_at_pct > 0 and np.isfinite(d_at_pct):
                        canvas.axes.axhline(y=pct, color=color, linestyle='--',
                                            linewidth=0.8, alpha=0.45)
                        lbl = label if not added else '_nolegend_'
                        canvas.axes.scatter([d_at_pct], [float(pct)], color=color, s=80,
                                          edgecolors='white', linewidths=0.8,
                                          alpha=0.85, zorder=5, label=lbl)
                        added = True

        for item in unlabeled_percentiles:
            pct = item[0] if isinstance(item, tuple) else item
            color = item[1] if isinstance(item, tuple) else CHART_PALETTE[3]
            is_labeled = isinstance(item, tuple) and len(item) > 2
            if np.min(s_sorted) <= pct <= np.max(s_sorted):
                d_at_pct = float(np.interp(pct, s_sorted, d_sorted))
                if d_at_pct > 0 and np.isfinite(d_at_pct):
                    canvas.axes.axhline(y=pct, color=color, linestyle='--',
                                        linewidth=0.7, alpha=0.4)
                    lbl = item[2] if is_labeled else '_nolegend_'
                    canvas.axes.scatter([d_at_pct], [float(pct)], color=color, s=60,
                                      edgecolors='white', linewidths=0.6,
                                      alpha=0.75, zorder=5, label=lbl)

        lines, labels = canvas.axes.get_legend_handles_labels()
        if lines:
            legend = canvas.axes.legend(lines, labels, facecolor=COLOR_WHITE,
                                        edgecolor=COLOR_BORDER, labelcolor=COLOR_SECONDARY,
                                        fontsize=font_settings.xs - 2, framealpha=0.92,
                                        loc='lower right', borderpad=0.4,
                                        fancybox=True, shadow=False, ncol=2,
                                        columnspacing=0.6, handletextpad=0.5)
            legend.get_frame().set_linewidth(0.8)
        canvas.fig.tight_layout()

    def draw_fractal(self, r, canvas):
        valid = np.isfinite(r.logP) & np.isfinite(r.log1_S) & (r.intrusion_pressure_mpa > 0)
        x_all = r.logP[valid]
        y_all = r.log1_S[valid]
        if len(x_all) < 2:
            return
        x_min_raw, x_max_raw = float(np.min(x_all)), float(np.max(x_all))
        x_margin = (x_max_raw - x_min_raw) * 0.08
        x_lo, x_hi = x_min_raw - x_margin, x_max_raw + x_margin
        canvas.axes.set_xlim(x_lo, x_hi)
        canvas.axes.scatter(x_all, y_all, color=CHART_PALETTE[0], s=45,
                          edgecolors='white', linewidths=0.5,
                          alpha=0.8, zorder=3, label="data")
        num_segments = len(r.fractal_slopes) if r.fractal_slopes else 0
        dims = r.fractal_dimensions if r.fractal_dimensions else []
        seg_data = getattr(r, '_frac_seg_data', None)
        bp = seg_data['breakpoint'] if seg_data else 2.0
        if num_segments > 0 and seg_data:
            masks = [np.array(m) for m in seg_data['seg_masks']]
            for seg_idx, mask in enumerate(masks):
                if np.sum(mask) < 2:
                    continue
                xs = x_all[mask]
                ys = y_all[mask]
                slope = r.fractal_slopes[seg_idx] if seg_idx < len(r.fractal_slopes) else 0
                if abs(slope) < 1e-10:
                    continue
                intercept_val = float(np.mean(ys - slope * xs))
                x_fit = np.linspace(float(xs.min()), float(xs.max()), 80)
                y_fit = slope * x_fit + intercept_val
                d_val = dims[seg_idx] if seg_idx < len(dims) else 0
                color = CHART_PALETTE[(seg_idx + 1) % len(CHART_PALETTE)]
                canvas.axes.plot(x_fit, y_fit, color=color, linewidth=CHART_LINEWIDTH, alpha=0.9,
                               label=f"$D_{{{seg_idx+1}}}$={d_val:.3f}", zorder=4)
            canvas.axes.axvline(bp, color=CHART_PALETTE[3], linestyle='--', alpha=0.5, linewidth=1.2,
                               label=f"$\\log P$={bp:.2f}")
            y_lo, y_hi = canvas.axes.get_ylim()
            canvas._frac_bp = bp
            canvas._frac_bp_marker = canvas.axes.scatter(
                [bp], [y_hi], marker='v', s=200,
                color=CHART_PALETTE[3], edgecolors='white', linewidths=1.5,
                zorder=12, picker=10, label='_nolegend_')
            canvas._frac_bp_label = canvas.axes.annotate(
                f"$\\log P$={bp:.2f}", (bp, y_hi),
                textcoords="offset points", xytext=(0, 8),
                fontsize=font_settings.xs, color=CHART_PALETTE[3],
                ha='center', fontweight='bold', zorder=12)
        card = canvas.parent()
        if isinstance(card, ChartCard):
            card._ensure_frac_drag_events()
        lines, labels = canvas.axes.get_legend_handles_labels()
        if lines:
            legend = canvas.axes.legend(lines, labels, facecolor=COLOR_WHITE,
                                        edgecolor=COLOR_BORDER, labelcolor=COLOR_SECONDARY,
                                        fontsize=font_settings.xs - 2, framealpha=0.92,
                                        loc='upper left', borderpad=0.4,
                                        fancybox=True, shadow=False, ncol=1,
                                        columnspacing=0.6, handletextpad=0.5)
            legend.get_frame().set_linewidth(0.8)
        canvas.fig.tight_layout()

    def draw_ratio(self, r, canvas):
        if r.pore_throat_ratio_data:
            d = r.pore_throat_ratio_data
            x, y = self._safe(np.array(d['threshold']), np.array(d['pore_throat_ratio']))
            if x:
                canvas.draw_line(x, y, "$R_{pt}$", CHART_PALETTE[3], marker='p')

    def draw_swanson_matrix(self, r, canvas):
        d_um = r.pore_throat_diameter_nm * 0.001
        cum_max = np.max(r.cum_intrusion)
        sat = r.cum_intrusion / cum_max if cum_max > 0 else r.cum_intrusion
        mask = (d_um >= 10) & (d_um <= 100)
        if np.sum(mask) < 2:
            return
        d_f = d_um[mask]
        s_f = sat[mask]
        product = s_f * d_f ** 3
        x, y = self._safe(d_f, product, positive_x=True, nonneg_y=True)
        if x:
            canvas.draw_line(x, y, "$S \\times D^3$ (基质)", CHART_PALETTE[0], marker='*')
        if len(product) > 0:
            idx = int(np.argmax(product))
            px, py = float(d_f[idx]), float(product[idx])
            if np.isfinite(px) and np.isfinite(py):
                canvas.draw_scatter([px], [py], f"峰值 D={px:.3f}μm", CHART_PALETTE[1], markersize=100)

    def draw_swanson_pore(self, r, canvas):
        d_um = r.pore_throat_diameter_nm * 0.001
        cum_max = np.max(r.cum_intrusion)
        sat = r.cum_intrusion / cum_max if cum_max > 0 else r.cum_intrusion
        mask = (d_um >= 1) & (d_um < 10)
        if np.sum(mask) < 2:
            return
        d_f = d_um[mask]
        s_f = sat[mask]
        product = s_f * d_f ** 3
        x, y = self._safe(d_f, product, positive_x=True, nonneg_y=True)
        if x:
            canvas.draw_line(x, y, "$S \\times D^3$ (孔隙)", CHART_PALETTE[3], marker='*')
        if len(product) > 0:
            idx = int(np.argmax(product))
            px, py = float(d_f[idx]), float(product[idx])
            if np.isfinite(px) and np.isfinite(py):
                canvas.draw_scatter([px], [py], f"峰值 D={px:.3f}μm", CHART_PALETTE[1], markersize=100)

    # ---- 多重分形图表 ----
    def draw_mf_spectrum(self, r, canvas):
        if len(r.mf_alpha) < 2:
            return
        valid = np.isfinite(r.mf_alpha) & np.isfinite(r.mf_falpha)
        x, y = r.mf_alpha[valid], r.mf_falpha[valid]
        if len(x) < 2:
            return
        ax = canvas.axes
        ax.scatter(x, y, c=CHART_PALETTE[0], s=40, zorder=5, edgecolors='white', linewidths=0.5)
        sort_idx = np.argsort(x)
        ax.plot(x[sort_idx], y[sort_idx], '-', color=CHART_PALETTE[0], alpha=0.5, linewidth=1.5)
        x_min, x_max = np.min(x), np.max(x)
        x_mid = (x_min + x_max) / 2
        y_lo, y_hi = ax.get_ylim()
        if r.mf_D0 > 0:
            idx0 = np.argmin(np.abs(r.mf_q))
            if idx0 < len(r.mf_alpha) and np.isfinite(r.mf_alpha[idx0]):
                ax.scatter([r.mf_alpha[idx0]], [r.mf_falpha[idx0]], c=CHART_PALETTE[3], s=100,
                           zorder=6, marker='D', edgecolors='white', linewidths=1)
                ax.annotate(f"$D(0)$={r.mf_D0:.3f}", (r.mf_alpha[idx0], r.mf_falpha[idx0]),
                            textcoords="offset points", xytext=(8, -14), fontsize=font_settings.xs,
                            color=CHART_PALETTE[3], fontweight='bold')
        if r.mf_delta_alpha > 0:
            ax.axvline(x_min, color=CHART_PALETTE[1], linestyle='--', alpha=0.4, linewidth=1)
            ax.axvline(x_max, color=CHART_PALETTE[1], linestyle='--', alpha=0.4, linewidth=1)
            ax.annotate(f"$\\alpha_{{\\min}}$", (x_min, y_hi - (y_hi - y_lo) * 0.05),
                        ha='left', fontsize=font_settings.xs - 1, color=CHART_PALETTE[1], fontweight='bold')
            ax.annotate(f"$\\alpha_{{\\max}}$", (x_max, y_hi - (y_hi - y_lo) * 0.05),
                        ha='right', fontsize=font_settings.xs - 1, color=CHART_PALETTE[1], fontweight='bold')
            ax.annotate(f"$\\Delta\\alpha$={r.mf_delta_alpha:.3f}", xy=(0.98, 0.03), xycoords='axes fraction',
                        ha='right', fontsize=font_settings.xs, color=CHART_PALETTE[1], fontweight='bold')
        ax.set_xlabel("$\\alpha$", fontsize=font_settings.sm)
        ax.set_ylabel("$f(\\alpha)$", fontsize=font_settings.sm)
        ax.set_title("$f(\\alpha)$", fontsize=font_settings.md, fontweight='bold', pad=10)
        canvas.fig.tight_layout(pad=1.2)

    def draw_mf_Dq(self, r, canvas):
        if len(r.mf_q) < 2:
            return
        valid = np.isfinite(r.mf_Dq)
        x, y = r.mf_q[valid], r.mf_Dq[valid]
        if len(x) < 2:
            return
        ax = canvas.axes
        ax.scatter(x, y, c=CHART_PALETTE[1], s=40, zorder=5, edgecolors='white', linewidths=0.5)
        sort_idx = np.argsort(x)
        ax.plot(x[sort_idx], y[sort_idx], '-', color=CHART_PALETTE[1], alpha=0.5, linewidth=1.5)
        x_lo, x_hi = ax.get_xlim()
        if r.mf_D0 > 0:
            ax.axhline(r.mf_D0, color=CHART_PALETTE[3], linestyle='--', alpha=0.6, linewidth=1.5)
            ax.annotate(f"$D(0)$={r.mf_D0:.3f}", (x_lo + (x_hi - x_lo) * 0.02, r.mf_D0),
                        textcoords="data", fontsize=font_settings.xs,
                        color=CHART_PALETTE[3], fontweight='bold', va='bottom')
        if r.mf_D1 > 0:
            idx1 = np.argmin(np.abs(r.mf_q - 1.0))
            if idx1 < len(x) and np.isfinite(x[idx1]):
                ax.annotate(f"$D(1)$={r.mf_D1:.3f}", (x[idx1], y[idx1]),
                            textcoords="offset points", xytext=(6, -14), fontsize=font_settings.xs,
                            color=CHART_PALETTE[0], fontweight='bold')
        if r.mf_D2 > 0:
            idx2 = np.argmin(np.abs(r.mf_q - 2.0))
            if idx2 < len(x) and np.isfinite(x[idx2]):
                ax.annotate(f"$D(2)$={r.mf_D2:.3f}", (x[idx2], y[idx2]),
                            textcoords="offset points", xytext=(6, 6), fontsize=font_settings.xs,
                            color=CHART_PALETTE[2], fontweight='bold')
        ax.set_xlabel("$q$", fontsize=font_settings.sm)
        ax.set_ylabel("$D(q)$", fontsize=font_settings.sm)
        ax.set_title("$D(q)$", fontsize=font_settings.md, fontweight='bold', pad=10)
        canvas.fig.tight_layout(pad=1.2)

    def draw_mf_tau(self, r, canvas):
        if len(r.mf_q) < 2:
            return
        valid = np.isfinite(r.mf_tau_q)
        x, y = r.mf_q[valid], r.mf_tau_q[valid]
        if len(x) < 2:
            return
        ax = canvas.axes
        ax.scatter(x, y, c=CHART_PALETTE[4], s=40, zorder=5, edgecolors='white', linewidths=0.5)
        sort_idx = np.argsort(x)
        ax.plot(x[sort_idx], y[sort_idx], '-', color=CHART_PALETTE[4], alpha=0.5, linewidth=1.5)
        ax.set_xlabel("$q$", fontsize=font_settings.sm)
        ax.set_ylabel("$\\tau(q)$", fontsize=font_settings.sm)
        ax.set_title("$\\tau(q)$", fontsize=font_settings.md, fontweight='bold', pad=10)
        canvas.fig.tight_layout(pad=1.2)

    def _table_style(self):
        return f"""
            QTableWidget {{
                background-color: {COLOR_WHITE}; alternate-background-color: {COLOR_ALT_ROW};
                color: {COLOR_TEXT_SECONDARY}; font-size: {font_settings.xs}px;
                border: 1px solid {COLOR_BORDER}; border-radius: {RADIUS_SM}px;
                gridline-color: {COLOR_BORDER};
            }}
            QTableWidget::item {{ padding: {SPACING_XS}px {SPACING_SM}px; }}
            QTableWidget::item:selected {{ background-color: {COLOR_PRIMARY_LIGHT}; color: {COLOR_PRIMARY}; }}
            QHeaderView::section {{
                background-color: {COLOR_PRIMARY_LIGHT}; color: {COLOR_PRIMARY};
                font-weight: 600; font-size: {font_settings.xs}px; border: none;
                border-bottom: 1px solid {COLOR_BORDER}; border-right: 1px solid {COLOR_BORDER};
                padding: {SPACING_SM}px {SPACING_MD}px;
            }}
            QHeaderView::section:last {{ border-right: none; }}
        """

    def _on_header_clicked_select_all(self, logical_index):
        self.data_table.selectAll()
        self.data_table.setFocus()

    def _header_context_menu(self, pos):
        logical_index = self.data_table.horizontalHeader().logicalIndexAt(pos)
        if logical_index < 0:
            return
        menu = QMenu(self.data_table)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLOR_WHITE}; border: 1px solid {COLOR_BORDER};
                padding: {SPACING_XS}px;
            }}
            QMenu::item {{
                padding: {SPACING_SM}px {SPACING_LG}px;
                font-size: {font_settings.sm}px; color: {COLOR_TEXT};
            }}
            QMenu::item:selected {{
                background-color: {COLOR_PRIMARY_LIGHT}; color: {COLOR_PRIMARY};
            }}
        """)
        h_item = self.data_table.horizontalHeaderItem(logical_index)
        col_name = h_item.text().replace("\n", " ") if h_item else f"第{logical_index+1}列"
        act_copy_col = menu.addAction(f"复制「{col_name}」列")
        act_copy_all = menu.addAction("复制全部数据")
        action = menu.exec_(self.data_table.horizontalHeader().mapToGlobal(pos))
        if action == act_copy_col:
            self._copy_column(logical_index)
        elif action == act_copy_all:
            self._copy_all_table_data()

    def _copy_column(self, col):
        row_count = self.data_table.rowCount()
        if row_count == 0:
            return
        h_item = self.data_table.horizontalHeaderItem(col)
        header = h_item.text().replace("\n", " ") if h_item else ""
        lines = [header]
        for r in range(row_count):
            item = self.data_table.item(r, col)
            lines.append(item.text() if item else "")
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"已复制「{header}」列 {row_count} 行数据到剪贴板", 3000)

    def _copy_selected_table_data(self):
        selected = self.data_table.selectedItems()
        if not selected:
            return
        rows = set()
        for item in selected:
            rows.add(item.row())
        if not rows:
            return
        sorted_rows = sorted(rows)
        col_count = self.data_table.columnCount()
        headers = []
        for c in range(col_count):
            h = self.data_table.horizontalHeaderItem(c)
            headers.append(h.text().replace("\n", " ") if h else "")
        lines = ["\t".join(headers)]
        for r in sorted_rows:
            row_vals = []
            for c in range(col_count):
                item = self.data_table.item(r, c)
                row_vals.append(item.text() if item else "")
            lines.append("\t".join(row_vals))
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self.statusBar().showMessage(f"已复制 {len(sorted_rows)} 行数据到剪贴板，可直接粘贴到 Excel", 3000)

    def _copy_all_table_data(self):
        row_count = self.data_table.rowCount()
        col_count = self.data_table.columnCount()
        if row_count == 0 or col_count == 0:
            return
        headers = []
        for c in range(col_count):
            h = self.data_table.horizontalHeaderItem(c)
            headers.append(h.text().replace("\n", " ") if h else "")
        lines = ["\t".join(headers)]
        for r in range(row_count):
            row_vals = []
            for c in range(col_count):
                item = self.data_table.item(r, c)
                row_vals.append(item.text() if item else "")
            lines.append("\t".join(row_vals))
        text = "\n".join(lines)
        QApplication.clipboard().setText(text)
        self.data_table.selectAll()
        self.statusBar().showMessage(f"已复制全部 {row_count} 行数据到剪贴板，可直接粘贴到 Excel", 3000)

    def _fill_table(self, headers, rows):
        self.data_table.clear()
        self.data_table.setColumnCount(len(headers))
        self.data_table.setHorizontalHeaderLabels(headers)
        self.data_table.setRowCount(len(rows))
        formatted_rows = []
        for row_data in rows:
            formatted_row = []
            for val in row_data:
                try:
                    float_val = float(val)
                    formatted_val = smart_format_number(float_val)
                except (ValueError, TypeError):
                    formatted_val = str(val)
                formatted_row.append(formatted_val)
            formatted_rows.append(formatted_row)
        for i, row_data in enumerate(formatted_rows):
            for j, val in enumerate(row_data):
                item = QTableWidgetItem(str(val))
                item.setTextAlignment(Qt.AlignCenter)
                self.data_table.setItem(i, j, item)
        self.data_table.resizeRowsToContents()

    def update_data_table(self):
        if not self.processor or not self.result:
            self._fill_table([], [])
            return
        tid = self.current_chart
        r = self.result
        d = self.processor.data
        psia_to_mpa = 1.0 / 145.036
        self.chart_data_label.setText(f"当前图表: {dict(CHARTS).get(tid, tid)}")
        self.chart_data_label.setStyleSheet(f"color: {COLOR_PRIMARY}; font-size: {font_settings.sm}px; font-weight: 600; padding: 0 2px;")

        if tid == "inj_ext":
            headers = ["阶段", "压力\n(psia)", "压力\n(MPa)", "累积体积\n(mL/g)"]
            rows = []
            for i in range(len(d.intrusion_pressure_psia)):
                p_psia = float(d.intrusion_pressure_psia[i])
                p_mpa = p_psia * psia_to_mpa
                cv = float(d.intrusion_cumulative_mlg[i]) if i < len(d.intrusion_cumulative_mlg) else 0.0
                rows.append(["进汞", p_psia, p_mpa, cv])
            if len(d.extrusion_pressure_psia) > 0:
                for i in range(len(d.extrusion_pressure_psia)):
                    p_psia = float(d.extrusion_pressure_psia[i])
                    p_mpa = p_psia * psia_to_mpa
                    cv = float(d.extrusion_cumulative_mlg[i]) if i < len(d.extrusion_cumulative_mlg) else 0.0
                    rows.append(["退汞", p_psia, p_mpa, cv])
            self._fill_table(headers, rows)
        elif tid == "interpolated":
            headers = ["阶段", "压力\n(MPa)", "累积体积\n(mL/g)"]
            rows = []
            for i in range(len(r.intrusion_pressure_mpa)):
                rows.append(["插值进汞", float(r.intrusion_pressure_mpa[i]), float(r.cum_intrusion[i])])
            if len(r.extrusion_pressure_mpa) > 0:
                for i in range(len(r.extrusion_pressure_mpa)):
                    rows.append(["插值退汞", float(r.extrusion_pressure_mpa[i]), float(r.cum_extrusion[i])])
            self._fill_table(headers, rows)
        elif tid == "capillary":
            cum_max = float(np.max(r.cum_intrusion)) if len(r.cum_intrusion) > 0 else 1.0
            sat_scale = self.get_display_sat_scale()
            sat = r.cum_intrusion / cum_max * 100.0 * sat_scale if cum_max > 0 else r.cum_intrusion
            headers = ["汞饱和度\n(%)", "压力\n(MPa)"]
            rows = [[float(sat[i]), float(r.intrusion_pressure_mpa[i])] for i in range(len(r.intrusion_pressure_mpa))]
            self._fill_table(headers, rows)
        elif tid == "dvdlogD":
            headers = ["孔径直径\n(nm)", "dV/dlogD\n(mL/g)"]
            rows = [[float(r.pore_throat_diameter_nm[i]), float(r.dv_dlogD[i])] for i in range(len(r.pore_throat_diameter_nm))]
            self._fill_table(headers, rows)
        elif tid == "pct":
            headers = ["孔径\n(nm)", "进汞量\n(%)"]
            rows = [[float(r.pore_throat_bins[i]), float(r.bin_pct[i])] for i in range(len(r.pore_throat_bins))]
            self._fill_table(headers, rows)
        elif tid == "characteristic":
            cum_max = float(np.max(r.cum_intrusion)) if len(r.cum_intrusion) > 0 else 1.0
            sat = r.cum_intrusion / cum_max * 100.0 if cum_max > 0 else r.cum_intrusion
            headers = ["孔径直径\n(μm)", "累积饱和度\n(%)"]
            rows = [[float(r.pore_throat_diameter_um[i]), float(sat[i])] for i in range(len(r.pore_throat_diameter_um))]
            self._fill_table(headers, rows)
        elif tid == "fractal":
            headers = ["log(P)", "log(1-S)"]
            valid = np.isfinite(r.logP) & np.isfinite(r.log1_S) & (r.intrusion_pressure_mpa > 0)
            rows = [[float(r.logP[i]), float(r.log1_S[i])] for i in range(len(r.logP)) if valid[i]]
            self._fill_table(headers, rows)
        elif tid == "ratio":
            if r.pore_throat_ratio_data:
                d_data = r.pore_throat_ratio_data
                thresholds = np.array(d_data['threshold'])
                ratios = np.array(d_data['pore_throat_ratio'])
                headers = ["汞饱和度\n(%)", "孔喉比"]
                rows = [[float(thresholds[i]), float(ratios[i])] for i in range(len(thresholds))]
                self._fill_table(headers, rows)
            else:
                self._fill_table([], [])
        elif tid == "swanson_matrix":
            cum_max = float(np.max(r.cum_intrusion)) if len(r.cum_intrusion) > 0 else 1.0
            sat = r.cum_intrusion / cum_max if cum_max > 0 else r.cum_intrusion
            d_um = r.pore_throat_diameter_nm * 0.001
            mask = (d_um >= 10) & (d_um <= 100)
            d_f = d_um[mask]
            s_f = sat[mask]
            product = s_f * d_f ** 3
            headers = ["孔径直径\n(μm)", "S×D³"]
            rows = [[float(d_f[i]), float(product[i])] for i in range(len(d_f)) if np.isfinite(d_f[i]) and np.isfinite(product[i]) and d_f[i] > 0]
            self._fill_table(headers, rows)
        elif tid == "swanson_pore":
            cum_max = float(np.max(r.cum_intrusion)) if len(r.cum_intrusion) > 0 else 1.0
            sat = r.cum_intrusion / cum_max if cum_max > 0 else r.cum_intrusion
            d_um = r.pore_throat_diameter_nm * 0.001
            mask = (d_um >= 1) & (d_um < 10)
            d_f = d_um[mask]
            s_f = sat[mask]
            product = s_f * d_f ** 3
            headers = ["孔径直径\n(μm)", "S×D³"]
            rows = [[float(d_f[i]), float(product[i])] for i in range(len(d_f)) if np.isfinite(d_f[i]) and np.isfinite(product[i]) and d_f[i] > 0]
            self._fill_table(headers, rows)
        else:
            self._fill_table([], [])

    def apply_params(self):
        if not self.processor:
            return
        try:
            self.processor.params.contact_angle = float(self.left_panel.get_param("contact_angle"))
            self.processor.params.surface_tension = float(self.left_panel.get_param("surface_tension"))
            self.processor.params.smoothing_window = int(self.left_panel.get_param("smoothing"))
            self.processor.params.enforce_monotonic = self.left_panel.get_enforce_monotonic()
            self.processor.params.use_fixed_interpolation = self.left_panel.get_use_fixed_interpolation()
            self.processor.params.interp_p_min = self.left_panel.get_interp_p_min()
            self.processor.params.interp_p_max = self.left_panel.get_interp_p_max()
            cap_card = self.chart_cards.get("capillary")
            if cap_card:
                self.processor.params.disp_p_sat_min = cap_card.get_disp_p_sat_min()
                self.processor.params.disp_p_sat_max = cap_card.get_disp_p_sat_max()
            frac_card = self.chart_cards.get("fractal")
            if frac_card:
                self.processor.params.frac_seg_breakpoint = frac_card.get_frac_seg_breakpoint()
            he_por = self.left_panel.get_he_porosity()
            self.processor.params.he_porosity_override = he_por if he_por else 0.0
        except Exception:
            pass

    def apply_params_to(self, proc):
        try:
            proc.params.contact_angle = float(self.left_panel.get_param("contact_angle"))
            proc.params.surface_tension = float(self.left_panel.get_param("surface_tension"))
            proc.params.smoothing_window = int(self.left_panel.get_param("smoothing"))
            proc.params.enforce_monotonic = self.left_panel.get_enforce_monotonic()
            proc.params.use_fixed_interpolation = self.left_panel.get_use_fixed_interpolation()
            proc.params.interp_p_min = self.left_panel.get_interp_p_min()
            proc.params.interp_p_max = self.left_panel.get_interp_p_max()
            cap_card = self.chart_cards.get("capillary")
            if cap_card:
                proc.params.disp_p_sat_min = cap_card.get_disp_p_sat_min()
                proc.params.disp_p_sat_max = cap_card.get_disp_p_sat_max()
            frac_card = self.chart_cards.get("fractal")
            if frac_card:
                proc.params.frac_seg_breakpoint = frac_card.get_frac_seg_breakpoint()
            he_por = self.left_panel.get_he_porosity()
            proc.params.he_porosity_override = he_por if he_por else 0.0
        except Exception:
            pass

    def show_open_file(self):
        fp, _ = QFileDialog.getOpenFileName(self, "打开压汞数据文件", DATA_DIR,
                                            "Excel文件 (*.xlsx *.xls *.xlsm);;所有文件 (*.*)")
        if fp:
            self.on_file(fp)

    def show_batch(self):
        folder = QFileDialog.getExistingDirectory(self, "选择文件夹", DATA_DIR)
        if folder:
            self.on_batch(folder)

    def on_file(self, fp):
        try:
            self.processor = MICPProcessor()
            self.processor.load(fp)
            self.apply_params()
            self.result = self.processor.process()
            self.left_panel.set_file_label(os.path.basename(fp))
            self.left_panel.update_sample_info(self.processor, self.result)
            self.current_chart = CHARTS[0][0]
            for cid, card in self.chart_cards.items():
                card.set_selected(cid == self.current_chart)
            self.update_all_charts()
        except LoadError as e:
            self.error(str(e))
        except Exception as e:
            self.error(str(e))

    def on_batch(self, folder):
        exts = ('.xlsx', '.XLS', '.xls', '.XLSX')
        try:
            files = [f for f in os.listdir(folder) if f.endswith(exts) and not f.startswith('~$')
                     and "template" not in f.lower() and not f.endswith('.xlsm')]
        except Exception as e:
            self.error(str(e))
            return
        if not files:
            return
        self.batch_results = []
        load_errors = []
        ok = 0
        for fname in files:
            try:
                p = MICPProcessor()
                p.load(os.path.join(folder, fname))
                self.apply_params_to(p)
                r = p.process()
                self.batch_results.append({'name': fname, 'result': r, 'data': p.data, 'processor': p})
                ok += 1
            except LoadError as e:
                load_errors.append(f"{fname}: {str(e)}")
                self.batch_results.append({'name': fname, 'error': str(e)})
            except Exception as e:
                self.batch_results.append({'name': fname, 'error': str(e)})
        if load_errors:
            error_msg = "以下文件加载数据列不匹配：\n\n" + "\n".join(load_errors)
            error_msg += "\n\n请确保原始数据包含 Pressure (psia) 和 Cumulative Pore Volume (mL/g) 两列。"
            QMessageBox.warning(self, "数据列缺失", error_msg)
        self.left_panel.set_file_label(f"批量: {ok}/{len(files)}")
        if ok > 0:
            self.processor = self.batch_results[0]['processor']
            self.result = self.batch_results[0]['result']
            self.left_panel.update_sample_info(self.processor, self.result)
            self.current_chart = CHARTS[0][0]
            for cid, card in self.chart_cards.items():
                card.set_selected(cid == self.current_chart)
            self.update_all_charts()

    def recalc(self):
        if not self.processor:
            return
        try:
            self.apply_params()
            self.result = self.processor.process()
            self.update_all_charts()
            AutoCloseDialog.show_message(self, "重新计算已完成，所有图表和数据已更新。", timeout=3000)
        except Exception as e:
            self.error(str(e))

    def error(self, msg):
        QMessageBox.critical(self, "错误", str(msg))

    def show_settings(self):
        dialog = SettingsDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            scale = dialog.get_font_scale()
            font_settings.set_scale(scale)
            self.setStyleSheet(self._main_style())
            self.left_panel.refresh_fonts()
            self._update_left_panel_width()
            self.update_all_charts()


class AutoCloseDialog(QDialog):
    def __init__(self, parent=None, message="", title="提示", timeout=3000):
        super().__init__(parent)
        self._timeout = timeout
        self._remaining = timeout
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)
        self._tick_interval = 200
        self.setWindowTitle(title)
        self.setMinimumSize(380, 150)
        self.setMaximumSize(500, 200)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)
        self.setStyleSheet(f"""
            QDialog {{ background-color: {COLOR_BG}; }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(14)
        self._label = QLabel(message)
        self._label.setWordWrap(True)
        self._label.setStyleSheet(f"color: {COLOR_TEXT}; font-size: {font_settings.sm}px;")
        layout.addWidget(self._label)
        btn_layout = QHBoxLayout()
        btn_layout.addStretch(1)
        self._btn = QPushButton("确定")
        self._btn.setMinimumHeight(CONTROL_HEIGHT_SM)
        self._btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_PRIMARY}; color: white;
                font-size: {font_settings.sm}px; font-weight: 500;
                border: none; border-radius: {RADIUS_SM}px;
                padding: 0 {SPACING_LG}px; min-height: {CONTROL_HEIGHT_SM}px;
            }}
            QPushButton:hover {{ background-color: {COLOR_PRIMARY_DARK}; }}
        """)
        self._btn.clicked.connect(self.accept)
        btn_layout.addWidget(self._btn)
        layout.addLayout(btn_layout)
        self.setMouseTracking(True)
        self._timer.start(self._tick_interval)

    def _on_tick(self):
        self._remaining -= self._tick_interval
        if self._remaining <= 0:
            self._timer.stop()
            self.accept()
        else:
            s = self._remaining / 1000
            self._btn.setText(f"确定 ({s:.0f}s)")

    def enterEvent(self, event):
        self._timer.stop()
        self._btn.setText("确定")
        super().enterEvent(event)

    def leaveEvent(self, event):
        if self._remaining > 0:
            self._timer.start(self._tick_interval)
        super().leaveEvent(event)

    @staticmethod
    def show_message(parent, message, title="提示", timeout=3000):
        dlg = AutoCloseDialog(parent, message, title, timeout)
        dlg.exec_()


def main():
    QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
    QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    app.setFont(QFont("Microsoft YaHei", font_settings.base))
    window = MainWindow()
    window.showMaximized()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()