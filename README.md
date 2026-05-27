# MICP 压汞法数据处理系统

基于 PyQt5 的压汞法（Mercury Injection Capillary Pressure）孔隙结构分析系统，支持从原始 Excel 数据导入到完整报告导出的全流程自动化处理。

## 功能概览

| 分析模块 | 说明 |
|---|---|
| 进退汞曲线 | 原始 / 插值后进退汞曲线对比，支持批量多样品叠加 |
| 毛细管压力曲线 | Washburn 方程孔径转换，排驱压力 $P_d$ 线性拟合（可拖动标记交互调整拟合范围） |
| 孔径分布 dV/dlogD | 连续孔隙体积分布 |
| 孔径分布 (%) | 分箱百分比分布 |
| 特征参数分布 | $S_{cum}$-d 曲线上的 $\Psi$ 分位数分析（$S_p$, $S_{kp}$, $K_p$, $D_M$） |
| **单分形维数（分段）** | $\log P$–$\log(1-S)$ 分段线性拟合，支持交互式拖动断点 |
| **多重分形谱** $f(\alpha)$ | Chhabra-Jensen 直接法 + box_size→0 外推法 |
| **广义维数** $D(q)$ | $D(0)$, $D(1)$, $D(2)$, $D(-10)$, $D(10)$, Hurst 指数 $H$ |
| **质量标度指数** $\tau(q)$ | — |
| 孔喉比 | 孔喉比曲线 |
| Swanson 渗透率 | 基质渗透率 $k_{413}$、孔隙渗透率 $k_{10}$ |
| 报告导出 | Excel / JSON |

## 安装

```bash
git clone https://github.com/CollapsibleLoess/MICP.git
cd MICP
pip install -r requirements.txt
```

## 运行

```bash
cd micp_gui
python run_pyqt.py
```

## 使用流程

1. **文件 → 打开文件**：选择原始压汞 Excel 数据（需包含 Pressure (psia) 和 Cumulative Pore Volume (mL/g) 列）
2. 左侧面板可调整接触角、表面张力、平滑窗口、氦气孔隙度等参数
3. 参数修改后点击 **重新计算** 刷新所有图表
4. 右侧滚动浏览所有分析图表
5. 点击图表卡片选中后，下方数据表同步显示对应数据
6. **文件 → 导出报告** 可导出 Excel 或 JSON

### 交互式操作

- **排驱压力拟合范围**：拖动毛细管压力曲线上方的 ▼ 标记调整拟合上下限
- **分形分段断点**：拖动分形维数图上的 ▼ 标记调整 $\log P$ 断点位置
- 右侧数据表支持点击表头全选复制

## 依赖

- Python ≥ 3.8
- numpy, scipy, matplotlib
- PyQt5
- pandas, openpyxl

## 项目结构

```
MICP/
├── micp_gui/
│   ├── core/               # 计算核心模块
│   │   ├── processor.py    # 主处理管线
│   │   ├── loader.py       # Excel 数据加载
│   │   ├── models.py       # 数据模型 (MICPData, MICPParams, MICPResult)
│   │   ├── density.py      # 密度/孔隙度计算
│   │   ├── pore_throat.py  # 孔喉直径 (Washburn 方程)
│   │   ├── psd.py          # 孔径分布
│   │   ├── displacement.py # 排驱压力拟合
│   │   ├── fractal.py      # 单分形维数
│   │   ├── multifractal.py # 多重分形分析
│   │   ├── characteristic.py # 特征参数分布
│   │   ├── permeability.py # Swanson 渗透率
│   │   ├── surface_area.py # 比表面积
│   │   ├── interpolate.py  # 固定压力区间插值
│   │   ├── correction.py   # 数据校正
│   │   └── exporter.py     # Excel/JSON 报告导出
│   ├── data/               # 示例数据文件
│   ├── pyqt_app.py         # PyQt5 GUI 主程序
│   ├── run_pyqt.py         # 启动入口
│   └── app.py              # DearPyGui 备选界面
├── reference/              # 多重分形原始参考代码
├── requirements.txt
└── README.md
```

## 参考文献

1. Chhabra A, Jensen R. Direct determination of the $f(\alpha)$ singularity spectrum. *Physical Review Letters*, 1989.
2. 冯光俊等. 基于多重分形理论的低阶煤孔隙结构非均质性及影响因素研究. *现代地质*, 2025.
