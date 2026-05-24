import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os

plt.rcParams['font.sans-serif'] = ['Times New Roman']  # 这就是我的系统内的字体,不要修改
plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号
plt.rcParams['mathtext.fontset'] = 'stix'  # 数学表达式使用兼容的字体

# 学术配色方案（16色，色盲友好）
ACADEMIC_COLORS = [
    '#0173B2',  # Science Blue
    '#DE8F05',  # Science Orange
    '#029E73',  # Science Green
    '#CC78BC',  # Science Purple
    '#CA9161',  # Science Brown
    '#FBAFE4',  # Science Pink
    '#949494',  # Science Grey
    '#ECE133',  # Science Olive
    '#56B4E9',  # Sky Blue
    '#B15928',  # Vermillion
    '#D55E00',  # Dark Orange
    '#0072B2',  # Darker Blue
    '#009E73',  # Darker Green
    '#CC79A7',  # Lighter Purple
    '#999933',  # Dark Olive
    '#88C999',  # Mint Green
]


def draw_mf_results(mf_results_list, title_prefix):
    """
    绘制多重分形分析结果并保存到report文件夹
    拆分为两张图:
    1. 四张子图: 配分函数标度关系-质量指数-广义维数-奇异强度
    2. 单独的多重分形谱f(α)
    """
    # 创建report目录
    report_dir = "report"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)

    # 提取所有样本的结果
    all_q_values = mf_results_list[0]['results']['q_values']
    all_Dq = [result['results']['generalized_dimensions_Dq'] for result in mf_results_list]
    all_tq = [result['results']['mass_exponents_tq'] for result in mf_results_list]
    all_aq = [result['results']['singularity_strengths_aq'] for result in mf_results_list]
    all_fq = [result['results']['singularity_spectrums_fq'] for result in mf_results_list]
    all_moment_curves = [result['results']['moment_boxsize_curves_Xq'] for result in mf_results_list]

    # 使用学术配色方案
    num_samples = len(mf_results_list)
    colors = [ACADEMIC_COLORS[i % len(ACADEMIC_COLORS)] for i in range(num_samples)]

    # 字体缩放因子
    scale_factor = 1.2
    base_fontsize = 16 * scale_factor
    label_fontsize = 18 * scale_factor
    title_fontsize = 20 * scale_factor
    legend_fontsize = 10 * scale_factor

    # ========== 图1: 四张子图 ==========
    fig1, axs1 = plt.subplots(2, 2, figsize=(16, 12))

    # 准备图例标签和句柄
    legend_labels = [f'Z {i + 1}' for i in range(num_samples)]
    legend_handles = []
    for i in range(num_samples):
        handle = plt.Line2D([0], [0], color=colors[i], marker='o', linestyle='-',
                            markersize=8, linewidth=2.5, alpha=0.8)
        legend_handles.append(handle)

    # 1. 配分函数与标度关系图 (左上) - 只显示第一个样本的6组q值，使用拟合直线

    # 为不同的q值定义不同的线条样式和标记
    line_styles = ['-', '--', ':', '-.', (0, (3, 1, 1, 1)), (0, (5, 2))]  # 6种线型
    markers = ['o', 's', '^', 'D', 'v', 'p']  # 6种标记

    # 选择第一个样本
    sample_idx = 0
    moment_curves = all_moment_curves[sample_idx]
    # 选择10个有代表性的q值的索引来绘图
    num_q_to_plot = 9
    total_q = len(moment_curves)
    q_indices_to_plot = [int(i * (total_q - 1) / (num_q_to_plot - 1)) for i in range(num_q_to_plot)]

    # 为每个q值绘制散点和拟合直线
    for style_idx, q_idx in enumerate(q_indices_to_plot):
        if q_idx < len(moment_curves):
            curve_data = moment_curves[q_idx]
            q_val = all_q_values[q_idx]

            log_box_size = np.array(curve_data['log_box_size'])
            log_moment = np.array(curve_data['log_moment'])

            # 线性拟合
            coefficients = np.polyfit(log_box_size, log_moment, 1)
            poly = np.poly1d(coefficients)
            fitted_line = poly(log_box_size)

            # 获取颜色（使用学术配色，循环使用）
            color = ACADEMIC_COLORS[style_idx % len(ACADEMIC_COLORS)]

            # 绘制散点
            axs1[0, 0].scatter(log_box_size, log_moment,
                               color=color,
                               marker=markers[style_idx % len(markers)],  # 循环使用标记
                               s=60,
                               alpha=0.6,
                               edgecolors='black',
                               linewidths=0.5,
                               label=f'q={q_val:.1f}')

            # 绘制拟合直线
            axs1[0, 0].plot(log_box_size, fitted_line,
                            color=color,
                            linestyle=line_styles[style_idx % len(line_styles)],  # 循环使用线型
                            linewidth=2.5,
                            alpha=0.9)

    axs1[0, 0].set_xlabel(r'log($\varepsilon$)', fontsize=label_fontsize, fontweight='bold')
    axs1[0, 0].set_ylabel(r'log($Z_q$)', fontsize=label_fontsize, fontweight='bold')
    axs1[0, 0].set_title('Partition Function Scaling (Z 1)', fontsize=title_fontsize, fontweight='bold', pad=15)
    axs1[0, 0].grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
    axs1[0, 0].tick_params(labelsize=base_fontsize)

    # 图例显示q值，调整为3列以适应10个条目
    axs1[0, 0].legend(loc='best', fontsize=legend_fontsize,
                      frameon=True, fancybox=True, shadow=True,
                      framealpha=0.95, ncol=3)  # 改为3列
    # 2. τ(q) vs q (右上)
    for i, tq in enumerate(all_tq):
        axs1[0, 1].plot(all_q_values, tq, 'o-', color=colors[i], alpha=0.8, markersize=5, linewidth=2.5)
    axs1[0, 1].set_xlabel(r'$q$', fontsize=label_fontsize, fontweight='bold')
    axs1[0, 1].set_ylabel(r'$\tau(q)$', fontsize=label_fontsize, fontweight='bold')
    axs1[0, 1].set_title(r'Mass Exponent $\tau(q)$', fontsize=title_fontsize, fontweight='bold', pad=15)
    axs1[0, 1].grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
    axs1[0, 1].tick_params(labelsize=base_fontsize)
    axs1[0, 1].legend(legend_handles, legend_labels, loc='best', fontsize=legend_fontsize,
                      frameon=True, fancybox=True, shadow=True, framealpha=0.95)

    # 3. D(q) vs q (左下)
    for i, Dq in enumerate(all_Dq):
        axs1[1, 0].plot(all_q_values, Dq, 'o-', color=colors[i], alpha=0.8, markersize=5,
                        linewidth=2.5, label=f'Z {i + 1}')
    axs1[1, 0].set_xlabel(r'$q$', fontsize=label_fontsize, fontweight='bold')
    axs1[1, 0].set_ylabel(r'$D(q)$', fontsize=label_fontsize, fontweight='bold')
    axs1[1, 0].set_title(r'Generalized Dimension $D(q)$', fontsize=title_fontsize, fontweight='bold', pad=15)
    axs1[1, 0].grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
    axs1[1, 0].tick_params(labelsize=base_fontsize)
    axs1[1, 0].legend(legend_handles, legend_labels, loc='best', fontsize=legend_fontsize,
                      frameon=True, fancybox=True, shadow=True, framealpha=0.95)

    # 4. α(q) vs q (右下)
    for i, aq in enumerate(all_aq):
        axs1[1, 1].plot(all_q_values, aq, 'o-', color=colors[i], alpha=0.8, markersize=5, linewidth=2.5)
    axs1[1, 1].set_xlabel(r'$q$', fontsize=label_fontsize, fontweight='bold')
    axs1[1, 1].set_ylabel(r'$\alpha(q)$', fontsize=label_fontsize, fontweight='bold')
    axs1[1, 1].set_title(r'Singularity Strength $\alpha(q)$', fontsize=title_fontsize, fontweight='bold', pad=15)
    axs1[1, 1].grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
    axs1[1, 1].tick_params(labelsize=base_fontsize)
    axs1[1, 1].legend(legend_handles, legend_labels, loc='best', fontsize=legend_fontsize,
                      frameon=True, fancybox=True, shadow=True, framealpha=0.95)

    # 调整子图间距
    plt.tight_layout(pad=2.5, w_pad=3.0, h_pad=3.5)

    # 保存第一张图为SVG格式
    fig1_filename_svg = os.path.join(report_dir, f'{title_prefix}_Multifractal_Analysis.svg')
    plt.savefig(fig1_filename_svg, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    # ========== 图2: 多重分形谱 f(α) ==========
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    for i, (aq, fq) in enumerate(zip(all_aq, all_fq)):
        ax2.plot(aq, fq, 'o-', color=colors[i], alpha=0.8, markersize=6,
                 linewidth=2.5, label=f'Z {i + 1}')

    ax2.set_xlabel(r'$\alpha$', fontsize=label_fontsize, fontweight='bold')
    ax2.set_ylabel(r'$f(\alpha)$', fontsize=label_fontsize, fontweight='bold')
    ax2.set_title(r'Multifractal Spectrum $f(\alpha)$', fontsize=title_fontsize, fontweight='bold', pad=15)
    ax2.grid(True, linestyle='--', alpha=0.7, linewidth=1.2)
    ax2.tick_params(labelsize=base_fontsize)
    ax2.legend(legend_handles, legend_labels, loc='best', fontsize=legend_fontsize,
               frameon=True, fancybox=True, shadow=True, framealpha=0.95)

    # 添加边框美化
    for spine in ax2.spines.values():
        spine.set_linewidth(1.5)

    plt.tight_layout(pad=2.0)

    # 保存第二张图为SVG格式
    fig2_filename_svg = os.path.join(report_dir, f'{title_prefix}_Multifractal_Spectrum.svg')
    plt.savefig(fig2_filename_svg, format='svg', bbox_inches='tight', dpi=300)
    plt.close()

    # 导出参数和绘图数据
    export_mf_results(mf_results_list, title_prefix, all_q_values, all_Dq, all_tq, all_aq, all_fq, all_moment_curves)


def export_mf_results(mf_results_list, title_prefix, all_q_values, all_Dq, all_tq, all_aq, all_fq, all_moment_curves):
    """
    导出统计参数和绘图数据
    """
    report_dir = "report"

    # 1. 导出统计参数到CSV文件
    all_params = []
    for i, result_data in enumerate(mf_results_list):
        params = result_data['params'].copy()
        params['Sample_ID'] = i + 1
        all_params.append(params)

    params_df = pd.DataFrame(all_params)
    columns = ['Sample_ID'] + [col for col in params_df.columns if col != 'Sample_ID']
    params_df = params_df[columns]

    # 根据title_prefix确定统计参数文件名
    if "粒径" in title_prefix or "Particle" in title_prefix:
        params_filename = os.path.join(report_dir, "Particle_Size_Statistical_Parameters.csv")
    else:
        params_filename = os.path.join(report_dir, "Pore_Size_Statistical_Parameters.csv")

    params_df.to_csv(params_filename, index=False, encoding='utf-8-sig')

    # 2. 导出绘图数据到Excel文件
    if "粒径" in title_prefix or "Particle" in title_prefix:
        excel_filename = os.path.join(report_dir, "Particle_Size_Plot_Data.xlsx")
    else:
        excel_filename = os.path.join(report_dir, "Pore_Size_Plot_Data.xlsx")

    with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
        # Sheet1: D(q) vs q 数据
        dq_data = {'q': all_q_values}
        for i, Dq in enumerate(all_Dq):
            dq_data[f'Sample_{i + 1}_D(q)'] = Dq
        pd.DataFrame(dq_data).to_excel(writer, sheet_name='Generalized_Dimension', index=False)

        # Sheet2: τ(q) vs q 数据
        tq_data = {'q': all_q_values}
        for i, tq in enumerate(all_tq):
            tq_data[f'Sample_{i + 1}_τ(q)'] = tq
        pd.DataFrame(tq_data).to_excel(writer, sheet_name='Mass_Exponent', index=False)

        # Sheet3: α(q) vs q 数据
        aq_data = {'q': all_q_values}
        for i, aq in enumerate(all_aq):
            aq_data[f'Sample_{i + 1}_α(q)'] = aq
        pd.DataFrame(aq_data).to_excel(writer, sheet_name='Singularity_Strength', index=False)

        # Sheet4: f(α) vs α 数据
        fa_data = {}
        max_len = max(len(aq) for aq in all_aq)
        for i, (aq, fq) in enumerate(zip(all_aq, all_fq)):
            # 补齐长度到最大长度
            aq_padded = list(aq) + [np.nan] * (max_len - len(aq))
            fq_padded = list(fq) + [np.nan] * (max_len - len(fq))
            fa_data[f'Z_{i + 1}_α'] = aq_padded
            fa_data[f'Z_{i + 1}_f(α)'] = fq_padded
        pd.DataFrame(fa_data).to_excel(writer, sheet_name='Multifractal_Spectrum', index=False)

        # Sheet5: 配分函数数据（包含拟合参数）
        partition_data = {}

        # 选择第一个样本的6个q值
        sample_idx = 0
        moment_curves = all_moment_curves[sample_idx]
        num_q_to_plot = 6
        total_q = len(moment_curves)
        q_indices = [int(i * (total_q - 1) / (num_q_to_plot - 1)) for i in range(num_q_to_plot)]

        for j in q_indices:
            if j < len(moment_curves):
                curve_data = moment_curves[j]
                q_val = all_q_values[j]

                log_box_size = curve_data['log_box_size']
                log_moment = curve_data['log_moment']

                # 计算拟合参数
                coefficients = np.polyfit(log_box_size, log_moment, 1)
                slope = coefficients[0]
                intercept = coefficients[1]

                partition_data[f'q{q_val:.1f}_log_ε'] = log_box_size
                partition_data[f'q{q_val:.1f}_log_Zq'] = log_moment

                # 添加拟合直线数据
                fitted_line = np.poly1d(coefficients)(log_box_size)
                partition_data[f'q{q_val:.1f}_fitted'] = fitted_line

        # 找到最大长度并补齐所有数据
        if partition_data:
            max_len = max(len(v) for v in partition_data.values())
            for key in partition_data:
                if len(partition_data[key]) < max_len:
                    partition_data[key] = list(partition_data[key]) + [np.nan] * (max_len - len(partition_data[key]))
            pd.DataFrame(partition_data).to_excel(writer, sheet_name='Partition_Function', index=False)

        # Sheet6: 拟合参数
        fit_params = {}
        for j in q_indices:
            if j < len(moment_curves):
                curve_data = moment_curves[j]
                q_val = all_q_values[j]

                log_box_size = curve_data['log_box_size']
                log_moment = curve_data['log_moment']

                coefficients = np.polyfit(log_box_size, log_moment, 1)

                fit_params[f'q={q_val:.1f}'] = {
                    'Slope': coefficients[0],
                    'Intercept': coefficients[1],
                    'R_squared': np.corrcoef(log_box_size, log_moment)[0, 1] ** 2
                }

        fit_params_df = pd.DataFrame(fit_params).T
        fit_params_df.to_excel(writer, sheet_name='Fitting_Parameters')

    print(f"Results for {title_prefix} have been saved to the report folder (SVG format)")
    print(f"  - Figure 1: {title_prefix}_Multifractal_Analysis.svg")
    print(f"  - Figure 2: {title_prefix}_Multifractal_Spectrum.svg")
    print(f"  - Statistical Parameters: {os.path.basename(params_filename)}")
    print(f"  - Plot Data: {os.path.basename(excel_filename)}")


def export_mf_results_to_csv(mf_results_list, title_prefix, export_dir="report"):
    """
    保持原函数名的兼容性包装函数
    """
    # 提取绘图需要的数据
    all_q_values = mf_results_list[0]['results']['q_values']
    all_Dq = [result['results']['generalized_dimensions_Dq'] for result in mf_results_list]
    all_tq = [result['results']['mass_exponents_tq'] for result in mf_results_list]
    all_aq = [result['results']['singularity_strengths_aq'] for result in mf_results_list]
    all_fq = [result['results']['singularity_spectrums_fq'] for result in mf_results_list]
    all_moment_curves = [result['results']['moment_boxsize_curves_Xq'] for result in mf_results_list]

    export_mf_results(mf_results_list, title_prefix, all_q_values, all_Dq, all_tq, all_aq, all_fq, all_moment_curves)