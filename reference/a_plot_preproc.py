import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import os

plt.rcParams['font.sans-serif'] = ['Times New Roman + SimSun']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'stix'


def visualize_results(preprocessing_results_list, data_category="", base_fontsize=16):
    """
    Batch visualization of multifractal preprocessing results

    Args:
        preprocessing_results_list (list): List of preprocessing results
        data_category (str): Data category name (e.g., "Particle Size", "Pore Size")
        base_fontsize (int): Base font size for all text elements (default: 10)
    """
    if not preprocessing_results_list:
        print(f"Warning: {data_category} data list is empty")
        return

    # Create report folder
    report_dir = "report"
    if not os.path.exists(report_dir):
        os.makedirs(report_dir)
        print(f"Created report folder: {report_dir}")

    # Calculate derived font sizes based on base_fontsize
    label_fontsize = base_fontsize
    tick_fontsize = base_fontsize * 0.9
    legend_fontsize = base_fontsize * 0.8
    textbox_fontsize = base_fontsize * 0.95

    n_datasets = len(preprocessing_results_list)

    # Optimized subplot layout for compact design
    if n_datasets == 1:
        fig_rows, fig_cols = 1, 1
        figsize = (8, 6)
    elif n_datasets <= 4:
        fig_rows, fig_cols = 2, 2
        figsize = (12, 9)
    elif n_datasets <= 6:
        fig_rows, fig_cols = 2, 3
        figsize = (14, 8)
    elif n_datasets <= 9:
        fig_rows, fig_cols = 3, 3
        figsize = (14, 12)
    else:
        fig_rows = int(np.ceil(n_datasets / 4))
        fig_cols = 4
        figsize = (20, 4 * fig_rows)

    # Color scheme
    colors = cm.tab10(np.linspace(0, 1, max(10, n_datasets)))

    # 1. Original Data Comparison
    fig1, axes1 = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                               constrained_layout=True)
    if n_datasets == 1:
        axes1 = [axes1]
    else:
        axes1 = axes1.flatten()

    # 2. Linearized Data Comparison
    fig2, axes2 = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                               constrained_layout=True)
    if n_datasets == 1:
        axes2 = [axes2]
    else:
        axes2 = axes2.flatten()

    # 3. Multi-scale Downsampling Results
    fig3, axes3 = plt.subplots(fig_rows, fig_cols, figsize=figsize,
                               constrained_layout=True)
    if n_datasets == 1:
        axes3 = [axes3]
    else:
        axes3 = axes3.flatten()

    for idx, result in enumerate(preprocessing_results_list):
        dataset_name = result['name']
        original_data = result['original_data']
        linearized_data = result['lineared_data']
        downsampled_data = result['downsampled_data']

        # Plot 1: Original Data (X-axis log, Y-axis linear)
        if idx < len(axes1):
            ax1 = axes1[idx]
            ax1.semilogx(original_data.iloc[:, 0], original_data.iloc[:, 1],
                         'o-', color=colors[idx], markersize=3, linewidth=1.2, alpha=0.8)
            ax1.set_xlabel('Size (log scale)', fontsize=label_fontsize)
            ax1.set_ylabel('Probability Density', fontsize=label_fontsize)
            ax1.tick_params(labelsize=tick_fontsize)
            ax1.grid(True, alpha=0.25, linewidth=0.5)

            # Add internal label instead of title
            ax1.text(0.05, 0.95, f'{dataset_name}\nOriginal Data',
                     transform=ax1.transAxes, fontsize=textbox_fontsize,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Plot 2: Linearized Data (both axes linear)
        if idx < len(axes2):
            ax2 = axes2[idx]
            ax2.plot(linearized_data.iloc[:, 0], linearized_data.iloc[:, 1],
                     'o-', color=colors[idx], markersize=3, linewidth=1.2, alpha=0.8)
            ax2.set_xlabel('Normalized Position [0,1]', fontsize=label_fontsize)
            ax2.set_ylabel('Probability Density', fontsize=label_fontsize)
            ax2.tick_params(labelsize=tick_fontsize)
            ax2.grid(True, alpha=0.25, linewidth=0.5)

            # Add internal label
            ax2.text(0.05, 0.95, f'{dataset_name}\nLinearized Data',
                     transform=ax2.transAxes, fontsize=textbox_fontsize,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))

        # Plot 3: Multi-scale Downsampling (both axes linear)
        if idx < len(axes3):
            ax3 = axes3[idx]
            box_counts = list(downsampled_data.keys())

            for i, box_count in enumerate(box_counts):
                centers = downsampled_data[box_count]['centers']
                probs = downsampled_data[box_count]['probs']

                alpha_val = 0.4 + 0.6 * (i / len(box_counts))
                ax3.plot(centers, probs, 'o-',
                         label=f'N={box_count}',
                         alpha=alpha_val, markersize=2.5, linewidth=0.8)

            ax3.set_xlabel('Box Centers', fontsize=label_fontsize)
            ax3.set_ylabel('Box Probabilities', fontsize=label_fontsize)
            ax3.tick_params(labelsize=tick_fontsize)
            ax3.grid(True, alpha=0.25, linewidth=0.5)
            ax3.legend(fontsize=legend_fontsize, loc='best', framealpha=0.7, ncol=2)

            # Add internal label
            ax3.text(0.05, 0.95, f'{dataset_name}',
                     transform=ax3.transAxes, fontsize=textbox_fontsize,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    # Hide unused subplots
    for idx in range(n_datasets, len(axes1)):
        axes1[idx].set_visible(False)
        axes2[idx].set_visible(False)
        axes3[idx].set_visible(False)

    # Save figures as SVG to report folder
    for fig, title_suffix in zip([fig1, fig2, fig3],
                                 ['Original_Data', 'Linearized_Data', 'Multiscale_Downsampling']):
        filename = os.path.join(report_dir, f'{data_category}_{title_suffix}.svg')
        fig.savefig(filename, format='svg', bbox_inches='tight', dpi=300)
        print(f"Saved figure: {filename}")

    plt.close('all')  # Clean up memory

    # Create data summary table
    create_data_summary_table(preprocessing_results_list, data_category, report_dir)


def create_data_summary_table(preprocessing_results_list, data_category="", report_dir="report"):
    """
    Create data processing summary table

    Args:
        preprocessing_results_list (list): List of preprocessing results
        data_category (str): Data category name
        report_dir (str): Report folder path
    """
    summary_data = []

    for result in preprocessing_results_list:
        dataset_name = result['name']
        original_data = result['original_data']
        downsampled_data = result['downsampled_data']

        # Statistical information
        n_original_points = len(original_data)
        x_range = f"{original_data.iloc[:, 0].min():.2e} - {original_data.iloc[:, 0].max():.2e}"
        y_mean = original_data.iloc[:, 1].mean()

        # Downsampling information
        box_counts = list(downsampled_data.keys())
        min_boxes = min(box_counts)
        max_boxes = max(box_counts)

        summary_data.append({
            'Dataset': dataset_name,
            'Original Points': n_original_points,
            'X Range': x_range,
            'Y Mean': f"{y_mean:.4f}",
            'Min Boxes': min_boxes,
            'Max Boxes': max_boxes
        })

    # Create DataFrame and print
    summary_df = pd.DataFrame(summary_data)
    print(f"\n{data_category} - Data Processing Summary:")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("=" * 80)

    # Save summary table to report folder
    summary_filename = os.path.join(report_dir, f'{data_category}_Data_Summary.csv')
    summary_df.to_csv(summary_filename, index=False, encoding='utf-8-sig')
    print(f"Saved data summary table: {summary_filename}")