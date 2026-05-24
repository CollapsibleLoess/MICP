import numpy as np
import pandas as pd


def load_distribution_data(file_path):
    """
    加载分布数据文件

    Args:
        file_path (str or file-like object): 数据文件路径或文件对象

    Returns:
        pd.DataFrame: 包含分布数据的DataFrame，第一列为尺寸值，第二列为概率密度
    """
    if isinstance(file_path, str):
        distribution_data = pd.read_csv(file_path, sep='\s+', header=0)
    else:
        distribution_data = pd.read_csv(file_path, sep='\s+', header=0)

    return distribution_data


def preprocess_power_law_to_linear(distribution_dataframe):
    """
    将幂律分布的横坐标数据对数变换后线性映射到[0,1]区间

    Args:
        distribution_dataframe (pd.DataFrame): 原始分布数据，第一列为横坐标值

    Returns:
        tuple: (线性化后的DataFrame, 原始DataFrame副本)
    """
    linearized_dataframe = distribution_dataframe.copy()
    original_dataframe = distribution_dataframe.copy()

    # 提取横坐标值并进行对数变换
    x_coordinates = distribution_dataframe.iloc[:, 0].values
    log_x_coordinates = np.log10(x_coordinates + 1e-10)  # 添加小值避免log(0)

    # 线性映射到[0, 1]区间
    log_min_value, log_max_value = np.min(log_x_coordinates), np.max(log_x_coordinates)

    if log_max_value == log_min_value:
        # 处理所有值相同的特殊情况
        linear_x_coordinates = np.ones_like(x_coordinates)
    else:
        linear_x_coordinates = (log_x_coordinates - log_min_value) / (log_max_value - log_min_value)

    # 更新线性化数据的横坐标
    linearized_dataframe.iloc[:, 0] = linear_x_coordinates

    # 检查是否有足够的数据点进行插值
    if len(linearized_dataframe) < 3:
        raise ValueError("数据点不足，无法进行二次插值")

    # 在线性空间内使用二次样条线插值到1024个数据点
    from scipy.interpolate import interp1d

    # 获取线性化后数据的范围
    linear_x = linearized_dataframe.iloc[:, 0].values
    x_min, x_max = np.min(linear_x), np.max(linear_x)

    # 在数据的实际范围内创建1024个均匀分布的点
    new_x = np.linspace(x_min, x_max, 1024)

    # 创建新的DataFrame存储插值结果
    interpolated_df = pd.DataFrame()
    interpolated_df[linearized_dataframe.columns[0]] = new_x

    # 对每一列数据进行二次样条线插值（除了第一列横坐标）
    for col in linearized_dataframe.columns[1:]:
        f = interp1d(linear_x, linearized_dataframe[col].values,
                     kind='quadratic', bounds_error=False, fill_value='extrapolate')
        interpolated_df[col] = f(new_x)

    # 对所有数据取绝对值
    for col in interpolated_df.columns:
        interpolated_df[col] = np.abs(interpolated_df[col])

    linearized_dataframe = interpolated_df

    return linearized_dataframe, original_dataframe


def downsample_1d_data(data_array, box_count):
    """
    对一维分布数据进行分箱降采样，计算每个箱子的概率质量

    Args:
        data_array (np.ndarray): 形状为(N, 2)的数组，第一列为位置，第二列为概率密度
        box_count (int): 分箱数量

    Returns:
        tuple: (非零概率数组, 对应的箱子中心位置数组)
    """
    x_positions, y_densities = data_array[:, 0], data_array[:, 1]

    # 处理所有x值相同的特殊情况
    if np.all(x_positions == x_positions[0]):
        return np.array([1.0]), np.array([x_positions[0]])

    # 创建等间距的分箱边界
    x_min, x_max = np.min(x_positions), np.max(x_positions)
    bin_edges = np.linspace(x_min, x_max, box_count + 1)
    bin_edges[-1] = x_max + 1e-9  # 确保最大值被包含在最后一个箱子中

    # 将数据点分配到对应的箱子
    bin_indices = np.digitize(x_positions, bin_edges)

    # 计算每个箱子内的概率密度总和
    density_sums_per_bin = np.bincount(
        bin_indices,
        weights=y_densities,
        minlength=box_count + 2
    )[1:box_count + 1]  # 去除边界箱子

    total_density_sum = np.sum(density_sums_per_bin)

    # 归一化为概率分布
    box_probabilities = (
        density_sums_per_bin / total_density_sum
        if total_density_sum > 0
        else np.zeros(box_count)
    )

    # 计算箱子中心位置
    box_center_positions = (bin_edges[:-1] + bin_edges[1:]) / 2

    return box_probabilities, box_center_positions

def remove_zero_y_values(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    清除DataFrame中y值为0的数据对。

    假设DataFrame有两列，第一列为x，第二列为y。
    此函数会移除所有第二列值为0的行。

    Args:
        dataframe (pd.DataFrame): 包含x、y值的DataFrame。

    Returns:
        pd.DataFrame: 清除掉y值为0的行之后的新DataFrame。
    """
    # 假设y值在第二列（索引为1）
    # 使用布尔索引来选择第二列值不为0的行
    # 使用.copy()避免后续操作出现SettingWithCopyWarning
    cleaned_dataframe = dataframe[dataframe.iloc[:, 1] != 0].copy()
    return cleaned_dataframe

def prepare_multifractal_data(distribution_dataframe, box_count_list, data_type_name=""):
    """
    对分布数据进行预处理和多尺度降采样，为多重分形分析准备数据

    Args:
        distribution_dataframe (pd.DataFrame): 原始分布数据
        box_count_list (list): 不同分箱数量的列表，如[2, 4, 8, 16, 32]
        data_type_name (str, optional): 数据类型标识符

    Returns:
        dict: 包含预处理结果的字典
            - 'name': 数据类型名称
            - 'original_data': 原始数据副本
            - 'lineared_data': 横坐标线性化后的数据
            - 'downsampled_data': 多尺度降采样结果字典
    """
    # 新增步骤：在进行任何处理之前，首先清除y值为0的数据点
    cleaned_dataframe = remove_zero_y_values(distribution_dataframe)
    # 对横坐标进行线性化预处理
    linearized_dataframe, original_dataframe = preprocess_power_law_to_linear(
        cleaned_dataframe
    )

    # 存储不同分箱数量下的降采样结果
    multi_scale_downsampled_data = {}

    for current_box_count in box_count_list:
        # 对线性化数据进行降采样分箱
        box_probabilities, box_center_positions = downsample_1d_data(
            linearized_dataframe.values,
            current_box_count
        )

        # 存储当前分箱数量下的结果
        multi_scale_downsampled_data[current_box_count] = {
            'probs': box_probabilities,
            'centers': box_center_positions
        }

    # 构建返回结果
    preprocessing_results = {
        'name': data_type_name,
        'original_data': original_dataframe,
        'lineared_data': linearized_dataframe,
        'downsampled_data': multi_scale_downsampled_data,
    }

    return preprocessing_results