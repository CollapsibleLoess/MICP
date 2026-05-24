import numpy as np
from numpy import polyfit
from sklearn.isotonic import IsotonicRegression

def calculate_partition_functions(prob, q):
    """计算配分函数相关参数"""
    moment_mq = np.power(prob, q)
    partition_sum = np.sum(moment_mq)
    measure_uq = moment_mq / partition_sum
    return partition_sum, measure_uq

def generate_q_values(q_min=-10, q_max=10):
    """生成q值序列"""
    ql0 = np.linspace(-4,4, 160)
    ql1 = np.linspace(q_min, -1, 80)
    ql2 = np.linspace(1, q_max, 80)
    return np.around(np.unique(np.append(np.append(ql1, ql2), np.append([0, 1, 2], ql0))), 2)

def multifractal_analysis(probabilities, q_values):
    """
    执行多重分形分析
    Args:
        probabilities: 概率密度数组，一个嵌套列表，每个子列表代表一个尺度下的概率分布
        q_values: q值序列，用于探测不同奇异性的矩阶数
    Returns:
        tuple: 返回多重分形分析的所有结果
    """
    # 根据每个尺度下概率列表的长度，计算该尺度下的盒子总数
    box_counts = np.array([len(sublist) for sublist in probabilities])
    # 盒子大小(尺度ε)定义为盒子总数的倒数
    box_size = 1 / box_counts

    # 初始化用于存储最终结果的列表
    moment_boxsize_curves_Xq = []  # 存储每个q值下，配分函数与盒子大小的对数关系，用于绘图验证
    moments_Mq = []  # 存储每个q值下，所有尺度的配分函数Z_q(ε)
    generalized_dimensions_Dq = []  # 存储广义维数D(q)
    singularity_strengths_aq = []  # 存储奇异性指数α(q)
    mass_exponents_tq = []  # 存储质量指数τ(q)
    singularity_spectrums_fq = []  # 存储多重分形谱f(α)

    # 遍历所有指定的q值，q是“扫描旋钮”，用于探测不同强度的奇异区域
    for q in q_values:
        # 初始化当前q值下，所有尺度的计算结果
        partition_sums_Zq = []  # 存储当前q值下，所有尺度的配分函数 Z_q(ε)
        alpha_numerators_num = []  # 存储当前q值下，计算α(q)所需的分子项
        dimension_numerators_num = []  # 存储q=1时，计算D(q)所需的分子项（香农熵）

        # 遍历所有尺度（即遍历概率分布的每个子列表）
        for prob in probabilities:
            # 对当前尺度的概率分布prob和当前q值，计算配分函数Z_q(ε)和归一化测度μ_i
            # Z_q(ε) = Σ[p_i(ε)]^q
            partition_sum, measure_uq = calculate_partition_functions(prob, q)
            # 将当前尺度的配分函数Z_q(ε)存入列表
            partition_sums_Zq.append(partition_sum)

            # 计算α(q)的分子项：Σ[μ_i * log(p_i)]，其中μ_i = p_i^q / Z_q
            # 这是计算α(q)的直接方法的基础
            alpha_numerators_num.append(np.sum(measure_uq * np.log(prob)))

            # 特殊处理q=1的情况，为计算信息维D(1)做准备
            if q == 1:
                # 计算香农熵 Σ[p_i * log(p_i)]，这是D(1)的分子项
                dimension_numerators_num.append(np.sum(prob * np.log(prob)))
        # 记录当前q值下所有尺度的配分函数原始值

        # --- 核心计算：通过线性拟合求解标度指数 ---

        # 根据标度关系 Z_q(ε) ~ ε^τ(q)，两边取对数得 log(Z_q) ~ τ(q)*log(ε)
        # 对 log(Z_q) 与 log(ε) 进行线性拟合，斜率即为质量指数τ(q)
        # 这里使用box_size对log(Z_q)/log(ε)进行线性外推到ε→0的极限值
        #tau_q = polyfit(np.log(box_size), np.log(partition_sums_Zq), 1)[0]
        tau_q = polyfit(box_size, np.log(partition_sums_Zq)/np.log(box_size), 1)[1]
        # 同样地，α(q)通过 Σ[μ_i*log(p_i)] 与 log(ε) 的标度关系获得
        # 使用box_size对α_numerators/log(ε)进行线性外推到ε→0的极限值
        alpha_q = polyfit(box_size, alpha_numerators_num/np.log(box_size), 1)[1]
        # 计算广义维数D(q)
        # 对于q≠1: D(q) = τ(q) / (q - 1)，基于质量指数与广义维数的关系
        # 对于q=1: D(1)是信息维数，通过香农熵与log(ε)的标度关系直接计算
        #          使用box_size对dimension_numerators/log(ε)进行线性外推到ε→0的极限值
        dim_q = (polyfit(box_size, dimension_numerators_num/np.log(box_size), 1)[1] if q == 1
                 else tau_q / (q - 1))

        # 计算多重分形谱f(α)
        # f(α)通过勒让德变换(Legendre Transform)从τ(q)得到：f(α) = q*α(q) - τ(q)
        f_alpha = q * alpha_q - tau_q

        # 将当前q值计算出的所有结果存入总列表
        mass_exponents_tq.append(tau_q)
        singularity_strengths_aq.append(alpha_q)
        #singularity_strengths_aq = list(np.gradient(mass_exponents_tq))
        generalized_dimensions_Dq.append(dim_q)
        singularity_spectrums_fq.append(f_alpha)
        # (可选) 记录配分函数与盒子大小的对数关系，用于检查幂律行为
        moment_boxsize_data = {
            'log_box_size': np.log(box_size),
            'log_moment': np.log(partition_sums_Zq)
        }
        moment_boxsize_curves_Xq.append(moment_boxsize_data)
        moments_Mq.append(partition_sums_Zq)
    # 返回所有计算结果
    return box_size, box_counts, moment_boxsize_curves_Xq, moments_Mq, generalized_dimensions_Dq, \
        singularity_strengths_aq, mass_exponents_tq, singularity_spectrums_fq


def preprocess_mf(sample_distributions):
    """
    预处理多重分形分析并返回所有结果

    Args:
        sample_distributions: 单个样本的概率分布列表，格式为 [prob_2boxes, prob_4boxes, prob_8boxes, ...]
                             来自 prepare_multifractal_data 返回结果中的 'distributions' 字段

    Returns:
        tuple: (mf_results, mf_params) - 多重分形分析结果和特征参数
    """
    sample_distributions = [data['probs'] for data in sample_distributions.values()]
    # 生成 q 值序列
    q_values = generate_q_values()

    # 执行多重分形分析
    (
        box_size,
        box_counts,
        moment_boxsize_curves_Xq,
        moments_Mq,
        generalized_dimensions_Dq,
        singularity_strengths_aq,
        mass_exponents_tq,
        singularity_spectrums_fq
    ) = multifractal_analysis(sample_distributions, q_values)

    # 合并所有结果到一个字典
    mf_results = {
        "probability_densities": sample_distributions,
        "q_values": q_values,
        "box_counts": box_counts,
        "box_size": box_size,
        "moment_boxsize_curves_Xq": moment_boxsize_curves_Xq,
        "moments_Mq": moments_Mq,
        "generalized_dimensions_Dq": generalized_dimensions_Dq,
        "singularity_strengths_aq": singularity_strengths_aq,
        "mass_exponents_tq": mass_exponents_tq,
        "singularity_spectrums_fq": singularity_spectrums_fq
    }

    # 提取分析结果
    ql = mf_results["q_values"]
    Dq = mf_results["generalized_dimensions_Dq"]
    Aq = mf_results["singularity_strengths_aq"]
    Fq = mf_results["singularity_spectrums_fq"]

    # 计算特定 q 值的 Dq 值
    D_q = {}
    for target_q in [0, 1, 2, 10, -10]:
        # 找到最接近目标q值的索引
        closest_idx = np.argmin(np.abs(ql - target_q))
        D_q[f"D_q{target_q}"] = Dq[closest_idx]

    # 计算特征参数
    D0_minus_D1 = D_q["D_q0"] - D_q["D_q1"]
    D_neg10_minus_D10 = D_q["D_q-10"] - D_q["D_q10"]
    H = (D_q["D_q2"] + 1) / 2
    D_neg10_minus_D0 = D_q["D_q-10"] - D_q["D_q0"]
    D0_minus_D10 = D_q["D_q0"] - D_q["D_q10"]

    # 计算与 a 和 Fq 相关的参数
    a_qmin, a_qmax = min(Aq), max(Aq)
    a_mid = Aq[len(Aq) // 2]
    D_a = a_qmax - a_qmin
    R_d = (a_mid - a_qmax) - (a_qmin - a_mid)
    Fq_max, Fq_min = max(Fq), min(Fq)
    D_Fa = Fq_min - Fq_max

    # 合并特征参数
    mf_params = {
        **D_q,
        "D0_minus_D1": D0_minus_D1,
        "D_neg10_minus_D10": D_neg10_minus_D10,
        "H": H,
        "D_neg10_minus_D0": D_neg10_minus_D0,
        "D0_minus_D10": D0_minus_D10,
        "a_min": a_qmin,
        "a_max": a_qmax,
        "a_mid": a_mid,
        "D_a": D_a,
        "R_d": R_d,
        "Fq_max": Fq_max,
        "Fq_min": Fq_min,
        "D_Fa": D_Fa
    }

    print("多重分形特征参数:")
    for key, value in mf_params.items():
        print(f"  {key}: {value:.4f}")

    return mf_results, mf_params
