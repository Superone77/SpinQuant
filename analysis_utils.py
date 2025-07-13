import torch
import os
from tqdm import tqdm
from datasets import load_dataset

def fp4_121_positive(x:torch.Tensor, stochastic_rounding:bool=False) -> torch.Tensor:
    if stochastic_rounding:
        noise = torch.rand_like(x) - 0.5
        step1 = torch.round(2.0 * x + noise) / 2.0
        step2 = torch.round(x + noise)
        step3 = 2.0 * torch.round(x / 2.0 + noise)
    else:
        step1 = torch.round(2.0 * x) / 2.0
        step2 = torch.round(x)
        step3 = 2.0 * torch.round(x / 2.0)
    
    mask1 = x < 2.0
    mask2 = x < 4.0

    return step1 * mask1 + step2 * (~mask1) * mask2 + step3 * (~mask1) * (~mask2)

FP8_E4M3_MAX = 448.0

def quant_nvfp4(x: torch.Tensor, 
                stochastic_rounding: bool = False, 
                scale_per_t = None,
                scale_per_b = None,
                batch_size = 1,
                vari_length = False):
    fp4_121_max = 6.0
    ori_shape = x.shape
    x = x.reshape(-1, 16)
    sign = x.sign()
    x_abs = x.abs()
    if scale_per_t == None:
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max
    quant_mode = os.environ['QUANT_MODE']
    if quant_mode == "Dynamic_Block" or quant_mode == "Calib_Block":
        scale_per_t = 1
    x_abs_scaled = x_abs / scale_per_t

    if scale_per_b == None:
        if batch_size == 1:
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        else:
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
            # scale_per_b = scale_per_b.max(dim=0, keepdim=True)[0]
        input_tensor = fp4_121_max / scale_per_b
        down_cast = input_tensor.to(torch.float8_e4m3fn)
        # down_cast = torch.ops.hpu.cast_to_fp8_v2(fp4_121_max / scale_per_b, 1.0, False, False, torch.float8_e4m3fn)[0]
        up_cast = down_cast.to(scale_per_b.dtype)
        scale_per_b = up_cast
        scale_per_b = torch.where((0 < scale_per_b) * (scale_per_b < torch.inf), scale_per_b, 1.0)
    
    x_fp4_abs = fp4_121_positive(x_abs_scaled * scale_per_b, stochastic_rounding) / scale_per_b

    return (sign * x_fp4_abs * scale_per_t).reshape(ori_shape)

def quant_mxfp4(x: torch.Tensor, 
                stochastic_rounding: bool = False, 
                scale = None,
                batch_size = 1,
                vari_length = False):
    fp4_121_max = 6.0
    ori_shape = x.shape
    x = x.reshape(-1, 32)
    sign = x.sign()
    x_abs = x.abs()
    if scale == None:
        scale = torch.pow(2.0, torch.floor(torch.log2(fp4_121_max / x_abs.max(dim=-1, keepdim=True)[0])))
        scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)
    else:
        scale = torch.pow(2.0, torch.floor(torch.log2(scale)))
        scale = torch.where((0 < scale) * (scale < torch.inf), scale, 1.0)
    x_fp4_abs = fp4_121_positive(x_abs * scale, stochastic_rounding) / scale
    return (sign * x_fp4_abs).reshape(ori_shape)

def quant_fp8(x: torch.Tensor):
    ori_type = x.dtype
    down_cast = x.to(torch.float8_e4m3fn)
    up_cast = down_cast.to(ori_type)
    return up_cast


import torch.nn.functional as F

def mse(a, b):
    return F.mse_loss(a.to(torch.float32), b.to(torch.float32))

LOG_EPS = 1e-12
def sqnr(x, x_hat):
    mse = (x - x_hat).pow(2).mean()
    power = x.pow(2).mean()
    return 10 * torch.log10(power / (mse + LOG_EPS))

def quant_nvfp4_analysis(x: torch.Tensor, 
                stochastic_rounding: bool = False, 
                scale_per_t = None,
                scale_per_b = None,
                batch_size = 1,
                vari_length = False,
                mode = "basic",
                group_size = 16):
    fp4_121_max = 6.0
    ori_shape = x.shape
    x = x.reshape(-1, group_size)  # shape: [num_blocks, 16]
    num_blocks = x.shape[0]
    sign = x.sign()
    x_abs = x.abs()

    if scale_per_t is None:
        nvfp4_max = fp4_121_max * FP8_E4M3_MAX
        scale_per_t = x_abs.max() / nvfp4_max


    x_abs_scaled = x_abs / scale_per_t

    if scale_per_b is None:
        if batch_size == 1 or vari_length:
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0]
        else:
            scale_per_b = x_abs_scaled.max(dim=-1, keepdim=True)[0].max(dim=0, keepdim=True)[0]

    input_tensor = fp4_121_max / scale_per_b
    down_cast = input_tensor.to(torch.float8_e4m3fn)
    up_cast = down_cast.to(scale_per_b.dtype)
    scale_per_b = torch.where((0 < up_cast) * (up_cast < torch.inf), up_cast, torch.tensor(1.0, device=up_cast.device))
    
    x_fp4_abs_list = []
    metrics = {"e2m1_mses":[], "e2m1_sqnrs":[],"e4m3_mses":[], "e4m3_sqnrs":[], "global_scaling":[scale_per_t.item()], "mses": [], "sqnrs":[]}
    
    
    for i in tqdm(range(num_blocks)):
        current_x_abs_scaled = x_abs_scaled[i:i+1]
        current_scale_q = scale_per_b[i:i+1]
        current_scale_ori = input_tensor[i:i+1]
        e2m1_ori = current_x_abs_scaled * current_scale_q
        e2m1_q = fp4_121_positive(current_x_abs_scaled * current_scale_q, stochastic_rounding)
        x_fp4_abs = e2m1_q / current_scale_q
        
        if mode == "basic":
            metrics["mses"].append(mse(e2m1_ori/current_scale_ori*scale_per_t, e2m1_q/current_scale_q*scale_per_t).item())
        else:
            metrics["e2m1_mses"].append(mse(e2m1_ori, e2m1_q).item())
            metrics["e2m1_sqnrs"].append(sqnr(e2m1_ori, e2m1_q).item())
            metrics["e4m3_mses"].append(mse(current_scale_ori, current_scale_q).item())
            metrics["e4m3_sqnrs"].append(sqnr(current_scale_ori, current_scale_q).item())
            metrics["mses"].append(mse(e2m1_ori/current_scale_ori*scale_per_t, e2m1_q/current_scale_q*scale_per_t).item())
            metrics["sqnrs"].append(sqnr(e2m1_ori/current_scale_ori*scale_per_t, e2m1_q/current_scale_q*scale_per_t).item())

        x_fp4_abs_list.append(x_fp4_abs)

    x_fp4_abs_all = torch.cat(x_fp4_abs_list, dim=0)
    result = (sign * x_fp4_abs_all * scale_per_t).reshape(ori_shape)

    # 打印每个 block 的 MSE 和 SQNR
    # print("Block-wise Metrics:")
    # for i, (mse_val, sqnr_val) in enumerate(zip(metrics["mses"], metrics["sqnrs"])):
    #     print(f"Block {i}: MSE = {mse_val:.6f}, SQNR = {sqnr_val:.2f} dB")

    return result, metrics
from concurrent.futures import ThreadPoolExecutor

def quant_nvfp4_analysis_wrapper(args):
    tensor, group_size = args
    return quant_nvfp4_analysis(tensor, group_size = group_size)

def parallel_quantization(in_act, out_act, weight, group_size):
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(quant_nvfp4_analysis_wrapper, 
                                    [(in_act, group_size), 
                                     (out_act, group_size), 
                                     (weight, group_size)]))
    q_in_act, metrics_in_act = results[0]
    q_out_act, metrics_out_act = results[1]
    q_weight, metrics_weight = results[2]
    
    return (q_in_act, metrics_in_act), (q_out_act, metrics_out_act), (q_weight, metrics_weight)

def global_scaling_grid_search_nvfp4(x, beta0, beta1, grid_num):
    device = x.device
    dtype = x.dtype
    absmax = x.abs().max().item()
    interval = (absmax * beta1 - absmax * beta0)/grid_num
    # 构建搜索范围
    scale_candidates = torch.arange(absmax * beta0, absmax * beta1 + interval, interval, device=device, dtype=dtype)
    
    min_mse = float('inf')
    best_scale = None
    best_result = None
    
    with torch.no_grad():
        for scale in scale_candidates:
            # 设置环境变量 QUANT_MODE
            import os
            os.environ['QUANT_MODE'] = 'Static'  # 防止进入 Dynamic Block 模式
            
            # 量化
            x_quant = quant_nvfp4(x, scale_per_t=scale.item())
            
            # 计算误差
            loss = mse(x, x_quant)
            
            if loss.item() < min_mse:
                min_mse = loss.item()
                best_scale = scale.item()
                best_result = x_quant
    print("absmax = ", absmax, "best_scale = ",best_scale)
    return best_scale, min_mse, best_result

def block_scaling_grid_search_nvfp4(x, beta0, beta1, grid_num):
    ori_shape = x.shape
    x = x.reshape(-1, 16)  # Flatten all but last dim to (N, 16)
    N = x.shape[0]
    x_abs = x.abs().max().item()
    fp4_121_max = 6.0
    nvfp4_max = fp4_121_max * FP8_E4M3_MAX
    scale_per_t = x_abs.max() / nvfp4_max
    # 初始化最佳 scale 和 MSE
    best_scales = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
    min_mse_tensor = torch.zeros(N, device=x.device, dtype=x.dtype)

    # 计算每个 block 的 absmax
    absmax_block = (x/scale_per_t).abs().max(dim=1, keepdim=True)[0]

    # 构建每个 block 的候选 scales
    scale_candidates = torch.linspace(beta0, beta1, grid_num, device=x.device, dtype=x.dtype)
    scale_candidates = scale_candidates.view(1, grid_num) * absmax_block  # shape: (N, grid_num)

    # 逐个 block 处理
    for i in range(N):
        x_block = x[i:i+1].clone().detach()
        candidates_i = scale_candidates[i:i+1]

        # 构造 batch 输入用于并行评估所有候选 scale
        x_block_expanded = x_block.expand(grid_num, -1)  # shape: (grid_num, 16)
        scale_expanded = candidates_i.expand(grid_num, -1)  # shape: (grid_num, 1)

        # 量化
        with torch.no_grad():
            quantized = quant_nvfp4(
                x_block_expanded,
                stochastic_rounding=False,
                scale_per_t=None,
                scale_per_b=scale_expanded,
                batch_size=grid_num,
                vari_length=False
            )

        # 计算 MSE
        mse_values = F.mse_loss(quantized, x_block_expanded, reduction='none').mean(dim=1)

        # 保存最小 MSE 及对应 scale
        min_idx = mse_values.argmin()
        best_scales[i] = candidates_i[0]
        min_mse_tensor[i] = mse_values[min_idx]

    best_result = quant_nvfp4(
                x.view(ori_shape),
                stochastic_rounding=False,
                scale_per_t=scale_per_t,
                scale_per_b=best_scales,
                batch_size=grid_num,
                vari_length=False)
    print("absmax = ", absmax_block, "best_scale = ",best_scales)
    return best_scales, min_mse_tensor,best_result





def block_scaling_grid_search_mxfp4(x, beta0, beta1, grid_num):
    ori_shape = x.shape
    x = x.reshape(-1, 32)  # Flatten all but last dim to (N, 16)
    N = x.shape[0]
    
    # 初始化最佳 scale 和 MSE
    best_scales = torch.zeros(N, 1, device=x.device, dtype=x.dtype)
    min_mse_tensor = torch.zeros(N, device=x.device, dtype=x.dtype)

    # 计算每个 block 的 absmax
    absmax_block = (x).abs().max(dim=-1, keepdim=True)[0]

    # 构建每个 block 的候选 scales
    scale_candidates = torch.linspace(beta0, beta1, grid_num, device=x.device, dtype=x.dtype)

    # 逐个 block 处理
    for i in range(N):
        x_block = x[i:i+1].clone()  # shape: (1, 32)
        absmax_i = absmax_block[i:i+1]  # shape: (1, 1)

        # 构造该 block 的候选 scales
        scale_candidates_i = scale_candidates * absmax_i.item()  # shape: (grid_num,)

        mse_list = []

        # 对每个候选 scale 单独处理
        for scale_val in scale_candidates_i:
            scale_val_tensor = torch.tensor([[scale_val]], device=x.device, dtype=x.dtype)

            with torch.no_grad():
                quantized = quant_mxfp4(
                    x_block,
                    stochastic_rounding=False,
                    scale=scale_val_tensor,
                    batch_size=1
                )

            # 计算 MSE
            mse = F.mse_loss(quantized, x_block).item()
            mse_list.append(mse)

        # 找出最小 MSE 对应的 scale
        min_idx = torch.argmin(torch.tensor(mse_list))
        best_scale = scale_candidates_i[min_idx]
        best_scales[i] = best_scale
        min_mse_tensor[i] = mse_list[min_idx]

    best_result = quant_mxfp4(
                x.view(ori_shape),
                stochastic_rounding=False,
                scale=best_scales,
                batch_size=grid_num)
                
    print("absmax = ", absmax_block, "best_scale = ",best_scales)
    return best_scales, min_mse_tensor,best_result


def global_scaling_grid_search_mxfp4(x, beta0, beta1, grid_num):
    device = x.device
    dtype = x.dtype
    absmax = x.abs().max().item()
    interval = (absmax * beta1 - absmax * beta0)/grid_num
    # 构建搜索范围
    scale_candidates = torch.arange(absmax * beta0, absmax * beta1 + interval, interval, device=device, dtype=dtype)
    
    min_mse = float('inf')
    best_scale = None
    best_result = None
    
    with torch.no_grad():
        for scale in scale_candidates:
            # 设置环境变量 QUANT_MODE
            import os
            os.environ['QUANT_MODE'] = 'Static'  # 防止进入 Dynamic Block 模式
            
            # 量化
            x_quant = quant_nvfp4(x, scale_per_t=scale.item())
            
            # 计算误差
            loss = mse(x, x_quant)
            
            if loss.item() < min_mse:
                min_mse = loss.item()
                best_scale = scale.item()
                best_result = x_quant
    print("absmax = ", absmax, "best_scale = ",best_scale)
    return best_scale, min_mse, best_result
    
def analysis_tensor(x):
    w = x.detach().float().flatten()
    w_range = (w.max() - w.min()).item()
    mu      = w.mean()
    var     = ((w - mu) ** 2).mean()
    m4      = ((w - mu) ** 4).mean()
    kurt    = (m4 / (var ** 2 + 1e-12)).item() - 3.0   # Fisher
    return w_range, kurt

def analysis_within_group(x, group_size):
    # Reshape x into groups of size group_size
    x = x.reshape(-1, group_size)

    # Compute mean of each group
    means = x.mean(dim=1)

    # Compute kurtosis of each group
    # Kurtosis formula: E[(X - μ)^4] / (E[(X - μ)^2]^2) - 3
    centered = x - means.unsqueeze(1)
    variance = (centered ** 2).mean(dim=1)
    fourth_moment = (centered ** 4).mean(dim=1)
    kurts = (fourth_moment / (variance ** 2)) - 3

    return means, kurts

def plot_weight(weight, name,save_file):
    # 获取权重矩阵的维度
    import torch
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    out_features, in_features = weight.shape

    # 创建 x, y 坐标网格
    x = np.arange(in_features)
    y = np.arange(out_features)
    x_mesh, y_mesh = np.meshgrid(x, y)

    # Flatten 数据以便绘图
    x_flat = x_mesh.flatten()
    y_flat = y_mesh.flatten()
    z_flat = weight.flatten()

    # 设置柱状图的尺寸
    dx = dy = 0.8  # 柱子的宽度
    dz = dz_flat = np.abs(z_flat)  # 使用绝对值作为高度

    # 创建 3D 图形
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 颜色映射
    colors = ['r' if val > 0 else 'b' for val in z_flat]  # 正值红色，负值蓝色

    # 绘制3D柱状图
    ax.bar3d(x_flat, y_flat, np.zeros_like(dz_flat), dx, dy, dz_flat, color=colors, shade=True)

    # 设置坐标轴标签
    ax.set_xlabel('Input Channels')
    ax.set_ylabel('Output Channels')
    ax.set_zlabel('Weight Magnitude')

    # 设置视角
    ax.view_init(elev=25, azim=-60)

    plt.title("3D Histogram of k_proj Weights")
    plt.savefig(save_file)

def plot_hist(data,filename):
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. 生成一些示例数据（比如正态分布）

    # 2. 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=200, color='skyblue', edgecolor='black', alpha=0.8)

    # 3. 添加标题和标签
    plt.title('Histogram of Numpy Data (Normal Distribution)', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # 4. 显示网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. 显示图形
    plt.tight_layout()
    plt.savefig(filename)

def plot_hist_2(data1, data2,filename):
    import numpy as np
    import matplotlib.pyplot as plt

    # 1. 生成一些示例数据（比如正态分布）

    # 2. 绘制直方图
    plt.figure(figsize=(10, 6))
    plt.hist(data1, bins=200, color='skyblue', edgecolor='black', alpha=0.5)
    plt.hist(data2, bins=200, color='yellow', edgecolor='black', alpha=0.5)

    # 3. 添加标题和标签
    plt.title('Histogram of Numpy Data (Normal Distribution)', fontsize=14)
    plt.xlabel('Value', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)

    # 4. 显示网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. 显示图形
    plt.tight_layout()
    plt.savefig(filename)

import numpy as np
import matplotlib.pyplot as plt

def plot_bar(x_data, y_data, filename):
    x_indexes = np.arange(len(x_data))
    bar_width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x_indexes, y_data, width=bar_width, label='Value')

    plt.xticks(x_indexes, x_data)
    plt.xlabel("Name")
    plt.ylabel("Value")
    plt.title("Bar")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plt.savefig(filename)
    plt.close()

def plot_box_2(data1, data2, filename):
    import matplotlib.pyplot as plt

    # 1. 创建绘图区域
    plt.figure(figsize=(8, 6))

    # 2. 绘制箱线图
    plt.boxplot([data1, data2], notch=False, patch_artist=True,
                boxprops=dict(facecolor='skyblue', color='blue'),
                medianprops=dict(color='red'),
                whiskerprops=dict(linestyle='--'),
                labels=['Data 1', 'Data 2'])

    # 3. 添加标题和标签
    plt.title('Box Plot Comparison', fontsize=14)
    plt.ylabel('Value', fontsize=12)

    # 4. 显示网格线
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 5. 保存图形并显示
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

from typing import List, Dict, Tuple
import os, pathlib

# ---------------------------------------------------------------------
# Plot util: N-group boxplot
# ---------------------------------------------------------------------
def plot_box_n(data_list: List[np.ndarray], labels: List[str], save_path: str) -> None:
    """
    Draw a boxplot for N groups of data side-by-side.

    Parameters
    ----------
    data_list : list of np.ndarray
        Each element = 1-D array of a group's statistic (e.g. mean/kurtosis).
    labels : list of str
        Tick labels for each group, same order as data_list.
    save_path : str
        File path to save the PNG.
    """
    
    plt.figure(figsize=(max(6, len(data_list) * 1.5), 5))
    plt.boxplot(data_list, labels=labels, showfliers=False)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    pathlib.Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()


def make_table(result_dict):
    """Generate table of results."""
    from pytablewriter import MarkdownTableWriter, LatexTableWriter

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]
    latex_writer.headers = ["Task", "Version", "Metric", "Value", "", "Stderr"]

    values = []

    for k, dic in result_dict["results"].items():
        version = result_dict["versions"][k]
        for m, v in dic.items():
            if m.endswith("_stderr"):
                continue

            if m + "_stderr" in dic:
                se = dic[m + "_stderr"]
                values.append([k, version, m,  v, "±",  se])
            else:
                print()
                values.append([k, version, m,  v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


# ---------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------
def get_wikitext_batch(tokenizer, seq_len, batch_size, device="cuda"):
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    txt = "\n".join(ds["text"][: batch_size * seq_len * 2])
    tk = tokenizer(
        txt,
        return_tensors="pt",
        truncation=True,
        max_length=seq_len * batch_size,
    )
    inp = {
        "input_ids": tk["input_ids"][:, :seq_len].to(device),
        "attention_mask": tk["attention_mask"][:, :seq_len].to(device),
    }
    return inp

    

def capture_nth_layer_act(model, substr: str, nth: int, inputs) -> torch.Tensor:
    """
    Capture input and activation from the `nth` Linear layer whose name contains `substr`.
    Returns (input_activation, output_activation, weight)
    """
    counter, captured = 0, {}

    def hook(_, input, output):
        # Assuming input is a single Tensor (common for Linear layers)
        captured["in_act"] = input[0].detach().reshape(-1, input[0].size(-1)).cpu()
        captured["out_act"] = output.detach().reshape(-1, output.size(-1)).cpu()

    handle = None
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and substr in name:
            weight = mod.weight
            if counter == nth:
                handle = mod.register_forward_hook(hook)
                break
            counter += 1

    if handle is None:
        raise RuntimeError(f"Layer {substr}[{nth}] not found in model.")

    with torch.no_grad():
        model(**inputs)

    handle.remove()
    return captured["in_act"], captured["out_act"], weight.detach().cpu()


def nvfp4_quantization_mse(in_act, out_act,weight,tag, out_dir):
    q_in_act,metrics_in_act = quant_nvfp4_analysis(in_act)
    q_out_act,metrics_out_act = quant_nvfp4_analysis(out_act)
    q_weight,metrics_weight = quant_nvfp4_analysis(weight)

    return metrics_in_act["mses"],metrics_out_act["mses"],metrics_weight["mses"]