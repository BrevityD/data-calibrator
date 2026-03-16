import json
import torch
import random
import torch.nn as nn


def load_and_extract(file_path):
    """提取原始数据并进行列采样"""
    with open(file_path, 'r') as file:
        data = json.load(file)
        
    row_keys = sorted(data.keys(), key=int)
    col_keys = sorted(data[row_keys[0]].keys(), key=int)
    
    matrix = [[data[r][c] for c in col_keys] for r in row_keys]
    matrix = [row[::19] for row in matrix] # 每 19 列采样
    
    matrix_tuples = [
        [{'eval_math_loss': item['eval_math_loss'], 'eval_code_loss': item['eval_code_loss']} for item in row]
        for row in matrix
    ]
    return matrix_tuples

def apply_normalization(matrix, math_min, math_range, code_min, code_range):
    """应用归一化"""
    matrix_normalized = [
        [
            {
                'eval_math_loss': 0.0 if math_range == 0 else (item['eval_math_loss'] - math_min) / math_range,
                'eval_code_loss': 0.0 if code_range == 0 else (item['eval_code_loss'] - code_min) / code_range
            }
            for item in row
        ]
        for row in matrix
    ]
    return matrix_normalized

def calculate_variances(matrix_normalized, task):
    """独立计算指定任务矩阵的 variances (Deltas)"""
    if task == 'm2c':
        main_direction, branch_direction = 'math', 'code'
    elif task == 'c2m':
        main_direction, branch_direction = 'code', 'math'
    else:
        raise ValueError("Task must be 'm2c' or 'c2m'")
        
    matrix_with_vars = []
    for row in matrix_normalized:
        new_row = []
        for item in row:
            new_item = item.copy()
            new_item[f"var_{branch_direction}"] = (None, None)
            new_item[f"var_{main_direction}"] = (None, None)
            new_row.append(new_item)
        matrix_with_vars.append(new_row)
        
    # 计算 Branch Direction Deltas (水平方向)
    for r_idx, row in enumerate(matrix_with_vars):
        for c_idx in range(len(row) - 1):
            curr_item = row[c_idx]
            next_item = row[c_idx+1]
            
            delta_math = next_item['eval_math_loss'] - curr_item['eval_math_loss']
            delta_code = next_item['eval_code_loss'] - curr_item['eval_code_loss']
            
            curr_item[f"var_{branch_direction}"] = (delta_math, delta_code)
            
    # 计算 Main Direction Deltas (垂直方向，基于第一列)
    for r_idx in range(len(matrix_with_vars) - 1):
        curr_item = matrix_with_vars[r_idx][0]
        next_item = matrix_with_vars[r_idx+1][0]
        
        delta_math = next_item['eval_math_loss'] - curr_item['eval_math_loss']
        delta_code = next_item['eval_code_loss'] - curr_item['eval_code_loss']
        
        curr_item[f"var_{main_direction}"] = (delta_math, delta_code)
        
    return matrix_with_vars

def process_and_normalize_globally(file_m2c, file_c2m):
    """主控函数：统一归一化，分别计算方差"""
    # 1. 独立提取数据
    mat_m2c = load_and_extract(file_m2c)
    mat_c2m = load_and_extract(file_c2m)
    
    # 2. 统一寻找全局 Min-Max
    all_math_vals = [item['eval_math_loss'] for mat in (mat_m2c, mat_c2m) for row in mat for item in row]
    all_code_vals = [item['eval_code_loss'] for mat in (mat_m2c, mat_c2m) for row in mat for item in row]
    
    math_min, math_max = min(all_math_vals), max(all_math_vals)
    code_min, code_max = min(all_code_vals), max(all_code_vals)
    
    math_range = math_max - math_min
    code_range = code_max - code_min
    
    # 3. 统一使用全局 Min-Max 进行归一化
    mat_m2c_norm = apply_normalization(mat_m2c, math_min, math_range, code_min, code_range)
    mat_c2m_norm = apply_normalization(mat_c2m, math_min, math_range, code_min, code_range)
    
    # 4. 独立计算各自的 variances
    mat_m2c_final = calculate_variances(mat_m2c_norm, task='m2c')
    mat_c2m_final = calculate_variances(mat_c2m_norm, task='c2m')
    
    return mat_m2c_final, mat_c2m_final

# ================================
# Example usage:
mat1_final, mat2_final = process_and_normalize_globally('m2c.json', 'c2m.json')
# ================================

def extract_unified_data_points(mat1, mat2, target_var_key):
    """
    将两个矩阵的数据扁平化，并根据指定的目标 variance 键提取合法的训练数据。
    
    :param mat1: 经过归一化和方差计算的矩阵1 (例如 m2c)
    :param mat2: 经过归一化和方差计算的矩阵2 (例如 c2m)
    :param target_var_key: 你想要提取的预测目标键名，如 'var_math' 或 'var_code'
    :return: 统一的 data_points 列表，格式为 [(features, targets), ...]
    """
    data_points = []
    
    # 将两个矩阵放在同一个列表中遍历
    combined_matrices = [mat1, mat2]
    
    for matrix in combined_matrices:
        for row in matrix:
            for item in row:
                # 检查该元素是否存在有效的 variance 数据（剔除边界的 None 值）
                if item.get(target_var_key) not in [(None, None), None]:
                    
                    # Input features: 当前的归一化 losses
                    features = [item['eval_math_loss'], item['eval_code_loss']]
                    
                    # Target: 对应方向上的变化量 Deltas
                    targets = list(item[target_var_key])
                    
                    data_points.append((features, targets))
                    
    return data_points

# ================================
# Example usage:
# 假设你想训练一个模型，专门预测向 math 优化时带来的 loss 变化
target_key = 'var_math'
math_dataset = extract_unified_data_points(mat1_final, mat2_final, target_key)
target_key = 'var_code'
code_dataset = extract_unified_data_points(mat1_final, mat2_final, target_key)
# ================================

def check_normalization(dataset, name="Dataset"):
    """
    Check if the features in the dataset are properly normalized (between 0 and 1).
    """
    if not dataset:
        print(f"{name} is empty.")
        return

    # Extract features: dataset is a list of (features, targets)
    # features is [math_loss_norm, code_loss_norm]
    features = [d[0] for d in dataset]
    
    # Convert to tensor for easier stat calculation
    features_tensor = torch.tensor(features)
    
    math_vals = features_tensor[:, 0]
    code_vals = features_tensor[:, 1]
    
    print(f"--- Checking {name} Normalization ---")
    print(f"Feature shape: {features_tensor.shape}")
    print(f"Math Loss (dim 0): Min = {math_vals.min():.4f}, Max = {math_vals.max():.4f}, Mean = {math_vals.mean():.4f}")
    print(f"Code Loss (dim 1): Min = {code_vals.min():.4f}, Max = {code_vals.max():.4f}, Mean = {code_vals.mean():.4f}")
    
    is_normal = (features_tensor >= 0.0).all() and (features_tensor <= 1.0).all()
    if is_normal:
        print("✅ Normalization check passed: All feature values are within [0, 1].")
    else:
        print("❌ Normalization check failed: Values found outside [0, 1].")
    print("-" * 30)

# Check both datasets
check_normalization(math_dataset, "Math Dataset")
check_normalization(code_dataset, "Code Dataset")


def check_targets(dataset, name="Dataset"):
    """
    Check statistics for the targets (deltas) in the dataset.
    """
    if not dataset:
        print(f"{name} is empty.")
        return

    # Extract targets: dataset is a list of (features, targets)
    # targets is [delta_math, delta_code]
    targets = [d[1] for d in dataset]
    
    # Convert to tensor for easier stat calculation
    targets_tensor = torch.tensor(targets)
    
    delta_math = targets_tensor[:, 0]
    delta_code = targets_tensor[:, 1]
    
    print(f"--- Checking {name} Targets (Deltas) ---")
    print(f"Target shape: {targets_tensor.shape}")
    print(f"Delta Math (dim 0): Min = {delta_math.min():.6f}, Max = {delta_math.max():.6f}, Mean = {delta_math.mean():.6f}, Std = {delta_math.std():.6f}")
    print(f"Delta Code (dim 1): Min = {delta_code.min():.6f}, Max = {delta_code.max():.6f}, Mean = {delta_code.mean():.6f}, Std = {delta_code.std():.6f}")
    print("-" * 30)

# Check targets for both datasets
check_targets(math_dataset, "Math Dataset")
check_targets(code_dataset, "Code Dataset")