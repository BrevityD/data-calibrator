# 数据处理与归一化 

data_process.py 

# 测地线绘制 

draw_geo.py

# 算法 

大体思路, 写入 train_with_geo.py

for epoch in epoches:
    eval_math_loss, eval_code_loss = eval_current_model 
    normalized_eval_math_loss, normalized_eval_code_loss = normalize(eval_math_loss, eval_code_loss)
    draw_geodesic_line, velocity_vector = draw_geo(normalized_eval_math_loss, normalized_eval_code_loss, target_point)
    # regenerate data mixing by velocity_vector
    SFT (refer to../samples/experiment1/code_domain/run_exp.py)

**WARNING**：这里我还没想好训练途中需要保存哪些数据