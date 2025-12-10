import matplotlib
# 必须放在 import pyplot 之前，强制使用非交互式后端 (服务器环境防报错)
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
import numpy as np

def smooth_data(y, window):
    """使用卷积进行滑动平均平滑"""
    if len(y) < window:
        return y
    box = np.ones(window) / window
    y_smooth = np.convolve(y, box, mode='valid')
    return y_smooth

def plot_single_metric(csv_files, metric_col, output_path, title, ylabel, smooth_window=100):
    """绘制单个指标的对比图"""
    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'] 
    
    has_data = False
    
    for i, file_path in enumerate(csv_files):
        try:
            df = pd.read_csv(file_path)
            exp_name = os.path.basename(file_path).replace("_log.csv", "")
            color = colors[i % len(colors)]
            
            # 确定 X 轴
            if 'step' in df.columns:
                x = df['step']
            elif 'epoch' in df.columns:
                x = df['epoch']
            else:
                x = df.index
            
            # 确定 Y 轴数据
            y = None
            if metric_col in df.columns:
                y = df[metric_col]
            # 兼容处理: 如果请求的是 mse_loss 但文件里只有 loss (通常是 Baseline), 则使用 loss
            elif metric_col == 'mse_loss' and 'loss' in df.columns:
                y = df['loss']
                exp_name += " (legacy)"
            
            if y is not None:
                has_data = True
                # 1. 绘制原始数据 (半透明背景)
                plt.plot(x, y, color=color, alpha=0.15, linewidth=1)
                
                # 2. 绘制平滑数据 (实线)
                # 处理卷积平滑导致的长度变化
                if len(y) > smooth_window:
                    y_smooth = smooth_data(y.values, smooth_window)
                    # 对齐 x 轴 (取中间段或后段)
                    x_smooth = x.iloc[smooth_window-1:]
                    plt.plot(x_smooth, y_smooth, color=color, alpha=1.0, linewidth=2, label=exp_name)
                else:
                    plt.plot(x, y, color=color, alpha=1.0, linewidth=2, label=exp_name)
                
                print(f"  -> Plotting {metric_col} for {exp_name}")
            else:
                print(f"  -> Skipping {exp_name} (column '{metric_col}' not found)")

        except Exception as e:
            print(f"  -> Error reading {file_path}: {e}")

    if has_data:
        plt.title(title, fontsize=16)
        plt.xlabel("Training Steps", fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.legend(fontsize=10, loc='upper right')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.yscale('linear') 
        # 如果是 MSE，限制一下 Y 轴范围防止某些极端值破坏视图
        if 'mse' in metric_col.lower():
            plt.ylim(0, 2.0) # 根据经验，MSE 通常在 0~1 之间
            
        plt.tight_layout()
        plt.savefig(output_path, dpi=200)
        print(f"✅ Saved chart to: {output_path}")
        plt.close()
    else:
        print(f"⚠️ No data found for metric '{metric_col}', skipping chart.")
        plt.close()

def plot_losses(log_dir="logs", output_dir="results", smooth_window=100):
    os.makedirs(output_dir, exist_ok=True)
    csv_files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    
    if not csv_files:
        print(f"No CSV files found in {log_dir}")
        return

    print(f"Found log files: {[os.path.basename(f) for f in csv_files]}")

    # 1. 绘制 MSE Loss (主要关注指标)
    # 这会展示真实的误差下降趋势
    plot_single_metric(
        csv_files, 
        metric_col="mse_loss", 
        output_path=os.path.join(output_dir, "loss_curve_mse.png"),
        title="Training MSE Loss",
        ylabel="MSE Loss",
        smooth_window=smooth_window
    )

    # 2. 绘制 Weighted Loss (优化目标)
    # 对于 p=1.0 的 MeanFlow，这条线应该在 1.0 附近
    plot_single_metric(
        csv_files, 
        metric_col="weighted_loss", 
        output_path=os.path.join(output_dir, "loss_curve_weighted.png"),
        title="Weighted Training Objective (Adaptive)",
        ylabel="Weighted Loss",
        smooth_window=smooth_window
    )

if __name__ == "__main__":
    # 平滑窗口大小，建议设为 100-500
    plot_losses(smooth_window=200)