import matplotlib
# 必须放在 import pyplot 之前，强制使用非交互式后端
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

def plot_losses(log_dir="logs", output_path="results/loss_curve.png", smooth_window=100):
    """
    读取 logs 目录下的所有 csv 文件并绘制对比曲线。
    smooth_window: 滑动平均窗口大小，用于平滑震荡的 Loss
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 查找所有 csv 文件
    csv_files = sorted(glob.glob(os.path.join(log_dir, "*.csv")))
    if not csv_files:
        print(f"No CSV files found in {log_dir}")
        return

    plt.figure(figsize=(12, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 蓝、橙、绿、红
    
    for i, file_path in enumerate(csv_files):
        try:
            # 读取数据
            df = pd.read_csv(file_path)
            
            # 获取实验名称 (文件名去掉 _log.csv)
            exp_name = os.path.basename(file_path).replace("_log.csv", "")
            color = colors[i % len(colors)]
            
            # 确定 X 轴 (Step) 和 Y 轴 (Loss)
            if 'step' in df.columns:
                x = df['step']
            elif 'epoch' in df.columns:
                x = df['epoch']
            else:
                x = df.index
            
            y = df['loss']
            
            # 1. 绘制原始数据 (半透明背景)
            plt.plot(x, y, color=color, alpha=0.15, linewidth=1)
            
            # 2. 绘制平滑数据 (实线)
            # 使用 pandas 的 rolling mean 进行平滑
            if len(y) > smooth_window:
                y_smooth = y.rolling(window=smooth_window, min_periods=1).mean()
                plt.plot(x, y_smooth, color=color, alpha=1.0, linewidth=2, label=f"{exp_name} (smoothed)")
            else:
                plt.plot(x, y, color=color, alpha=1.0, linewidth=2, label=exp_name)
                
            print(f"Loaded {exp_name}: {len(df)} steps")
            
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    # 图表装饰
    plt.title("Training Loss Comparison (MSE)", fontsize=16)
    plt.xlabel("Training Steps", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # 保存
    plt.savefig(output_path, dpi=200)
    print(f"\n✅ Loss curve saved to: {output_path}")

if __name__ == "__main__":
    # 您可以调整 window 大小，数据点越多，window 应该越大
    plot_losses(smooth_window=500)
