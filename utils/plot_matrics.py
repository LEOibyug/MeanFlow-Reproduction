# utils/plot_metrics.py
import matplotlib
matplotlib.use('Agg') # <--- 必须放在 import pyplot 之前！
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob

def plot_all_logs(log_dir="logs", output_file="results/loss_comparison.png"):
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # 查找所有 csv log
    csv_files = glob.glob(os.path.join(log_dir, "*.csv"))
    if not csv_files:
        print("No log files found in logs/")
        return

    plt.figure(figsize=(10, 6))
    
    for csv_file in csv_files:
        try:
            # 读取数据
            df = pd.read_csv(csv_file)
            exp_name = os.path.basename(csv_file).replace("_log.csv", "")
            
            # 绘制曲线
            plt.plot(df['epoch'], df['loss'], label=exp_name, linewidth=2)
            print(f"Plotting {exp_name}...")
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 保存图片而非显示
    plt.savefig(output_file, dpi=150)
    print(f"Loss curve saved to {output_file}")
    plt.close()

if __name__ == "__main__":
    plot_all_logs()
