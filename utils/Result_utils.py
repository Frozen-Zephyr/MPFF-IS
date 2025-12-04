import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator




# 平滑函数（TensorBoard 样式）
def tensorboard_smoothing(values, smooth):
    smoothed = []
    last = values[0]
    for val in values:
        last = last * smooth + (1 - smooth) * val
        smoothed.append(last)
    return smoothed

def png(input_path, output_path):
    '''作Tensorboard训练图'''
    # 输入输出路径
    # 读取数据
    data = pd.read_csv(input_path)

    # 准备数据
    steps = data['Step']
    raw_values = data['Value']
    smoothed_values = tensorboard_smoothing(raw_values, smooth=0.7)

    # 绘图
    fig, ax = plt.subplots()

    # 原始曲线（浅色）
    ax.plot(steps, raw_values, color="#FF9966", alpha=0.4, label="Raw Loss")

    # 平滑曲线（深色）
    ax.plot(steps, smoothed_values, color="#FF6600", label="Smoothed Loss (0.7)")

    # 添加网格
    ax.grid(True, linestyle='--', alpha=0.6)



    # 设置坐标轴和标题
    ax.set_xlabel("Epoch of Training")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Accuracy of per Training Epoch")

    # 设置 x 轴刻度更密
    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(rotation=0)
    # 图例
    ax.legend(loc='lower right')

    # 显示 + 保存为 PNG
    plt.tight_layout()
    plt.show()
    fig.savefig(output_path, dpi=1000)



input_path = "/Users/zephyr/Documents/Bioengineering Project/25大创KRASG12D抑制剂筛选预测/result/训练结果文件/泛数据集训练/acc.csv"
output_path = "/Users/zephyr/Documents/Bioengineering Project/25大创KRASG12D抑制剂筛选预测/result/训练结果文件/泛数据集训练/acc.png"  # 输出为 PNG 图片

csv_path = '/Users/zephyr/Desktop/log/layernorm最优/testset.csv'

if __name__ == "__main__":
    code=input('code:')
    while True:
        if code == 'png':
            png(input_path, output_path)
            break



        else:
            code = input('code:')
