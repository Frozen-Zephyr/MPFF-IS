import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from matplotlib import rcParams

rcParams['font.family'] = 'Times New Roman'  # 字体
rcParams['axes.unicode_minus'] = False       # 负号正常显示
rcParams['figure.dpi'] = 1000                # 默认显示分辨率
rcParams['savefig.dpi'] = 1000               # 保存分辨率
rcParams['font.size'] = 12                   # 全局字体大小
rcParams['axes.labelsize'] = 14              # 坐标轴标签字体
rcParams['axes.titlesize'] = 16              # 标题字体
rcParams['legend.fontsize'] = 12             # 图例字体

def read_xvg(file_path):
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f if not line.startswith(('#', '@'))]
    data = np.array([list(map(float, line.split())) for line in lines])
    return data[:, 0], data[:, 1]

file_paths = [
    r"D:\md2\1133\results\sasa.xvg",
    r"D:\md2\1133\1_5079\results\sasa.xvg",
    r"D:\md2\1133\2_5406\results\sasa.xvg",
    r"D:\md2\1133\3_4876\results\sasa.xvg"
]
labels = ["MRTX1133", "Lig1", "Lig2", "Lig3"]
colors = ['b', 'r', 'g', 'orange']

plt.figure(figsize=(12, 6),dpi=150)

for file, label, color in zip(file_paths, labels, colors):
    time, rmsd = read_xvg(file)

    mean = np.mean(rmsd)
    std = np.std(rmsd)
    threshold = mean + 3 * std
    outliers = np.where(rmsd > threshold)[0]
    print(f"{label} {len(outliers)}{outliers}")

    clean_time = np.delete(time, outliers)
    clean_rmsd = np.delete(rmsd, outliers)

    # window_size = min(21, len(clean_rmsd)//2*2+1)
    # if window_size > 3:
    #     smooth_rmsd = savgol_filter(clean_rmsd, window_size, 3)
    # else:
    #     smooth_rmsd = clean_rmsd

    plt.plot(clean_time, clean_rmsd, color=color, linewidth=1.2, alpha=0.85, label=label)

plt.title("Comparison of SASA Profiles", fontsize=16)
plt.xlabel("Time (ps)", fontsize=14)
plt.ylabel("Area (nm$^2$)", fontsize=14)
plt.legend(loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()

all_times = []
for file in file_paths:
    t, _ = read_xvg(file)
    all_times.extend(t)
plt.xlim(min(all_times), max(all_times))


plt.show()
