import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import savgol_filter
from matplotlib import rcParams

# ======== 全局样式设置 ========
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

# 定义三个XVG文件路径和对应的标签
file_paths = [
    r"D:\md2\1133\results\rmsf.xvg",
    r"D:\md2\1133\1_5079\results\rmsf.xvg",
    r"D:\md2\1133\2_5406\results\rmsf.xvg",
    r"D:\md2\1133\3_4876\results\rmsf.xvg"
]
labels = ["MRTX1133","Lig1", "Lig2", "Lig3"]
colors = ['b', 'r', 'g', 'orange']

plt.figure(figsize=(12, 6),dpi=150)

for file, label, color in zip(file_paths, labels, colors):
    # 读取数据
    time, rmsd = read_xvg(file)
    
    # 异常值检测和处理
    mean = np.mean(rmsd)
    std = np.std(rmsd)
    threshold = mean + 3 * std
    outliers = np.where(rmsd > threshold)[0]
    # 标记异常值位置
    print(f"发现 {len(outliers)} 个异常值，位置：{outliers}")
    clean_time = np.delete(time, outliers)
    clean_rmsd = np.delete(rmsd, outliers)
    
    # 平滑处理
    window_size = min(21, len(clean_rmsd)//2*2+1)
    if window_size > 3:
        smooth_rmsd = savgol_filter(clean_rmsd, window_size, 3)
    else:
        smooth_rmsd = clean_rmsd
    
    # 绘制原始数据和平滑曲线
    #plt.plot(clean_time, clean_rmsd*10, color=color, alpha=0.3, label=f"{label} (original)")
    plt.plot(clean_time, clean_rmsd*10, color=color, linewidth=2, label=f"{label} ")
    #plt.plot(clean_time, smooth_rmsd*10, color=color, linewidth=2, label=f"{label} (smoothed)")
    #plt.scatter(time[outliers], rmsd[outliers], c=color, s=50, label=f"outlier removal>{threshold:.2f} Å")
    # 绘制平均值线
    #plt.axhline(mean*10, color=color, linestyle=':', alpha=0.5)

# 添加图表元素
plt.title("Comparison of RMSF Profiles", fontsize=14)
plt.xlabel("Residue", fontsize=12)
plt.ylabel("RMSF ($\AA$)", fontsize=12)
#plt.ylabel("SASA (nm)", fontsize=12)
# 移动图例到图内右上角
plt.legend(fontsize=10, loc='upper right', bbox_to_anchor=(0.95, 0.95), frameon=True)
plt.grid(True, alpha=0.3)
plt.tight_layout()

all_times = []
for file in file_paths:
    t, _ = read_xvg(file)
    all_times.extend(t)
plt.xlim(min(all_times), max(all_times))
# ======== 保存高分辨率图片 ========

plt.show()