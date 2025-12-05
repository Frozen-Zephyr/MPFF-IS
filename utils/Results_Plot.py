import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def data_png(file,out):
    '''样本数据分布图'''
    df = pd.read_csv(file)

    # 统计逻辑
    positive_data = []
    negative_data = []

    i = 0
    while i < len(df):
        row = df.iloc[i]
        if row['Label'] == 1:
            std_val = row['Standard Value']
            positive_data.append(std_val)

            neg_count = 0
            i += 1
            while i < len(df) and df.iloc[i]['Label'] == 0:
                neg_count += 1
                i += 1
            negative_data.append((std_val, neg_count))
        else:
            i += 1

    # 转 DataFrame
    pos_df = pd.DataFrame({'Standard Value': positive_data})
    neg_df = pd.DataFrame(negative_data, columns=['Standard Value', 'Neg Count'])

    # 自定义区间
    min_val = min(pos_df['Standard Value'].min(), neg_df['Standard Value'].min())
    max_val = max(pos_df['Standard Value'].max(), neg_df['Standard Value'].max())
    bins = np.linspace(min_val, max_val, 21)  # 分成 20 个小区间

    # 分箱
    pos_df['bin'] = pd.cut(pos_df['Standard Value'], bins=bins)
    neg_df['bin'] = pd.cut(neg_df['Standard Value'], bins=bins)

    # 聚合
    merged_df = pos_df.groupby('bin').size().reset_index(name='Positive Count')
    neg_df_grouped = neg_df.groupby('bin')['Neg Count'].sum().reset_index(name='Negative Count')

    # 合并
    merged_df = pd.merge(merged_df, neg_df_grouped, on='bin', how='left').fillna(0)

    # 取对数
    merged_df['Positive Count (log)'] = np.log10(merged_df['Positive Count'].replace(0, np.nan))
    merged_df['Negative Count (log)'] = np.log10(merged_df['Negative Count'].replace(0, np.nan))

    plt.figure(figsize=(12, 6))
    x = np.arange(len(merged_df))  # x轴位置
    bar_width = 0.4  # 每个柱子的宽度

    plt.bar(x - bar_width / 2, merged_df['Positive Count (log)'], width=bar_width, label='Positive Samples (log)',
            color='skyblue')
    plt.bar(x + bar_width / 2, merged_df['Negative Count (log)'], width=bar_width, label='Negative Samples (log)',
            color='lightcoral')

    # X 轴标签
    x_labels = [f'{interval.left:.2f}–{interval.right:.2f}' for interval in merged_df['bin']]
    plt.xticks(x, x_labels, rotation=45)

    # 图形修饰
    plt.xlabel('IC50 Value (nM)')
    plt.ylim(0, 5.5)
    plt.ylabel('Lg(Sample Count)')
    plt.title('Sample Count by Standard Value (Log Scale)')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.savefig(out,dpi=800)
    plt.close()


def result_data_png(file,out):
    '''结果分布lg柱状图'''
    df = pd.read_csv(file)

    # 设定分区间，0-1 区间分为 20 个
    bins = np.linspace(0, 1, 21)

    # 按照 out 列的值将数据分箱
    df['bin'] = pd.cut(df['out'], bins=bins, include_lowest=True)

    # 统计每个区间的正负样本数量
    positive_data = df[df['Label'] == 1]
    negative_data = df[df['Label'] == 0]

    positive_counts = positive_data.groupby('bin').size()
    negative_counts = negative_data.groupby('bin').size()

    # 合并正负样本计数
    merged_counts = pd.DataFrame({
        'Positive Count': positive_counts,
        'Negative Count': negative_counts
    }).fillna(0)

    merged_counts['Positive Count (log)'] = np.log10(merged_counts['Positive Count'].replace(0, np.nan))
    merged_counts['Negative Count (log)'] = np.log10(merged_counts['Negative Count'].replace(0, np.nan))

    # 绘制并列柱状图
    plt.figure(figsize=(12, 6))

    # x 轴位置
    x = np.arange(len(merged_counts))
    bar_width = 0.35

    plt.bar(x - bar_width / 2, merged_counts['Positive Count (log)'], width=bar_width, label='Positive Samples (log)',
            color='skyblue')
    plt.bar(x + bar_width / 2, merged_counts['Negative Count (log)'], width=bar_width, label='Negative Samples (log)',
            color='lightcoral')

    x_labels = [f'{interval.left:.2f}–{interval.right:.2f}' for interval in merged_counts.index]
    plt.xticks(x, x_labels, rotation=45)

    # 修饰图表
    plt.xlabel('Output Value')
    plt.ylabel('Lg(Sample Count)')
    plt.title('Sample Count by Output Value (Log Scale)')
    plt.legend(loc='upper right')

    # 设定 y 轴范围，并添加折线
    plt.ylim(0, 4)  # y轴上限设为5
    yticks = plt.gca().get_yticks()
    yticks = [tick for tick in yticks if tick < 4]  # 显示小于5的y值
    plt.gca().set_yticks(yticks)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(out, dpi=800)


def dot(file,out):
    df = pd.read_csv(file)

    # 只取正样本（Label == 1）
    positive_df = df[df['Label'] == 1]

    # 提取x和y
    x = positive_df['out']
    y = positive_df['Standard Value']

    # 绘制散点图
    plt.figure(figsize=(8, 6))
    plt.ylim(-0.5,7)
    plt.scatter(x, y, c='skyblue', alpha=0.85,  label='Positive Samples',linewidths=0 )

    # 图形修饰
    plt.xlabel('Output Value')
    plt.ylabel('IC50 (nM)')
    plt.title('Scatter Plot of Positive Samples')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.legend(loc='upper right')

    # 保存或展示
    plt.savefig(out, dpi=800)
    plt.show()


def violin(file,out):
    df = pd.read_csv(file)

    # 画小提琴图
    plt.figure(figsize=(8, 6))
    ax=sns.violinplot(x='Label', y='out', data=df, palette='Set2', cut=0,  inner=None,   )

    ax.collections[0].set_facecolor('lightcoral')
    ax.collections[1].set_facecolor('skyblue')

    # 修饰
    plt.xlabel('Label')
    plt.ylabel('out value')
    plt.title('Violin plot of out values by Label')
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # 保存或展示
    plt.tight_layout()
    plt.savefig(out, dpi=800)
    plt.show()


def tensorboard_smoothing(values, smooth):
    smoothed = []
    last = values[0]
    for val in values:
        last = last * smooth + (1 - smooth) * val
        smoothed.append(last)
    return smoothed


def training_png(input_path, output_path):
    '''作Tensorboard训练图'''
    data = pd.read_csv(input_path)

    steps = data['Step']
    raw_values = data['Value']
    smoothed_values = tensorboard_smoothing(raw_values, smooth=0)

    fig, ax = plt.subplots()
    ax.plot(steps, raw_values, color="#FF9966", alpha=0.4, label="Raw Loss")
    ax.plot(steps, smoothed_values, color="#FF6600", label="Smoothed Loss (0)")
    ax.grid(True, linestyle='--', alpha=0.6)
    ax.set_xlabel("Epochs of Training")
    ax.set_ylabel("Epoch Loss")
    ax.set_title("Loss per Training epoch")
    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.legend(loc='upper right')

    plt.tight_layout()
    plt.show()
    fig.savefig(output_path, dpi=500)


def KDE(file,out):
    '''结果分布核密度图（log10纵轴，适应密度极度集中的情况）'''

    df = pd.read_csv(file)

    plt.figure(figsize=(12, 6))

    # 绘制核密度曲线
    sns.kdeplot(
        data=df[df['Label'] == 1],
        x='out',
        color='skyblue',
        linewidth=2,
        label='Positive Samples'
    )
    sns.kdeplot(
        data=df[df['Label'] == 0],
        x='out',
        color='lightcoral',
        linewidth=2,
        label='Negative Samples'
    )

    # 取log10纵轴
    ax = plt.gca()
    y_values = ax.get_lines()[0].get_ydata()
    y_values2 = ax.get_lines()[1].get_ydata()
    ax.set_yscale('log')

    # 修饰图表外观
    plt.xlabel('Output Value')
    plt.ylabel('Density (log10 scale)')
    plt.title('Prediction Distribution by Label (Log-scaled KDE)')
    plt.legend(loc='upper right')
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # 坐标范围
    plt.xlim(0, 1)
    plt.tight_layout()

    # 保存高分辨率图像
    plt.savefig(out, dpi=800)
    plt.close()


def step(file_path,out):     #不同阈值下的正确率

    out_min, out_max = 0.98, 1.0
    step = 0.001


    df = pd.read_csv(file_path)
    df = df[(df['out'] >= out_min) & (df['out'] <= out_max)]


    outs = np.arange(out_min, out_max + step, step)
    accs = []

    for threshold in outs:
        subset = df[df['out'] >= threshold]
        if len(subset) == 0:
            accs.append(np.nan)
            continue
        correct = (subset['Label'] == subset['pre_Label']).sum()
        acc = correct / len(subset)
        accs.append(acc)


    plt.figure(figsize=(8, 5))
    plt.plot(outs, accs, '-', lw=1.5, color='steelblue')

    plt.xlabel("out (prediction value)")
    plt.ylabel("Accuracy (out ≥ x)")
    plt.title("Accuracy vs Prediction Confidence (0.999–1.0)")

    # 拉伸纵轴，增加视觉区分
    plt.ylim(0.995, 1.001)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(out, dpi=800)
    plt.show()


if __name__ == '__main__':
    path = ''
    out = ''
    while True:
        code = input('mission:')

        if code == 'data_png':
            path=''
            out=''
            data_png(path,out)
            break

        elif code == 'result_png':
            path=''
            out=''
            result_data_png(path,out)
            break

        elif  code == 'dot':
            path=''
            out=''
            dot(path,out)
            break

        elif code == 'violin':
            path = ''
            out = ''
            violin(path,out)
            break

        elif code == 'training_png':
            input_path = ""
            output_path = ""
            training_png(input_path, output_path)
            break

        elif code == 'kde':
            path=''
            out=''
            KDE(path,out)
            break

        elif code == 'step':
            step(path,out)
            break

        else:
            continue