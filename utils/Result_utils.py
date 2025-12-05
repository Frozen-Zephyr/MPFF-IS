import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator





def tensorboard_smoothing(values, smooth):
    smoothed = []
    last = values[0]
    for val in values:
        last = last * smooth + (1 - smooth) * val
        smoothed.append(last)
    return smoothed

def png(input_path, output_path):
    '''作Tensorboard训练图'''
    data = pd.read_csv(input_path)

    steps = data['Step']
    raw_values = data['Value']
    smoothed_values = tensorboard_smoothing(raw_values, smooth=0.7)

    fig, ax = plt.subplots()
    ax.plot(steps, raw_values, color="#FF9966", alpha=0.4, label="Raw Loss")
    ax.plot(steps, smoothed_values, color="#FF6600", label="Smoothed Loss (0.7)")
    ax.grid(True, linestyle='--', alpha=0.6)

    ax.set_xlabel("Epoch of Training")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Accuracy of per Training Epoch")

    ax.xaxis.set_major_locator(MultipleLocator(5))
    plt.xticks(rotation=0)

    ax.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
    fig.savefig(output_path, dpi=1000)



input_path = ""
output_path = ""
csv_path = ''

if __name__ == "__main__":
    code=input('code:')
    while True:
        if code == 'png':
            png(input_path, output_path)
            break

        else:
            code = input('code:')
