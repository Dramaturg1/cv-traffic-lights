import cv2
import numpy as np
import matplotlib.pyplot as plt
from prepare_dataset import load_annotations

annotations = load_annotations('./data/tune/labels.yaml')


def plot_histograms(img, box, filename):
    y_min = int(box['y_min'])
    y_max = int(box['y_max'])
    x_min = int(box['x_min'])
    x_max = int(box['x_max'])

    roi = img[y_min:y_max, x_min:x_max]

    hsv_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    v_full = hsv_full[:, :, 2]
    v_roi = hsv_roi[:, :, 2]

    hist_full = cv2.calcHist([v_full], [0], None, [256], [0, 256])
    hist_roi = cv2.calcHist([v_roi], [0], None, [256], [0, 256])

    hist_full = hist_full / hist_full.sum()
    hist_roi = hist_roi / hist_roi.sum()

    plt.figure(figsize=(10, 5))
    plt.plot(hist_full, color='blue', label='Полный кадр')
    plt.plot(hist_roi, color='red', label='ROI (светофор)')
    plt.xlabel('Яроксть (V)')
    plt.ylabel('Нормированная частота')
    plt.title(f'Гистограмма: {filename} (label={box["label"]})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'histogram_{filename}', dpi=100)
    plt.show()


filename = list(annotations.keys())[2]
img = cv2.imread(f'./data/tune/images/{filename}')

for box in annotations[filename]:
    if box['occluded'] == False:
        plot_histograms(img, box, filename)
        break