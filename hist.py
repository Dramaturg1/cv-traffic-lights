import cv2
import numpy as np
import matplotlib.pyplot as plt
from prepare_dataset import load_annotations

EXPOSURE_LABELS_RU = {
    'normal': 'дневной (обычная экспозиция)',
    'low': 'тёмный (низкая экспозиция)',
    'backlit': 'контровой свет',
}


def _first_frame_with_tag(annotations, tags, tag):
    for filename, boxes in annotations.items():
        if tags.get(filename) != tag:
            continue
        for box in boxes:
            if not box['occluded']:
                return filename, box
    return None, None


def _hist_v(values):
    hist = cv2.calcHist([values], [0], None, [256], [0, 256])
    hist = hist / hist.sum()
    return hist


def normalized_histograms(dataset='tune', images_dir=None, save_prefix='histogram'):
    images_dir = images_dir or f'./data/{dataset}/images'
    annotations = load_annotations(f'./data/{dataset}/labels.yaml')
    tags = load_annotations(f'./data/{dataset}/exposure_tags.yaml')

    picks = {}
    for tag in ('normal', 'low', 'backlit'):
        filename, box = _first_frame_with_tag(annotations, tags, tag)
        if filename is None:
            print(f"[normalized_histograms] Кадров с тегом '{tag}' в '{dataset}' не найдено ")
            continue
        picks[tag] = (filename, box)

    fig, axes = plt.subplots(len(picks), 1, figsize=(9, 4 * max(len(picks), 1)), squeeze=False)

    for i, (tag, (filename, box)) in enumerate(picks.items()):
        img = cv2.imread(f'{images_dir}/{filename}')
        y_min, y_max = int(box['y_min']), int(box['y_max'])
        x_min, x_max = int(box['x_min']), int(box['x_max'])
        roi = img[y_min:y_max, x_min:x_max]

        v_full = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:, :, 2]
        v_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)[:, :, 2]

        hist_full = _hist_v(v_full)
        hist_roi = _hist_v(v_roi)

        ax = axes[i, 0]
        ax.plot(hist_full, color='blue', label='Полный кадр')
        ax.plot(hist_roi, color='red', label='ROI (светофор)')
        ax.set_xlim(0, 255)
        ax.set_ylim(0, 0.25)
        ax.set_xlabel('Яркость (V)')
        ax.set_ylabel('Нормированная частота')
        ax.set_title(f'{EXPOSURE_LABELS_RU[tag]}: {filename} (label={box["label"]})')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    out_path = f'{save_prefix}_{dataset}.png'
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Сохранено: {out_path}")

    return picks


if __name__ == "__main__":
    normalized_histograms('tune')