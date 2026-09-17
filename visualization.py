import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from prepare_dataset import load_annotations

DAY_PHOTO = 'data/tune/images/207468.png'


def find_small_box_frame(dataset='tune', max_area=400):
    annotations = load_annotations(f'./data/{dataset}/labels.yaml')
    for filename, boxes in annotations.items():
        for box in boxes:
            if box['occluded']:
                continue
            area = (int(box['x_max']) - int(box['x_min'])) * \
                   (int(box['y_max']) - int(box['y_min']))
            if area <= max_area:
                return filename, box
    return None, None


def describe_and_save(path, out_prefix, box=None):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f"Ошибка: не удалось загрузить {path}")
        return None

    print(f"\n=== {path} ===")
    print(f'Размер (H, W, C): {img_bgr.shape}')
    print(f'Тип данных: {img_bgr.dtype}')
    print(f'Минимальное значение: {img_bgr.min()}')
    print(f'Максимальное значение: {img_bgr.max()}')

    img_rgb_correct = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    img_wrong_as_rgb = img_bgr

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(cv2.cvtColor(img_wrong_as_rgb, cv2.COLOR_BGR2RGB) * 0 + img_wrong_as_rgb[..., ::-1] * 0
                    + img_wrong_as_rgb)  # показываем как есть, без cvtColor
    axes[0].set_title('BGR-массив показан как RGB (ошибка)')
    axes[1].imshow(img_rgb_correct)
    axes[1].set_title('Корректный RGB (cvtColor BGR2RGB)')
    axes[2].imshow(img_hsv[:, :, 0], cmap='hsv')
    axes[2].set_title('HSV: канал H')
    axes[3].imshow(img_hsv[:, :, 2], cmap='gray')
    axes[3].set_title('HSV: канал V (яркость)')

    if box is not None:
        for ax in axes[1:2]:
            x_min, x_max = int(box['x_min']), int(box['x_max'])
            y_min, y_max = int(box['y_min']), int(box['y_max'])
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                  edgecolor='lime', facecolor='none', linewidth=1.5)
            ax.add_patch(rect)

    for ax in axes:
        ax.axis('off')
    fig.suptitle(os.path.basename(path))
    plt.tight_layout()
    out_path = f'{out_prefix}_comparison.png'
    plt.savefig(out_path, dpi=100)
    plt.close(fig)
    print(f"Сохранено: {out_path}")

    return img_bgr, img_rgb_correct


def compare_compression(img_rgb, out_prefix, box=None):
    png_path = f'{out_prefix}_result.png'
    jpg_path = f'{out_prefix}_result.jpg'
    cv2.imwrite(png_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    cv2.imwrite(jpg_path, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
                [int(cv2.IMWRITE_JPEG_QUALITY), 50])

    png_size = os.path.getsize(png_path)
    jpg_size = os.path.getsize(jpg_path)
    print(f"PNG: {png_size} байт")
    print(f"JPEG (q=50): {jpg_size} байт")

    if box is None:
        return

    x_min, x_max = int(box['x_min']), int(box['x_max'])
    y_min, y_max = int(box['y_min']), int(box['y_max'])
    pad = 10
    y0, y1 = max(0, y_min - pad), y_max + pad
    x0, x1 = max(0, x_min - pad), x_max + pad

    png_reloaded = cv2.cvtColor(cv2.imread(png_path), cv2.COLOR_BGR2RGB)
    jpg_reloaded = cv2.cvtColor(cv2.imread(jpg_path), cv2.COLOR_BGR2RGB)

    crop_png = png_reloaded[y0:y1, x0:x1]
    crop_jpg = jpg_reloaded[y0:y1, x0:x1]

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(crop_png)
    axes[0].set_title(f'PNG (без потерь), {png_size} байт')
    axes[1].imshow(crop_jpg)
    axes[1].set_title(f'JPEG q=50, {jpg_size} байт')
    for ax in axes:
        ax.axis('off')
    fig.suptitle('Детализация малого светофора: PNG vs JPEG')
    plt.tight_layout()
    detail_path = f'{out_prefix}_detail_png_vs_jpg.png'
    plt.savefig(detail_path, dpi=150)
    plt.close(fig)
    print(f"Сохранено сравнение детализации: {detail_path}")


if __name__ == "__main__":
    result1 = describe_and_save(DAY_PHOTO, out_prefix='frame1_day')
    if result1:
        _, img_rgb1 = result1
        compare_compression(img_rgb1, out_prefix='frame1_day')

    small_filename, small_box = find_small_box_frame('tune')
    if small_filename:
        small_path = f'data/tune/images/{small_filename}'
        result2 = describe_and_save(small_path, out_prefix='frame2_small', box=small_box)
        if result2:
            _, img_rgb2 = result2
            compare_compression(img_rgb2, out_prefix='frame2_small', box=small_box)
    else:
        print("Не найден кадр с малым светофором (area <= 400 px^2) — "
              "нужно увеличить max_area или проверить разметку.")