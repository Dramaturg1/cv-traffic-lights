import cv2
import numpy as np
import matplotlib.pyplot as plt
from prepare_dataset import load_annotations
from color_masks import process_masks, pixels_rate, make_inference, DEFAULT_FIXED_THRESHOLD
from dataset_processing import normalize_label

OVERLAY_COLORS_BGR = {
    'red': (0, 0, 255),
    'green': (0, 255, 0),
    'yellow': (0, 255, 255),
}


def overlay_masks_on_roi(roi_bgr, red_mask, green_mask, yellow_mask, alpha=0.5):
    overlay = roi_bgr.copy().astype(np.float32)
    color_layer = np.zeros_like(overlay)

    for name, mask in (('red', red_mask), ('green', green_mask), ('yellow', yellow_mask)):
        color = OVERLAY_COLORS_BGR[name]
        color_layer[mask == 255] = color

    active = (red_mask == 255) | (green_mask == 255) | (yellow_mask == 255)
    overlay[active] = (1 - alpha) * overlay[active] + alpha * color_layer[active]
    return overlay.astype(np.uint8)


def demonstrate_mask_overlay(dataset='tune', filename=None, method='otsu',
                              fixed_threshold=DEFAULT_FIXED_THRESHOLD,
                              out_path=None):
    annotations = load_annotations(f'./data/{dataset}/labels.yaml')

    if filename is None:
        for fname, boxes in annotations.items():
            for box in boxes:
                if not box['occluded']:
                    filename, chosen_box = fname, box
                    break
            if filename:
                break
    else:
        chosen_box = next(b for b in annotations[filename] if not b['occluded'])

    img = cv2.imread(f'./data/{dataset}/images/{filename}')
    if img is None:
        raise FileNotFoundError(f'./data/{dataset}/images/{filename}')

    box = chosen_box
    x_min, x_max = int(box['x_min']), int(box['x_max'])
    y_min, y_max = int(box['y_min']), int(box['y_max'])

    roi, red_mask, green_mask, yellow_mask, used_threshold = process_masks(
        img, box, method=method, fixed_threshold=fixed_threshold)

    rates = pixels_rate(red_mask, green_mask, yellow_mask)
    prediction = make_inference(rates)
    true_label = normalize_label(box['label'])

    overlay = overlay_masks_on_roi(roi, red_mask, green_mask, yellow_mask)

    full_with_box = img.copy()
    cv2.rectangle(full_with_box, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                              gridspec_kw={'width_ratios': [2, 1, 1]})

    axes[0].imshow(cv2.cvtColor(full_with_box, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f'Полный кадр: {filename}\n(рамка — размеченная область)')
    axes[0].axis('off')

    axes[1].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[1].set_title('ROI (исходный)')
    axes[1].axis('off')

    axes[2].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[2].set_title(f'ROI с наложенной маской\n(method={method}, thr={used_threshold:.0f})')
    axes[2].axis('off')

    legend_patches = [
        plt.Rectangle((0, 0), 1, 1, color=np.array(OVERLAY_COLORS_BGR[c][::-1]) / 255, label=c)
        for c in ('red', 'green', 'yellow')
    ]
    fig.legend(handles=legend_patches, loc='lower center', ncol=3)

    fig.suptitle(f'true={true_label}  |  prediction={prediction}  |  '
                 f'rates: red={rates["red"]:.3f} green={rates["green"]:.3f} yellow={rates["yellow"]:.3f}')
    plt.tight_layout(rect=[0, 0.05, 1, 1])

    out_path = out_path or f'mask_overlay_{filename}'
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"Сохранено: {out_path}")
    print(f"Кадр: {filename}, true={true_label}, prediction={prediction}, "
          f"method={method}, threshold={used_threshold}")
    print(f"Доли активных пикселей: {rates}")

    return out_path


if __name__ == "__main__":
    demonstrate_mask_overlay(dataset='tune', filename='208772.png', method='otsu')