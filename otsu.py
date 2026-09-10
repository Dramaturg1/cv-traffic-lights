import cv2
import numpy as np
from prepare_dataset import load_annotations
from color_masks import pixels_rate, make_inference
from dataset_processing import normalize_label


def process_masks_with_threshold(img, box, method='fixed', fixed_threshold=127):
    h, w = img.shape[:2]
    y_min = max(0, int(box['y_min']))
    y_max = min(h, int(box['y_max']))
    x_min = max(0, int(box['x_min']))
    x_max = min(w, int(box['x_max']))

    if y_min >= y_max or x_min >= x_max:
        return None, None, None, None, 0

    roi = img[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask_1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    red_mask_2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([19, 50, 50]), np.array([35, 255, 255]))

    v_channel = hsv[:, :, 2]

    if method == 'fixed':
        _, brightness_mask = cv2.threshold(v_channel, fixed_threshold, 255, cv2.THRESH_BINARY)
        otsu_value = fixed_threshold
    else:
        otsu_value, brightness_mask = cv2.threshold(v_channel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    red_mask = cv2.bitwise_and(red_mask, brightness_mask)
    green_mask = cv2.bitwise_and(green_mask, brightness_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, brightness_mask)

    return red_mask, green_mask, yellow_mask, roi, otsu_value


def evaluate_with_method(dataset, method, fixed_threshold=127):
    annotations = load_annotations(f'./data/{dataset}/labels.yaml')
    total, correct = 0, 0
    otsu_values = []

    for filename, boxes in annotations.items():
        img = cv2.imread(f'./data/{dataset}/images/{filename}')
        if img is None:
            continue

        for box in boxes:
            if box['occluded'] == True:
                continue

            red_mask, green_mask, yellow_mask, roi, otsu_val = process_masks_with_threshold(
                img, box, method=method, fixed_threshold=fixed_threshold
            )

            if red_mask is None:
                continue

            rates = pixels_rate(red_mask, green_mask, yellow_mask)
            predicted = make_inference(rates)
            true_label = normalize_label(box['label'])

            total += 1
            if predicted == true_label:
                correct += 1

            if method == 'otsu':
                otsu_values.append(otsu_val)

    accuracy = correct / total if total > 0 else 0
    return accuracy, total, otsu_values


acc_fixed, total_fixed, _ = evaluate_with_method('tune', 'fixed', fixed_threshold=127)
print(f"Фиксированный порог (127): {acc_fixed:.2%} ({int(acc_fixed * total_fixed)}/{total_fixed})")

acc_otsu, total_otsu, otsu_vals = evaluate_with_method('tune', 'otsu')
print(f"Метод Оцу: {acc_otsu:.2%} ({int(acc_otsu * total_otsu)}/{total_otsu})")

if otsu_vals:
    print(f"\nОцу статистика:")
    print(f"  Min: {min(otsu_vals):.1f}")
    print(f"  Max: {max(otsu_vals):.1f}")
    print(f"  Mean: {np.mean(otsu_vals):.1f}")
    print(f"  Median: {np.median(otsu_vals):.1f}")