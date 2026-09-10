import cv2
import numpy as np
from collections import defaultdict
from prepare_dataset import load_annotations
from dataset_processing import normalize_label
from color_masks import make_inference


def process_one_ROI(img, box, fixed_threshold=127):
    h, w = img.shape[:2]
    y_min = max(0, int(box['y_min']))
    y_max = min(h, int(box['y_max']))
    x_min = max(0, int(box['x_min']))
    x_max = min(w, int(box['x_max']))

    if y_min >= y_max or x_min >= x_max:
        return None

    roi = img[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_mask_1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    red_mask_2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([19, 50, 50]), np.array([35, 255, 255]))

    v_channel = hsv[:, :, 2]
    _, brightness_mask = cv2.threshold(v_channel, fixed_threshold, 255, cv2.THRESH_BINARY)

    red_mask = cv2.bitwise_and(red_mask, brightness_mask)
    green_mask = cv2.bitwise_and(green_mask, brightness_mask)
    yellow_mask = cv2.bitwise_and(yellow_mask, brightness_mask)

    rates = {
        'red': np.sum(red_mask == 255) / red_mask.size,
        'green': np.sum(green_mask == 255) / green_mask.size,
        'yellow': np.sum(yellow_mask == 255) / yellow_mask.size
    }

    return {
        'prediction': make_inference(rates),
        'rates': rates
    }


def collect_predictions(dataset, fixed_threshold=127):
    annotations = load_annotations(f'./data/{dataset}/labels.yaml')
    results = []

    for filename, boxes in annotations.items():
        img = cv2.imread(f'./data/{dataset}/images/{filename}')
        if img is None:
            continue

        for box in boxes:
            if box['occluded'] == True:
                continue

            result = process_one_ROI(img, box, fixed_threshold)
            if result is None:
                continue

            results.append({
                'filename': filename,
                'true': normalize_label(box['label']),
                'pred': result['prediction']
            })

    return results


def accuracy(results):
    correct = sum(1 for r in results if r['true'] == r['pred'])
    return correct / len(results) if results else 0


def f1_per_class(results, classes=['red', 'yellow', 'green', 'off']):
    f1_scores = {}

    for cls in classes:
        tp = sum(1 for r in results if r['true'] == cls and r['pred'] == cls)
        fp = sum(1 for r in results if r['true'] != cls and r['pred'] == cls)
        fn = sum(1 for r in results if r['true'] == cls and r['pred'] != cls)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        f1_scores[cls] = {
            'precision': round(precision, 3),
            'recall': round(recall, 3),
            'f1': round(f1, 3),
            'support': tp + fn
        }

    return f1_scores


def macro_f1(f1_scores):
    return sum(v['f1'] for v in f1_scores.values()) / len(f1_scores)


def frame_exact(results):
    by_file = defaultdict(list)
    for r in results:
        by_file[r['filename']].append(r)

    exact = 0
    for filename, items in by_file.items():
        if all(r['true'] == r['pred'] for r in items):
            exact += 1

    return exact / len(by_file) if by_file else 0


def print_metrics(dataset='tune', fixed_threshold=127):
    results = collect_predictions(dataset, fixed_threshold)

    acc = accuracy(results)
    f1s = f1_per_class(results)
    mf1 = macro_f1(f1s)
    fe = frame_exact(results)

    print(f"METRICS: {dataset}")
    print(f"Total objects: {len(results)}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Macro-F1: {mf1:.3f}")
    print(f"Frame-Exact: {fe:.2%}")

    print(f"\nF1 per class:")
    print(f"{'class':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for cls, m in f1s.items():
        print(f"{cls:>10} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10}")

    return {
        'accuracy': acc,
        'macro_f1': mf1,
        'frame_exact': fe,
        'f1_per_class': f1s,
        'results': results
    }


metrics = print_metrics('tune', fixed_threshold=127)
metrics = print_metrics('validation', fixed_threshold=127)
metrics = print_metrics('test', fixed_threshold=127)