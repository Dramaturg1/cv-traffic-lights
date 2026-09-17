import numpy as np
from collections import defaultdict
import cv2
from prepare_dataset import load_annotations
from dataset_processing import normalize_label
from color_masks import process_one_ROI, DEFAULT_FIXED_THRESHOLD
from config import FINAL_CONFIG, lock_config


def collect_predictions(dataset, method='fixed', fixed_threshold=DEFAULT_FIXED_THRESHOLD):
    annotations = load_annotations(f'./data/{dataset}/labels.yaml')
    results = []

    for filename, boxes in annotations.items():
        img = cv2.imread(f'./data/{dataset}/images/{filename}')
        if img is None:
            continue

        for box in boxes:
            if box['occluded'] == True:
                continue

            result = process_one_ROI(img, box, method=method, fixed_threshold=fixed_threshold)
            if result is None:
                continue

            results.append({
                'filename': filename,
                'box': box,
                'true': normalize_label(box['label']),
                'pred': result['prediction'],
                'threshold': result['threshold'],
                'rates': result['rates'],
            })

    return results


def accuracy(results):
    correct = sum(1 for r in results if r['true'] == r['pred'])
    return correct / len(results) if results else 0


def f1_per_class(results, classes=('red', 'yellow', 'green', 'off')):
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


def confusion_matrix(results, classes=('red', 'yellow', 'green', 'off')):
    matrix = {t: {p: 0 for p in classes} for t in classes}
    for r in results:
        if r['true'] in matrix and r['pred'] in matrix[r['true']]:
            matrix[r['true']][r['pred']] += 1
    return matrix


def print_confusion_matrix(matrix, classes=('red', 'yellow', 'green', 'off')):
    header = " " * 12 + "".join(f"{c:>8}" for c in classes)
    print(header)
    print("-" * len(header))
    for true in classes:
        row = f"{true:>8} |" + "".join(f"{matrix[true][pred]:>8}" for pred in classes)
        print(row)


def print_metrics_for_method(dataset, method, fixed_threshold=DEFAULT_FIXED_THRESHOLD):
    results = collect_predictions(dataset, method=method, fixed_threshold=fixed_threshold)

    acc = accuracy(results)
    f1s = f1_per_class(results)
    mf1 = macro_f1(f1s)
    fe = frame_exact(results)
    cm = confusion_matrix(results)

    print(f"\n=== METRICS: dataset={dataset}, method={method} ===")
    print(f"Total objects: {len(results)}")
    print(f"Accuracy: {acc:.2%}")
    print(f"Macro-F1: {mf1:.3f}")
    print(f"Frame-Exact: {fe:.2%}")
    print(f"\nF1 per class:")
    print(f"{'class':>10} {'precision':>10} {'recall':>10} {'f1':>10} {'support':>10}")
    for cls, m in f1s.items():
        print(f"{cls:>10} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>10.3f} {m['support']:>10}")
    print("\nConfusion matrix:")
    print_confusion_matrix(cm)

    return {
        'dataset': dataset, 'method': method,
        'accuracy': acc, 'macro_f1': mf1, 'frame_exact': fe,
        'f1_per_class': f1s, 'confusion_matrix': cm, 'results': results,
    }


if __name__ == "__main__":
    all_metrics = {}

    for dataset in ('tune', 'validation'):
        for method in ('fixed', 'otsu'):
            all_metrics[(dataset, method)] = print_metrics_for_method(
                dataset, method, fixed_threshold=FINAL_CONFIG['fixed_threshold'])

    lock_config()
    final_method = FINAL_CONFIG['brightness_method']
    all_metrics[('test', final_method)] = print_metrics_for_method(
        'test', final_method, fixed_threshold=FINAL_CONFIG['fixed_threshold'])