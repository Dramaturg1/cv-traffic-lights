import cv2
import csv
from collections import defaultdict
from prepare_dataset import load_annotations
from color_masks import process_one_ROI, DEFAULT_FIXED_THRESHOLD
from dataset_processing import normalize_label


def compare_value_thresholds(dataset='tune', fixed_threshold=DEFAULT_FIXED_THRESHOLD,
                              save_csv=True):
    annotations = load_annotations(f'./data/{dataset}/labels.yaml')
    tags = load_annotations(f'./data/{dataset}/exposure_tags.yaml')

    records = []
    for filename, boxes in annotations.items():
        img = cv2.imread(f'./data/{dataset}/images/{filename}')
        if img is None:
            continue
        exposure = tags.get(filename, 'unknown')

        for box in boxes:
            if box['occluded']:
                continue

            fixed_res = process_one_ROI(img, box, method='fixed', fixed_threshold=fixed_threshold)
            otsu_res = process_one_ROI(img, box, method='otsu')
            if fixed_res is None or otsu_res is None:
                continue

            true_label = normalize_label(box['label'])
            records.append({
                'filename': filename,
                'exposure': exposure,
                'true': true_label,
                'fixed_pred': fixed_res['prediction'],
                'fixed_threshold': fixed_res['threshold'],
                'fixed_correct': fixed_res['prediction'] == true_label,
                'otsu_pred': otsu_res['prediction'],
                'otsu_threshold': otsu_res['threshold'],
                'otsu_correct': otsu_res['prediction'] == true_label,
                'box_area': (int(box['x_max']) - int(box['x_min'])) *
                            (int(box['y_max']) - int(box['y_min'])),
            })

    if save_csv:
        out_path = f'value_thresholds_{dataset}.csv'
        with open(out_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()) if records else [])
            writer.writeheader()
            writer.writerows(records)
        print(f"Пороги по каждой области сохранены: {out_path}")

    n = len(records)
    fixed_acc = sum(r['fixed_correct'] for r in records) / n if n else 0
    otsu_acc = sum(r['otsu_correct'] for r in records) / n if n else 0
    print(f"\n[{dataset}] Fixed({fixed_threshold}) accuracy: {fixed_acc:.2%} ({n} объектов)")
    print(f"[{dataset}] Otsu accuracy: {otsu_acc:.2%} ({n} объектов)")

    by_exposure = defaultdict(lambda: {'n': 0, 'fixed': 0, 'otsu': 0, 'otsu_vals': []})
    for r in records:
        g = by_exposure[r['exposure']]
        g['n'] += 1
        g['fixed'] += r['fixed_correct']
        g['otsu'] += r['otsu_correct']
        g['otsu_vals'].append(r['otsu_threshold'])

    print("\nПо условиям освещения:")
    print(f"{'exposure':>10} {'n':>5} {'fixed_acc':>10} {'otsu_acc':>10} {'otsu_thr_mean':>14}")
    for tag, g in by_exposure.items():
        fa = g['fixed'] / g['n'] if g['n'] else 0
        oa = g['otsu'] / g['n'] if g['n'] else 0
        mean_thr = sum(g['otsu_vals']) / len(g['otsu_vals']) if g['otsu_vals'] else 0
        print(f"{tag:>10} {g['n']:>5} {fa:>10.2%} {oa:>10.2%} {mean_thr:>14.1f}")

    otsu_helps = [r for r in records if (not r['fixed_correct']) and r['otsu_correct']]
    if otsu_helps:
        ex = otsu_helps[0]
        print(f"\nПример, где Оцу помогает: {ex['filename']} "
              f"(true={ex['true']}, fixed_pred={ex['fixed_pred']} -> неверно, "
              f"otsu_pred={ex['otsu_pred']} -> верно, otsu_threshold={ex['otsu_threshold']:.1f})")
    else:
        print("\nПримеров, где Оцу помогает, в этой части не найдено.")

    otsu_fails = [r for r in records if (not r['otsu_correct']) and
                  (r['otsu_threshold'] < 20 or r['otsu_threshold'] > 235)]
    if not otsu_fails:
        otsu_fails = [r for r in records if (not r['fixed_correct']) and (not r['otsu_correct'])]
    if otsu_fails:
        ex = otsu_fails[0]
        print(f"Пример, где двухгрупповая модель не соответствует пикселям: "
              f"{ex['filename']} (true={ex['true']}, otsu_pred={ex['otsu_pred']}, "
              f"otsu_threshold={ex['otsu_threshold']:.1f}")
    else:
        print("Явного провала двухгрупповой модели в этой части не найдено.")

    return records


if __name__ == "__main__":
    compare_value_thresholds('tune')