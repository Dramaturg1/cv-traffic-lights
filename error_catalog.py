import cv2
import matplotlib.pyplot as plt
from prepare_dataset import load_annotations
from dataset_processing import normalize_label
from color_masks import process_one_ROI, DEFAULT_FIXED_THRESHOLD


def build_error_catalog(dataset='validation', method='fixed',
                         fixed_threshold=DEFAULT_FIXED_THRESHOLD, min_examples=4):
    annotations = load_annotations(f'./data/{dataset}/labels.yaml')
    errors = []

    for filename, boxes in annotations.items():
        img = cv2.imread(f'./data/{dataset}/images/{filename}')
        if img is None:
            continue
        for box in boxes:
            if box['occluded']:
                continue
            result = process_one_ROI(img, box, method=method, fixed_threshold=fixed_threshold)
            if result is None:
                continue
            true_label = normalize_label(box['label'])
            if result['prediction'] != true_label:
                errors.append({
                    'filename': filename,
                    'box': box,
                    'width': int(box['x_max']) - int(box['x_min']),
                    'height': int(box['y_max']) - int(box['y_min']),
                    'true': true_label,
                    'pred': result['prediction'],
                    'rates': result['rates'],
                    'threshold': result['threshold'],
                    'masks': result['masks'],
                    'roi': result['roi'],
                })

    seen_pairs = set()
    diverse = []
    for e in errors:
        pair = (e['true'], e['pred'])
        if pair not in seen_pairs:
            diverse.append(e)
            seen_pairs.add(pair)
    if len(diverse) < min_examples:
        for e in errors:
            if e not in diverse:
                diverse.append(e)
            if len(diverse) >= min_examples:
                break

    diverse = diverse[:max(min_examples, len(diverse))]

    print(f"\n=== КАТАЛОГ ОШИБОК ({dataset}, method={method}) ===")
    print(f"Всего ошибок: {len(errors)}, показано разборов: {len(diverse)}\n")

    for i, e in enumerate(diverse, 1):
        print(f"--- Ошибка {i} ---")
        print(f"Кадр: {e['filename']}")
        print(f"Размер области: {e['width']}x{e['height']} px")
        print(f"Истинная метка: {e['true']}  |  Прогноз: {e['pred']}")
        print(f"Порог яркости, использованный при классификации: {e['threshold']}")
        print(f"Доли активных пикселей по маскам: "
              f"red={e['rates']['red']:.4f}, green={e['rates']['green']:.4f}, "
              f"yellow={e['rates']['yellow']:.4f}")

        # сохраняем иллюстрацию: ROI + три маски, подписанные
        fig, axes = plt.subplots(1, 4, figsize=(12, 3))
        axes[0].imshow(cv2.cvtColor(e['roi'], cv2.COLOR_BGR2RGB))
        axes[0].set_title('ROI')
        for ax, name in zip(axes[1:], ('red', 'green', 'yellow')):
            ax.imshow(e['masks'][name], cmap='gray')
            ax.set_title(f'{name} mask')
        for ax in axes:
            ax.axis('off')
        fig.suptitle(f"{e['filename']}: true={e['true']} pred={e['pred']} thr={e['threshold']}")
        plt.tight_layout()
        out_path = f"error_{i}_{e['filename']}"
        plt.savefig(out_path, dpi=100)
        plt.close(fig)
        print(f"Иллюстрация сохранена: {out_path}")

        # Проверяемая гипотеза, специфичная для типа ошибки
        if e['true'] == 'off' and e['pred'] != 'off':
            hint = ("Возможно, порог яркости слишком низкий для условий этого "
                    "кадра, и фоновый блик проходит долю > min_active_rate. "
                    "Проверить: поднять min_active_rate или порог для этого "
                    "диапазона освещения и пересчитать ТОЛЬКО на новой части данных.")
        elif e['pred'] == 'off' and e['true'] != 'off':
            hint = ("Возможно, лампа слишком мала или недостаточно яркая "
                    "относительно порога (маленькая область — доля пикселей "
                    "лампы тонет в min_active_rate). Проверить: понизить "
                    "min_active_rate или использовать адаптивный порог по "
                    "площади области.")
        else:
            hint = ("Возможно, цветовые границы соседних классов (напр. "
                    "жёлтый/красный) пересекаются при данном освещении. "
                    "Проверить: сузить/сдвинуть границу тона между "
                    "конфликтующими классами и повторно оценить на validation.")
        print(f"Проверяемая гипотеза для следующего эксперимента: {hint}\n")

    return diverse


if __name__ == "__main__":
    build_error_catalog('validation', method='fixed', min_examples=4)