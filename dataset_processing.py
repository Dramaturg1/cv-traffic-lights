from color_masks import *
from collections import Counter
from prepare_dataset import load_annotations

def normalize_label(label):
    label = label.lower()
    if 'red' in label:
        return 'red'
    elif 'green' in label:
        return 'green'
    elif 'yellow' in label:
        return 'yellow'
    elif label == 'off':
        return 'off'
    else:
        return label

def gather_metrics(type):
    annotations = load_annotations(f'./data/{type}/labels.yaml')
    total, correct = 0, 0
    errors = []
    for key in annotations.keys():
        img = cv2.imread(f'./data/{type}/images/{key}')
        if img is None:
            print('нема')
            continue
        for box in annotations[key]:
            if box['occluded'] == True:
                continue
            result = process_one_ROI(img, box)
            true_label = normalize_label(box['label'])
            predicted = result['prediction']

            total += 1
            if predicted == true_label:
                    correct += 1
            else:
                errors.append((key, true_label, predicted, result['rates']))

    accuracy = correct / total if total > 0 else 0
    print(f'Правильно: {correct} / {total} = {accuracy:.2%}')

    return errors, accuracy


def confusion_matrix(errors):
    classes = ['red', 'yellow', 'green', 'off']
    matrix = {true: {pred: 0 for pred in classes} for true in classes}

    for _, true_label, predicted, _ in errors:
        if true_label in matrix and predicted in matrix[true_label]:
            matrix[true_label][predicted] += 1

    return matrix


def print_confusion_matrix(matrix):
    classes = ['red', 'yellow', 'green', 'off']

    print("\n" + "=" * 50)
    print("CONFUSION MATRIX")
    print("=" * 50)

    header = " " * 12
    for c in classes:
        header += f"{c:>8}"
    print(header)
    print("-" * 50)

    for true in classes:
        row = f"{true:>8} |"
        for pred in classes:
            row += f"{matrix[true][pred]:>8}"
        print(row)

    print("=" * 50)


def print_error_summary(errors):
    from collections import Counter
    error_types = Counter()
    for _, true_label, predicted, _ in errors:
        if true_label != predicted:
            error_types[(true_label, predicted)] += 1

    print("\n" + "=" * 50)
    print("MOST COMMON ERRORS")
    print("=" * 50)
    for (true, pred), count in error_types.most_common(10):
        print(f"{true} -> {pred}: {count}")

if __name__ == "__main__":
    print('tune')
    errors, accuracy = gather_metrics('tune')
    print('validation')
    errors, accuracy = gather_metrics('validation')
    print('test')
    errors, accuracy = gather_metrics('test')
