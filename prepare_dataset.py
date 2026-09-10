import yaml
import os
import random
import shutil
from collections import defaultdict

TRAIN = 'dataset_train_rgb/rgb/train'
TEST = 'dataset_test_rgb/rgb/test'
SEED = 20260827

def load_annotations(yaml_path):
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    return data

def gather_paths(annotations):
    sequences = defaultdict(list)
    for annotation in annotations:
        if len(annotation['boxes']) != 0:
            parts = annotation['path'].split('/')
            seq_name = parts[-2]
            sequences[seq_name].append(parts[-1])
    return dict(sequences)

def check_intersections(dict1, dict2):
    files1 = set()
    for files_list in dict1.values():
        files1.update(files_list)
    files2 = set()
    for files_list in dict2.values():
        files2.update(files_list)
    intersections = files1.intersection(files2)
    print(f'Найдено {len(intersections)} пересечений')
    if len(intersections) == 0:
        return dict1, dict2
    print('Удаление пересечений...')
    new_dict1 = {}
    for key, files_list in dict1.items():
        filtered_files = [f for f in files_list if f not in intersections]
        if filtered_files:
            new_dict1[key] = filtered_files
    new_dict2 = {}
    for key, files_list in dict2.items():
        filtered_files = [f for f in files_list if f not in intersections]
        if filtered_files:
            new_dict2[key] = filtered_files
    return new_dict1, new_dict2

def stats(dict):
    for key, value in dict.items():
        print(f'{key}: {len(value)}')
    return


def make_subsets(train, test, output_root='data'):
    os.makedirs(os.path.join(output_root, 'tune', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'validation', 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_root, 'test', 'images'), exist_ok=True)

    random.seed(SEED)

    train_keys = list(train.keys())
    random.shuffle(train_keys)
    tune_files = []
    tune_count = 0
    tune_seq_count = 0

    for seq in train_keys:
        if tune_count < 120:
            files = train[seq]
            for filename in files:
                if tune_count >= 120:
                    break
                src = os.path.join(TRAIN, seq, filename)
                dst = os.path.join(output_root, 'tune', 'images', filename)
                shutil.copy2(src, dst)
                tune_files.append(filename)
                tune_count += 1
            tune_seq_count += 1
        else:
            break

    remaining_keys = train_keys[tune_seq_count:]
    val_files = []
    val_count = 0

    for seq in remaining_keys:
        if val_count < 48:
            files = train[seq]
            for filename in files:
                if val_count >= 48:
                    break
                src = os.path.join(TRAIN, seq, filename)
                dst = os.path.join(output_root, 'validation', 'images', filename)
                shutil.copy2(src, dst)
                val_files.append(filename)
                val_count += 1
        else:
            break

    test_keys = list(test.keys())
    random.shuffle(test_keys)
    test_files = []
    test_count = 0

    for seq in test_keys:
        if test_count < 72:
            files = test[seq]
            for filename in files[:72]:
                src = os.path.join(TEST, filename)
                dst = os.path.join(output_root, 'test', 'images', filename)
                shutil.copy2(src, dst)
                test_files.append(filename)
                test_count += 1
        else:
            break

    tune_ann = {}
    for filename in tune_files:
        for ann in train_annotations:
            if os.path.basename(ann['path']) == filename:
                boxes = []
                for box in ann['boxes']:
                    boxes.append({
                        'label': box['label'],
                        'x_min': box['x_min'],
                        'x_max': box['x_max'],
                        'y_min': box['y_min'],
                        'y_max': box['y_max'],
                        'occluded': box['occluded']
                    })
                tune_ann[filename] = boxes
                break
    with open(os.path.join(output_root, 'tune', 'labels.yaml'), 'w') as f:
        yaml.dump(tune_ann, f)

    val_ann = {}
    for filename in val_files:
        for ann in train_annotations:
            if os.path.basename(ann['path']) == filename:
                boxes = []
                for box in ann['boxes']:
                    boxes.append({
                        'label': box['label'],
                        'x_min': box['x_min'],
                        'x_max': box['x_max'],
                        'y_min': box['y_min'],
                        'y_max': box['y_max'],
                        'occluded': box['occluded']
                    })
                val_ann[filename] = boxes
                break
    with open(os.path.join(output_root, 'validation', 'labels.yaml'), 'w') as f:
        yaml.dump(val_ann, f)

    test_ann = {}
    for filename in test_files:
        for ann in test_annotations:
            if os.path.basename(ann['path']) == filename:
                boxes = []
                for box in ann['boxes']:
                    boxes.append({
                        'label': box['label'],
                        'x_min': box['x_min'],
                        'x_max': box['x_max'],
                        'y_min': box['y_min'],
                        'y_max': box['y_max'],
                        'occluded': box['occluded']
                    })
                test_ann[filename] = boxes
                break
    with open(os.path.join(output_root, 'test', 'labels.yaml'), 'w') as f:
        yaml.dump(test_ann, f)

    print(f"Tune: {tune_count} кадров")
    print(f"Validation: {val_count} кадров")
    print(f"Test: {test_count} кадров")

    return {
        'tune': tune_files,
        'validation': val_files,
        'test': test_files
    }

if __name__ == "__main__":
    train_annotations = load_annotations('dataset_train_rgb/train.yaml')
    test_annotations = load_annotations('dataset_test_rgb/test.yaml')

    sequences_train = gather_paths(train_annotations)
    sequences_test = gather_paths(test_annotations)

    new_train, new_test = check_intersections(sequences_train, sequences_test)
    make_subsets(new_train, new_test)

