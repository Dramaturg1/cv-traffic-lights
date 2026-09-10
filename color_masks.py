from prepare_dataset import load_annotations
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

annotations = load_annotations("./data/tune/labels.yaml")

def make_ROI():
    types = ['tune', 'validation', 'test']
    for type in types:
        os.makedirs(f'./data/{type}/boxes', exist_ok=True)
        ann = load_annotations(f'./data/{type}/labels.yaml')
        for key in ann.keys():
            img = cv2.imread(f'./data/{type}/images/{key}')
            if img is None:
                print('нема')
                continue
            for label in ann[key]:
                if label['occluded'] == False:
                    y_min = int(label['y_min'])
                    y_max = int(label['y_max'])
                    x_min = int(label['x_min'])
                    x_max = int(label['x_max'])
                    cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 1)
            cv2.imwrite(f'./data/{type}/boxes/{key}', img)
    return

def process_masks(img, box):
    h, w = img.shape[:2]
    y_min = int(box['y_min'])
    y_max = int(box['y_max'])
    x_min = int(box['x_min'])
    x_max = int(box['x_max'])
    y_min = max(0, y_min)
    y_max = min(h, y_max)
    x_min = max(0, x_min)
    x_max = min(w, x_max)
    roi = img[y_min:y_max, x_min:x_max]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    red_mask_1 = cv2.inRange(hsv, np.array([0, 50, 50]), np.array([15, 255, 255]))
    red_mask_2 = cv2.inRange(hsv, np.array([160, 50, 50]), np.array([180, 255, 255]))
    v_channel = hsv[:, :, 2]
    red_mask = cv2.bitwise_or(red_mask_1, red_mask_2)
    green_mask = cv2.inRange(hsv, np.array([40, 50, 50]), np.array([90, 255, 255]))
    yellow_mask = cv2.inRange(hsv, np.array([19, 50, 50]), np.array([35, 255, 255]))
    _, bright_red = cv2.threshold(v_channel, 30, 255, cv2.THRESH_BINARY)
    red_mask = cv2.bitwise_and(red_mask, bright_red)
    _, bright_green = cv2.threshold(v_channel, 30, 255, cv2.THRESH_BINARY)
    green_mask = cv2.bitwise_and(green_mask, bright_green)
    _, bright_yellow = cv2.threshold(v_channel, 40, 255, cv2.THRESH_BINARY)
    yellow_mask = cv2.bitwise_and(yellow_mask, bright_yellow)
    return roi, red_mask, green_mask, yellow_mask

def show_masks(img, box):
    roi, red_mask, green_mask, yellow_mask = process_masks(img, box)
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('ROI')
    axes[0, 0].axis('off')
    axes[0, 1].imshow(red_mask, cmap='gray')
    axes[0, 1].set_title('Red mask')
    axes[0, 1].axis('off')
    axes[0, 2].imshow(green_mask, cmap='gray')
    axes[0, 2].set_title('Green mask')
    axes[0, 2].axis('off')
    axes[1, 0].imshow(yellow_mask, cmap='gray')
    axes[1, 0].set_title('Yellow mask')
    axes[1, 0].axis('off')
    plt.tight_layout()
    plt.show()
    return

def pixels_rate(red_mask, green_mask, yellow_mask):
    masks = {
        'red': red_mask,
        'green': green_mask,
        'yellow': yellow_mask
    }
    rates = {}
    for name, mask in masks.items():
        rate = round((np.sum(mask == 255) / mask.size), 4)
        rates[name] = rate
    return rates

def make_inference(rates):
    if max(rates.values()) < 0.001:
        return 'off'
    return max(rates, key=rates.get)

def process_one_ROI(img, box):
    roi, r, g, y = process_masks(img, box)
    rates = pixels_rate(r, g, y)
    prediction = make_inference(rates)
    return {
        'prediction': prediction,
        'rates': rates,
        'roi': roi,
        'masks': {'red': r, 'green': g, 'yellow': y}
    }

