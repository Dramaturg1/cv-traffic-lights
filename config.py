import hashlib
import json
import os

FINAL_CONFIG = {
    "color_ranges": {
        "red_1": [[0, 50, 50], [15, 255, 255]],
        "red_2": [[160, 50, 50], [180, 255, 255]],
        "green": [[40, 50, 50], [90, 255, 255]],
        "yellow": [[19, 50, 50], [35, 255, 255]],
    },
    "brightness_method": "otsu",
    "fixed_threshold": 127,
    "min_active_rate": 0.001,
    "seed": 20260827,
    "explanation": (
        "Метод Оцу даёт похожую точность на tune, но на кадрах с низкой "
        "экспозицией и контровым светом плывёт (порог уходит к краям "
        "диапазона на почти однородных областях). Фиксированный порог 127 "
        "выбран как более предсказуемая схема для итогового запуска; "
        "конкретные случаи расхождения приведены в отчёте по разбору ошибок."
    ),
}

LOCK_FILE = "config.lock"


def config_hash(config=None):
    config = config if config is not None else FINAL_CONFIG
    serialized = json.dumps(config, sort_keys=True, ensure_ascii=False).encode('utf-8')
    return hashlib.sha256(serialized).hexdigest()


def lock_config(config=None):
    config = config if config is not None else FINAL_CONFIG
    h = config_hash(config)
    if os.path.exists(LOCK_FILE):
        with open(LOCK_FILE) as f:
            prev_hash = f.read().strip()
        if prev_hash != h:
            raise RuntimeError(
                "Конфигурация изменилась после фиксации! "
                f"было={prev_hash}, стало={h}. "
                "Итоговый запуск на test должен использовать зафиксированную конфигурацию."
            )
        print(f"Конфигурация уже зафиксирована, hash совпадает: {h}")
    else:
        with open(LOCK_FILE, 'w') as f:
            f.write(h)
        print(f"Конфигурация зафиксирована. SHA-256: {h}")
    return h


if __name__ == "__main__":
    print("SHA-256 текущей конфигурации:", config_hash())