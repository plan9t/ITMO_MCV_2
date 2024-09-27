import cv2
import numpy as np


def template_matching(template_path, image_path):
    # Загрузка изображения и шаблона
    image = cv2.imread(image_path)
    template = cv2.imread(template_path, 0)

    # Проверка размеров
    if template.shape[0] > image.shape[0] or template.shape[1] > image.shape[1]:
        print("Ошибка: Шаблон больше входного изображения.")
        return

    # Преобразование изображения в оттенки серого
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Применение сопоставления шаблонов
    res = cv2.matchTemplate(image_gray, template, cv2.TM_CCOEFF_NORMED)

    # Нахождение координат наилучшего совпадения
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

    # Устанавливаем порог вероятности
    threshold = 0.1  # 70%

    if max_val >= threshold:
        # Вычисление координат рамки
        top_left = max_loc
        bottom_right = (top_left[0] + template.shape[1], top_left[1] + template.shape[0])

        # Рисование рамки на изображении
        cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), 2)

        # Подготовка текста с вероятностью совпадения
        match_probability = f'%: {max_val:.2f}'

        # Добавление текста на изображение
        cv2.putText(image, match_probability, (top_left[0], top_left[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)

    # Отображение результата
    cv2.imshow('Result', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Пример использования
template_matching('src/6_crop.jpg', 'src/6_input.jpg')