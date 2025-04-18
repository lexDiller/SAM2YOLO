# Инференс SAM и конвертация предсказаний в формат YOLO

## Обзор
В данном проекте реализован процесс автоматического извлечения сегментаций с помощью модели **SAM (Segment Anything Model)** и последующая конвертация полученных предсказаний в аннотации, совместимые с форматом YOLO. Это позволяет использовать автоматизированные разметки для дальнейшего обучения модели YOLO, что значительно ускоряет и упрощает процесс подготовки датасета.

## Основные этапы

1. **Инференс SAM**  
   - Выполняется сегментация объектов на изображениях с использованием модели SAM.
   - Полученные маски представляют собой точные сегментационные разметки объектов.

2. **Конвертация в формат YOLO**  
   - Из сегментационных масок вычисляются ограничивающие рамки (bounding boxes).
   - Координаты рамок нормализуются относительно размеров изображения.
   - Аннотации сохраняются в виде текстовых файлов, где каждая строка соответствует одному объекту и имеет формат:
     ```
     class_id x_center y_center width height
     ```
     Все значения нормализованы в диапазоне [0, 1].

3. **Использование для обучения YOLO**  
   - Полученные аннотации можно напрямую применять для обучения моделей YOLO.
   - Этот подход позволяет автоматизировать процесс создания размеченного датасета, сокращая трудозатраты на ручную разметку.

## Содержимое репозитория

- **SAM2YOLO.ipynb**  
  Jupyter Notebook, в котором:
  - Производится инференс с использованием SAM для сегментации объектов.
  - Реализована логика конвертации сегментаций в аннотации в формате YOLO.
  - Сохраняются результаты в виде файлов разметки, готовых для обучения YOLO.

## Преимущества подхода

- **Автоматизация разметки**  
  Снижение необходимости в ручной разметке датасета за счёт использования мощных сегментационных моделей.
- **Гибкость**  
  Возможность легко адаптировать разметки под различные версии YOLO и другие модели обнаружения объектов.
- **Масштабируемость**  
  Подходит для обработки больших объемов данных, что особенно актуально при работе с реальными потоками изображений.
