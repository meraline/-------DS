"""
visualizations.py - Функции визуализации для процесса обучения и оценки модели RWKV
"""
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

def visualize_data_distribution(y_train, y_val, action_mapping, timestamp=None):
    """
    Визуализация распределения классов в обучающей и валидационной выборках
    
    Args:
        y_train: Метки обучающей выборки
        y_val: Метки валидационной выборки
        action_mapping: Отображение действий в числовые метки
        timestamp: Временная метка для сохранения файла
        
    Returns:
        matplotlib.figure.Figure: Фигура с визуализацией
    """
    if timestamp is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Инвертируем mapping для отображения имен классов
    action_names = {v: k for k, v in action_mapping.items()}
    
    # Подготовка данных для визуализации
    train_counts = pd.Series(y_train).value_counts().sort_index()
    val_counts = pd.Series(y_val).value_counts().sort_index()
    
    train_labels = [action_names.get(i, f"Класс {i}") for i in train_counts.index]
    val_labels = [action_names.get(i, f"Класс {i}") for i in val_counts.index]
    
    # Создание фигуры
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Обучающая выборка
    ax1.bar(train_labels, train_counts.values)
    ax1.set_title('Распределение классов в обучающей выборке')
    ax1.set_ylabel('Количество примеров')
    ax1.set_xlabel('Действие')
    for i, v in enumerate(train_counts.values):
        ax1.text(i, v + 50, str(v), ha='center')
    ax1.tick_params(axis='x', rotation=45)
    
    # Валидационная выборка
    ax2.bar(val_labels, val_counts.values)
    ax2.set_title('Распределение классов в валидационной выборке')
    ax2.set_ylabel('Количество примеров')
    ax2.set_xlabel('Действие')
    for i, v in enumerate(val_counts.values):
        ax2.text(i, v + 20, str(v), ha='center')
    ax2.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    return fig

def plot_training_history(train_losses, val_losses, train_accuracies, val_accuracies):
    """
    Визуализация истории обучения
    
    Args:
        train_losses: Потери на обучающей выборке
        val_losses: Потери на валидационной выборке
        train_accuracies: Точность на обучающей выборке
        val_accuracies: Точность на валидационной выборке
        
    Returns:
        matplotlib.figure.Figure: Фигура с графиками
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # График потерь
    ax1.plot(train_losses, label='Обучение')
    ax1.plot(val_losses, label='Валидация')
    ax1.set_title('Динамика функции потерь')
    ax1.set_xlabel('Эпоха')
    ax1.set_ylabel('Потери')
    ax1.legend()
    ax1.grid(True)
    
    # График точности
    ax2.plot(train_accuracies, label='Обучение')
    ax2.plot(val_accuracies, label='Валидация')
    ax2.set_title('Динамика точности')
    ax2.set_xlabel('Эпоха')
    ax2.set_ylabel('Точность')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    return fig

def visualize_confusion_matrix(true_labels, predictions, action_mapping, output_path=None):
    """
    Визуализация матрицы ошибок
    
    Args:
        true_labels: Истинные метки
        predictions: Предсказанные метки
        action_mapping: Отображение действий в числовые метки
        output_path: Путь для сохранения изображения
        
    Returns:
        matplotlib.figure.Figure: Фигура с матрицей ошибок
    """
    # Добавляем временную метку к имени файла
    if output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{timestamp}{ext}"
    # Обратное отображение меток
    reverse_mapping = {v: k for k, v in action_mapping.items()}
    target_names = [reverse_mapping[i] for i in sorted(reverse_mapping.keys())]
    
    # Вычисляем матрицу ошибок
    cm = confusion_matrix(true_labels, predictions)
    
    # Нормализация матрицы по строкам для получения процентов
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Создание тепловой карты
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Нормализованная матрица ошибок')
    plt.ylabel('Истинный класс')
    plt.xlabel('Предсказанный класс')
    plt.tight_layout()
    
    # Сохранение, если указан путь
    if output_path:
        plt.savefig(output_path)
        plt.close()
    
    return plt.gcf()

def visualize_class_distribution(true_labels, predictions, target_names, output_path=None):
    """
    Визуализация распределения классов в предсказаниях и истинных метках
    """
    # Добавляем временную метку к имени файла
    if output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{timestamp}{ext}"
    
    Args:
        true_labels: Истинные метки
        predictions: Предсказанные метки
        target_names: Имена классов
        output_path: Путь для сохранения изображения
        
    Returns:
        matplotlib.figure.Figure: Фигура с гистограммами
    """
    # Подсчет частот
    true_counts = pd.Series(true_labels).value_counts().sort_index()
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    
    # Индексы могут не содержать все классы, добавляем отсутствующие
    for i in range(len(target_names)):
        if i not in true_counts.index:
            true_counts[i] = 0
        if i not in pred_counts.index:
            pred_counts[i] = 0
    
    # Сортировка
    true_counts = true_counts.sort_index()
    pred_counts = pred_counts.sort_index()
    
    # Создание меток для осей
    x_labels = [target_names[i] for i in range(len(target_names))]
    
    # Создание гистограмм
    fig, ax = plt.subplots(figsize=(12, 6))
    bar_width = 0.35
    opacity = 0.8
    
    # Позиции баров
    index = np.arange(len(target_names))
    
    # Строим гистограммы
    plt.bar(index, true_counts, bar_width, alpha=opacity, color='b', label='Истинные')
    plt.bar(index + bar_width, pred_counts, bar_width, alpha=opacity, color='r', label='Предсказанные')
    
    plt.xlabel('Действие')
    plt.ylabel('Количество')
    plt.title('Распределение классов')
    plt.xticks(index + bar_width/2, x_labels, rotation=45)
    plt.legend()
    plt.tight_layout()
    
    # Сохранение, если указан путь
    if output_path:
        plt.savefig(output_path)
        plt.close()
    
    return fig

def visualize_prediction_confidence(predictions, probabilities, true_labels, target_names, output_path=None, n_samples=20):
    """
    Визуализация уверенности модели в предсказаниях
    """
    # Добавляем временную метку к имени файла
    if output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{timestamp}{ext}"
    
    Args:
        predictions: Предсказанные метки
        probabilities: Вероятности для каждого класса
        true_labels: Истинные метки (если имеются)
        target_names: Имена классов
        output_path: Путь для сохранения изображения
        n_samples: Количество примеров для отображения
        
    Returns:
        matplotlib.figure.Figure: Фигура с гистограммой уверенности
    """
    # Ограничиваем количество примеров для наглядности
    n_samples = min(n_samples, len(predictions))
    
    # Получаем предсказанные классы и их вероятности
    sample_probas = probabilities[:n_samples]
    confidences = np.max(sample_probas, axis=1)
    pred_labels = [target_names[p] for p in predictions[:n_samples]]
    
    if true_labels is not None:
        true_labels = true_labels[:n_samples]
        true_label_names = [target_names[t] for t in true_labels]
        is_correct = predictions[:n_samples] == true_labels
    else:
        true_label_names = ["N/A"] * n_samples
        is_correct = [False] * n_samples  # Просто для цвета баров
    
    # Создаем гистограмму уверенности
    plt.figure(figsize=(14, 6))
    colors = ['g' if c else 'r' for c in is_correct] if true_labels is not None else ['b'] * n_samples
    bars = plt.bar(range(n_samples), confidences, color=colors)
    
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.7)
    plt.title('Уверенность модели в предсказаниях')
    plt.xlabel('Номер примера')
    plt.ylabel('Уверенность (вероятность)')
    plt.ylim(0, 1.05)
    
    # Добавляем аннотации
    for i, (bar, label_true, label_pred) in enumerate(zip(bars, true_label_names, pred_labels)):
        plt.text(bar.get_x() + bar.get_width()/2, 0.05, 
                f"T: {label_true}\nP: {label_pred}", 
                ha='center', va='bottom', rotation=90, color='black', fontsize=8)
    
    if true_labels is not None:
        plt.legend([plt.Rectangle((0,0),1,1,color='g'), plt.Rectangle((0,0),1,1,color='r')], 
                ['Правильно', 'Неправильно'])
    
    plt.tight_layout()
    
    # Сохранение, если указан путь
    if output_path:
        plt.savefig(output_path)
        plt.close()
    
    return plt.gcf()

def visualize_predicted_distribution(predictions, target_names, output_path=None):
    """
    Визуализация распределения предсказанных классов (когда истинные метки отсутствуют)
    
    Args:
        predictions: Предсказанные метки
        target_names: Имена классов
        output_path: Путь для сохранения изображения
        
    Returns:
        matplotlib.figure.Figure: Фигура с гистограммой
    """
    # Подсчет частот предсказанных классов
    pred_counts = pd.Series(predictions).value_counts().sort_index()
    
    # Индексы могут не содержать все классы, добавляем отсутствующие
    for i in range(len(target_names)):
        if i not in pred_counts.index:
            pred_counts[i] = 0
    
    # Сортировка
    pred_counts = pred_counts.sort_index()
    
    # Создание меток для осей
    x_labels = [target_names[i] for i in range(len(target_names))]
    
    # Строим гистограмму
    plt.figure(figsize=(10, 6))
    plt.bar(x_labels, pred_counts.sort_index())
    
    plt.xlabel('Предсказанное действие')
    plt.ylabel('Количество')
    plt.title('Распределение предсказанных классов')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    # Сохранение, если указан путь
    if output_path:
        plt.savefig(output_path)
        plt.close()
    
    return plt.gcf()


def visualize_tsne(features, predictions, true_labels=None, output_path=None, max_samples=10000):
    """
    Создает t-SNE визуализацию для признаков
    """
    # Добавляем временную метку к имени файла
    if output_path:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base, ext = os.path.splitext(output_path)
        output_path = f"{base}_{timestamp}{ext}"
    """
    Создает t-SNE визуализацию для признаков, раскрашенную по предсказанным (и опционально истинным) меткам
    
    Args:
        features: Признаки для визуализации
        predictions: Предсказанные метки классов
        true_labels: Истинные метки классов (опционально)
        output_path: Путь для сохранения изображения
        max_samples: Максимальное количество образцов для визуализации
        
    Returns:
        matplotlib.figure.Figure: Фигура с t-SNE визуализацией
    """
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Проверяем размер входных данных
    print(f"Размеры данных для t-SNE: features {features.shape}, predictions {len(predictions)}")
    if true_labels is not None:
        print(f"true_labels {len(true_labels)}")
    
    # Уменьшаем размерность с помощью t-SNE
    print(f"Применение t-SNE для уменьшения размерности...")
    
    # Если данных слишком много, берем подвыборку
    if len(features) > max_samples:
        print(f"Выбираем случайную подвыборку из {max_samples} примеров для t-SNE...")
        indices = np.random.choice(len(features), max_samples, replace=False)
        features_subset = features[indices]
        predictions_subset = predictions[indices]
        if true_labels is not None:
            true_labels_subset = true_labels[indices]
    else:
        features_subset = features
        predictions_subset = predictions
        true_labels_subset = true_labels
    
    # Применяем t-SNE с оптимальными параметрами
    tsne = TSNE(
        n_components=2,
        perplexity=min(30, len(features_subset) - 1),  # perplexity должна быть меньше размера данных
        n_iter=1000,
        random_state=42,
        init='pca'
    )
    tsne_result = tsne.fit_transform(features_subset)
    
    # Создаем визуализацию
    plt.figure(figsize=(12, 10))
    
    # Определяем цвета для классов
    colors = ['blue', 'orange', 'green', 'red', 'purple']
    
    # Определяем метки классов
    class_names = ['Check', 'Call', 'Fold', 'Bet', 'Raise']
    
    # Отображаем точки
    for i, class_name in enumerate(class_names):
        mask = predictions_subset == i
        count = mask.sum()
        if count > 0:
            plt.scatter(
                tsne_result[mask, 0], 
                tsne_result[mask, 1], 
                c=colors[i % len(colors)], 
                label=f"{class_name} ({count})",
                alpha=0.7,
                s=30  # Размер точек
            )
    
    plt.title('t-SNE визуализация покерных данных')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    
    # Сохранение, если указан путь
    if output_path:
        plt.savefig(output_path, dpi=300)  # Увеличиваем DPI для лучшего качества
        plt.close()
    
    return plt.gcf()


