#!/usr/bin/env python3
"""
evaluate_model.py - Скрипт для оценки предварительно обученной модели RWKV на покерных данных
Позволяет загрузить сохраненную модель и провести оценку её производительности.
"""
import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns

# Импортируем функции визуализации из отдельного файла
from visualizations import (
    visualize_confusion_matrix,
    visualize_class_distribution,
    visualize_prediction_confidence,
    visualize_predicted_distribution,
    visualize_tsne,

)

# ---------------------- 1. Модель RWKV ----------------------
class RWKV_Block(nn.Module):
    def __init__(self, hidden_dim):
        super(RWKV_Block, self).__init__()
        self.hidden_dim = hidden_dim

        # Параметры time-mixing
        self.time_decay = nn.Parameter(torch.randn(hidden_dim) * 0.01)
        self.time_mix_k = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_v = nn.Parameter(torch.ones(hidden_dim) * 0.5)
        self.time_mix_r = nn.Parameter(torch.ones(hidden_dim) * 0.5)

        # Слои
        self.key = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.receptance = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

        # Состояние
        self.register_buffer("state", None, persistent=False)

    def reset_state(self):
        self.state = None

    def _init_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden_dim).to(self.time_decay.device)

    def forward(self, x):
        # x имеет форму [batch_size, seq_len, hidden_dim]
        batch_size, seq_len, _ = x.size()

        # Инициализация состояния
        if self.state is None or self.state.size(0) != batch_size:
            self.state = self._init_state(batch_size)

        output = []
        for t in range(seq_len):
            # Текущий временной шаг
            xt = x[:, t]

            # Time-mixing
            k = self.key(xt * self.time_mix_k + self.state * (1 - self.time_mix_k))
            v = self.value(xt * self.time_mix_v + self.state * (1 - self.time_mix_v))
            r = torch.sigmoid(self.receptance(xt * self.time_mix_r + self.state * (1 - self.time_mix_r)))

            # Обновление состояния
            self.state = xt + self.state * torch.exp(-torch.exp(self.time_decay))

            # Вычисление выхода
            out = r * self.output(v)
            output.append(out)

        return torch.stack(output, dim=1)

class SimpleRWKV(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=2):
        super(SimpleRWKV, self).__init__()

        self.embedding = nn.Linear(input_dim, hidden_dim)
        self.rwkv_layers = nn.ModuleList([RWKV_Block(hidden_dim) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(0.2)

    def reset_states(self):
        for layer in self.rwkv_layers:
            layer.reset_state()

    def forward(self, x):
        # x имеет форму [batch_size, input_dim]
        x = self.embedding(x).unsqueeze(1)  # [batch_size, 1, hidden_dim]

        for layer in self.rwkv_layers:
            residual = x
            x = layer(x)
            x = residual + x  # Остаточное соединение
            x = self.norm(x)
            x = self.dropout(x)

        x = x.squeeze(1)  # [batch_size, hidden_dim]
        return self.classifier(x)

# ---------------------- 2. Класс датасета ----------------------
class PokerDataset(Dataset):
    def __init__(self, features, targets=None):
        # Убедимся, что все данные числовые
        self.features = torch.tensor(features.astype(np.float32).values, dtype=torch.float32)

        # Если переданы метки, это датасет для оценки
        if targets is not None:
            self.targets = torch.tensor(targets.values, dtype=torch.long)
            self.has_targets = True
        else:
            self.has_targets = False

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        if self.has_targets:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]

# ---------------------- 3. Функции для загрузки данных и модели ----------------------
def load_model_artifacts(model_dir):
    """
    Загрузка сохраненной модели и всех артефактов

    Args:
        model_dir (str): Директория с сохраненной моделью

    Returns:
        dict: Словарь с загруженными артефактами
    """
    print(f"Загрузка артефактов модели из {model_dir}...")

    # Проверка существования директории
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Директория {model_dir} не существует")

    # Загрузка метаданных модели
    model_info_path = os.path.join(model_dir, "model_info.pkl")
    with open(model_info_path, 'rb') as f:
        model_info = pickle.load(f)
    print(f"Загружены метаданные модели: input_dim={model_info['input_dim']}, hidden_dim={model_info['hidden_dim']}")

    # Загрузка скейлера
    scaler_path = os.path.join(model_dir, "scaler.pkl")
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    print("Загружен скейлер для нормализации признаков")

    # Получение параметров модели
    input_dim = model_info['input_dim']
    hidden_dim = model_info['hidden_dim']
    num_layers = model_info['num_layers']
    action_mapping = model_info['action_mapping']
    num_classes = len(action_mapping)

    # Создание модели
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    model = SimpleRWKV(input_dim, hidden_dim, num_classes, num_layers).to(device)

    # Загрузка весов модели
    model_path = os.path.join(model_dir, "best_rwkv_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Модель загружена из {model_path}")

    # Получение имен признаков, если они есть
    feature_columns = model_info.get('feature_columns', [])
    if not feature_columns and 'input_dim' in model_info:
        # Если имен признаков нет, создаем фиктивные имена
        feature_columns = [f"feature_{i}" for i in range(model_info['input_dim'])]

    return {
        'model': model,
        'model_info': model_info,
        'scaler': scaler,
        'action_mapping': action_mapping,
        'feature_columns': feature_columns,
        'device': device
    }

def prepare_test_data(file_path, artifacts, max_rows=None, skip_rows=0):
    """
    Подготовка тестовых данных для оценки модели

    Args:
        file_path (str): Путь к CSV-файлу с тестовыми данными
        artifacts (dict): Словарь с артефактами модели
        max_rows (int, optional): Максимальное количество строк для загрузки
        skip_rows (int, optional): Количество строк для пропуска перед чтением (после заголовка)

    Returns:
        dict: Словарь с подготовленными тестовыми данными
    """
    print(f"Загрузка тестовых данных из {file_path}...")
    print(f"Пропуск первых {skip_rows} строк данных (после заголовка), чтение до {max_rows} строк после пропуска")

    # Проверка существования файла
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Файл {file_path} не существует")

    # Загрузка данных с учетом пропуска строк
    # header=0 указывает, что первая строка содержит заголовки
    # skiprows=skip_rows+1 пропускает заголовок (строка 0) и еще skip_rows строк данных
    if skip_rows > 0:
        if max_rows:
            df = pd.read_csv(file_path, header=0, skiprows=range(1, skip_rows+1), nrows=max_rows)
        else:
            df = pd.read_csv(file_path, header=0, skiprows=range(1, skip_rows+1))
    else:
        # Если skip_rows=0, просто читаем с начала
        if max_rows:
            df = pd.read_csv(file_path, nrows=max_rows)
        else:
            df = pd.read_csv(file_path)

    print(f"Загружено {len(df)} строк тестовых данных")

    # Остальной код функции остается без изменений...

    # Получение информации из артефактов
    action_mapping = artifacts['action_mapping']
    feature_columns = artifacts['feature_columns']
    scaler = artifacts['scaler']

    # Проверка наличия целевой переменной
    has_targets = 'Action' in df.columns
    print(f"Наличие меток в тестовых данных: {has_targets}")

    # Подготовка данных так же, как при обучении
    if has_targets:
        # Кодирование целевой переменной
        df['Action_encoded'] = df['Action'].map(action_mapping)
        df = df.dropna(subset=['Action_encoded'])
        print(f"После удаления строк с неизвестными действиями: {len(df)}")

    # Подготовка признаков
    # Заполнение пропущенных значений
    for col in df.columns:
        if col in ['Action', 'Action_encoded']:
            continue

        if df[col].dtype.kind in 'ifb':  # числовой тип или булев
            df[col] = df[col].fillna(df[col].median())
        else:  # строковый/категориальный тип
            df[col] = df[col].fillna('Unknown')

    # One-hot кодирование категориальных признаков
    cat_columns = df.select_dtypes(include=['object']).columns.tolist()
    cat_columns = [col for col in cat_columns if col != 'Action']

    # One-hot кодирование
    if cat_columns:
        df_encoded = pd.get_dummies(df, columns=cat_columns, dummy_na=False)
    else:
        df_encoded = df.copy()

    # Проверка наличия всех необходимых признаков
    print(f"Требуемое количество признаков: {len(feature_columns)}")

    # Добавление отсутствующих признаков
    missing_features = [col for col in feature_columns if col not in df_encoded.columns]

    if missing_features:
        print(f"Добавление {len(missing_features)} отсутствующих признаков.")
        for col in missing_features:
            df_encoded[col] = 0

    # Удаление лишних признаков
    extra_features = [col for col in df_encoded.columns if col not in feature_columns + ['Action', 'Action_encoded']]
    if extra_features:
        print(f"Удаление {len(extra_features)} лишних признаков.")
        df_encoded = df_encoded.drop(columns=extra_features)

    # Убеждаемся, что все признаки присутствуют и в правильном порядке
    X_test = df_encoded[feature_columns]

    print(f"Нормализация {X_test.shape[1]} признаков...")

    # Масштабирование с использованием сохраненного скейлера
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    X_test_float32 = X_test_scaled.astype(np.float32)

    result = {
        'X_test': X_test_float32,
        'feature_columns': feature_columns,
        'has_targets': has_targets,
        'df': df #add df to result
    }

    # Если есть метки, добавляем их
    if has_targets:
        y_test = df_encoded['Action_encoded']
        result['y_test'] = y_test

        # Создание DataLoader
        batch_size = 32
        test_dataset = PokerDataset(X_test_float32, y_test)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        result['test_loader'] = test_loader

        print(f"Создан DataLoader с {len(test_dataset)} примерами и размером батча {batch_size}")
    else:
        # Создание DataLoader без меток
        batch_size = 32
        test_dataset = PokerDataset(X_test_float32)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        result['test_loader'] = test_loader

        print(f"Создан DataLoader без меток с {len(test_dataset)} примерами и размером батча {batch_size}")

    return result

# ---------------------- 4. Функции для оценки модели ----------------------
def evaluate_model(model, test_loader, action_mapping, device, output_dir, test_data=None):
    """
    Оценка модели на тестовых данных
    """
    print("Оценка модели на тестовых данных...")

    model.eval()

    # Обратное отображение меток
    reverse_mapping = {v: k for k, v in action_mapping.items()}

    # Списки для хранения результатов
    all_predictions = []
    all_targets = []
    all_probabilities = []

    # Оценка модели
    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 2:  # С метками
                inputs, targets = batch
                inputs, targets = inputs.to(device), targets.to(device)
                has_targets = True
            else:  # Без меток
                inputs = batch
                inputs = inputs.to(device)
                has_targets = False

            # Сброс состояний RWKV
            model.reset_states()

            # Прямой проход
            outputs = model(inputs)
            probabilities = torch.softmax(outputs, dim=1)

            # Получение предсказаний
            _, predictions = torch.max(outputs, 1)

            # Сохранение результатов
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.append(probabilities.cpu().numpy())

            if has_targets:
                all_targets.extend(targets.cpu().numpy())

    # Конвертация в numpy массивы
    y_pred = np.array(all_predictions)
    probas = np.concatenate(all_probabilities, axis=0)

    results = {
        'predictions': y_pred,
        'probabilities': probas,
        'target_names': [reverse_mapping[i] for i in sorted(reverse_mapping.keys())],
        'test_data': test_data
    }

    # Если есть целевые метки, вычисляем метрики
    if all_targets:
        y_true = np.array(all_targets)
        results['true_labels'] = y_true

        # Вычисление метрик
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        results['accuracy'] = accuracy
        results['report'] = report
        results['confusion_matrix'] = cm

        # Вывод результатов
        print(f"Точность модели: {accuracy:.4f}")

        # Формирование текстового отчета
        target_names = [reverse_mapping[i] for i in sorted(reverse_mapping.keys())]
        text_report = classification_report(y_true, y_pred, target_names=target_names, zero_division=0)
        print("\nОтчет о классификации:")
        print(text_report)

    # Дополнительные визуализации для модели размеров ставок
    df = test_data['df'] # Accessing df from test_data
    if 'size' in output_dir:
        # Визуализация распределения размеров ставок
        sizes_dist_path = os.path.join(output_dir, "bet_sizes_distribution.png")
        plt.figure(figsize=(12, 6))
        sns.histplot(data=df[df['Action'].isin(['Bet', 'Raise'])], x='Bet', bins=50)
        plt.title('Распределение размеров ставок')
        plt.xlabel('Размер ставки')
        plt.ylabel('Количество')
        plt.savefig(sizes_dist_path)
        plt.close()
        print(f"Распределение размеров ставок сохранено в {sizes_dist_path}")

        # Визуализация распределения категорий размеров ставок
        bet_categories_path = os.path.join(output_dir, "bet_size_categories.png")
        plt.figure(figsize=(12, 6))
        bet_df = df[df['Action'].isin(['Bet', 'Raise'])].copy()
        bet_df['BetToPot'] = (bet_df['Bet'] / bet_df['Pot']) * 100
        conditions = [
            (bet_df['BetToPot'] < 26),
            (bet_df['BetToPot'] >= 26) & (bet_df['BetToPot'] < 44),
            (bet_df['BetToPot'] >= 44) & (bet_df['BetToPot'] < 58),
            (bet_df['BetToPot'] >= 58) & (bet_df['BetToPot'] < 78),
            (bet_df['BetToPot'] >= 78) & (bet_df['BetToPot'] < 92),
            (bet_df['BetToPot'] >= 92)
        ]
        choices = ['very_small', 'small', 'medium', 'medium_large', 'large', 'very_large']
        bet_df['BetSizeCategory'] = np.select(conditions, choices, default='medium')

        # Plot distribution
        sns.countplot(data=bet_df, x='BetSizeCategory', order=choices)
        plt.title('Распределение категорий размеров ставок')
        plt.xlabel('Категория ставки')
        plt.ylabel('Количество')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(bet_categories_path)
        plt.close()
        print(f"Распределение категорий размеров ставок сохранено в {bet_categories_path}")


        # Визуализация размеров ставок относительно банка
        pot_sizes_path = os.path.join(output_dir, "pot_bet_sizes.png")
        plt.figure(figsize=(12, 6))
        df_bets = df[df['Action'].isin(['Bet', 'Raise'])].copy()
        df_bets['BetToPot'] = df_bets['Bet'] / df_bets['Pot'] * 100
        sns.scatterplot(data=df_bets, x='Pot', y='BetToPot', alpha=0.5)
        plt.title('Размеры ставок относительно банка')
        plt.xlabel('Размер банка')
        plt.ylabel('Ставка (% от банка)')
        plt.savefig(pot_sizes_path)
        plt.close()
        print(f"Размеры ставок относительно банка сохранены в {pot_sizes_path}")


    # Добавляем матрицу ошибок для размеров ставок
    bet_size_mask = df['Action'].isin(['Bet', 'Raise'])
    if bet_size_mask.any():
        bet_df = df[bet_size_mask].copy()
        # Категоризация ставок относительно банка (в процентах)
        bet_df['BetToPot'] = (bet_df['Bet'] / bet_df['Pot']) * 100
        conditions = [
            (bet_df['BetToPot'] < 26),
            (bet_df['BetToPot'] >= 26) & (bet_df['BetToPot'] < 44),
            (bet_df['BetToPot'] >= 44) & (bet_df['BetToPot'] < 58),
            (bet_df['BetToPot'] >= 58) & (bet_df['BetToPot'] < 78),
            (bet_df['BetToPot'] >= 78) & (bet_df['BetToPot'] < 92),
            (bet_df['BetToPot'] >= 92)
        ]
        choices = ['very_small', 'small', 'medium', 'medium_large', 'large', 'very_large']
        bet_df['BetSizeCategory'] = np.select(conditions, choices, default='medium')

        # Создаем и сохраняем матрицу ошибок для размеров ставок
        if all_targets and bet_size_mask.any():
            # Получаем предсказания для ставок
            bet_predictions = []
            for i, prob in enumerate(probas):
                if bet_size_mask.iloc[i]:
                    bet_predictions.append(choices[np.argmax(prob)])
            
            plt.figure(figsize=(12, 8))
            cm_sizes = confusion_matrix(bet_df['BetSizeCategory'].values, 
                                     bet_predictions,
                                     labels=choices)
            sns.heatmap(cm_sizes, annot=True, fmt='d',
                       xticklabels=choices,
                       yticklabels=choices)
            plt.title('Матрица ошибок для размеров ставок')
            plt.xlabel('Предсказанные значения')
            plt.ylabel('Истинные значения')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'bet_size_confusion_matrix.png'))
            plt.close()

        if all_targets:
            # Получаем истинные метки размеров ставок
            y_true_sizes = bet_df['BetSizeCategory'].values

            # Предсказываем размеры ставок для тех же строк
            bet_indices = df[bet_size_mask].index
            y_pred_sizes = ['very_small'] * len(y_true_sizes)  # Дефолтное значение

            for i, prob in enumerate(probas):
                if i in bet_indices:
                    bet_idx = list(bet_indices).index(i)
                    predicted_class = np.argmax(prob)
                    y_pred_sizes[bet_idx] = choices[predicted_class]

            # Создаем матрицу ошибок для размеров ставок
            plt.figure(figsize=(10, 8))
            cm_sizes = confusion_matrix(y_true_sizes, y_pred_sizes, labels=choices)
            sns.heatmap(cm_sizes, annot=True, fmt='d',
                       xticklabels=choices,
                       yticklabels=choices)
            plt.title('Матрица ошибок для размеров ставок')
            plt.xlabel('Предсказанные значения')
            plt.ylabel('Истинные значения')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'bet_size_confusion_matrix.png'))
            plt.close()


            plt.figure(figsize=(10, 8))
            sns.heatmap(confusion_matrix(y_true_sizes, y_pred_sizes),
                       annot=True, fmt='d',
                       xticklabels=np.unique(y_true_sizes),
                       yticklabels=np.unique(y_true_sizes))
            plt.title('Матрица ошибок для размеров ставок')
            plt.xlabel('Предсказанные значения')
            plt.ylabel('Истинные значения')
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'bet_size_confusion_matrix.png'))
            plt.close()

            # Сохраняем отчет о классификации размеров ставок
            with open(os.path.join(output_dir, 'bet_size_classification_report.txt'), 'w') as f:
                f.write(classification_report(y_true_sizes, y_pred_sizes, zero_division=0))

            # Визуализация ROC-кривой для all-in предсказаний
            df_allin = test_data['df'].copy()
            if 'Allin' in df_allin.columns:
                y_true_allin = (df_allin['Allin'] == 1).astype(int)

                # Получаем вероятности для all-in класса
                all_in_probs = probas[:, 1] if len(probas.shape) > 1 else probas

                fpr, tpr, _ = roc_curve(y_true_allin, all_in_probs)
                roc_auc = auc(fpr, tpr)

                plt.figure(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve for All-in Predictions')
                plt.legend(loc="lower right")
                plt.savefig(os.path.join(output_dir, 'allin_roc_curve.png'))
                plt.close()

                # Анализ распределения all-in
                print("\nСтатистика по all-in решениям:")
                print(df_allin['Allin'].value_counts())
                print("\nПримеры all-in ситуаций:")
                print(df_allin[df_allin['Allin'] == 1][['Bet', 'Stack', 'Pot', 'Street_id']].head())

                # Анализ ситуаций с all-in
                allin_df = df[df['Allin'] == 1].copy()
                plt.figure(figsize=(12, 6))
                sns.boxplot(data=allin_df, x='Street_id', y='Stack')
                plt.title('Распределение стеков при all-in по улицам')
                plt.savefig(os.path.join(output_dir, 'allin_stack_distribution.png'))
                plt.close()

    # Displaying all features in a table
    if all_targets:
        feature_table_path = os.path.join(output_dir, 'feature_table.csv')
        test_data['df'][test_data['df'].columns].to_csv(feature_table_path, index=False)
        print(f"Таблица признаков сохранена в {feature_table_path}")


    return results

# ---------------------- 6. Основная функция ----------------------
def main():
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(description='Оценка модели RWKV на покерных данных')
    parser.add_argument('--model_dir', type=str, default='/home/tofan/Документы/GitLab_grace/ДИПЛОМ DS/model_dir', help='Путь к директории с моделью')
    parser.add_argument('--test', type=str, default='/home/tofan/data1/csv/split_data/buyin_type_MTT_250.csv', required=False, help='Путь к файлу с тестовыми данными')
    parser.add_argument('--output', type=str, default='/home/tofan/Документы/GitLab_grace/ДИПЛОМ DS/evaluation_results', help='Путь к директории для сохранения результатов')
    parser.add_argument('--max_rows', type=int, default=150000, help='Максимальное количество строк для загрузки')
    parser.add_argument('--skip_rows', type=int, default=0, help='Количество строк данных для пропуска после заголовка')
    parser.add_argument('--tsne_samples', type=int, default=15000, help='Максимальное количество образцов для t-SNE визуализации')
    args = parser.parse_args()

    # Создание директории для результатов, если она не существует
    os.makedirs(args.output, exist_ok=True)

    print(f"Результаты оценки будут сохранены в {args.output}")

    try:
        # Загрузка артефактов модели
        artifacts = load_model_artifacts(args.model_dir)

        # Подготовка тестовых данных с пропуском строк
        test_data = prepare_test_data(args.test, artifacts, args.max_rows, args.skip_rows)

        # Оценка модели
        results = evaluate_model(
            artifacts['model'],
            test_data['test_loader'],
            artifacts['action_mapping'],
            artifacts['device'],
            args.output,
            test_data
        )

        # Визуализация результатов - используем импортированные функции напрямую
        print(f"Сохранение визуализаций в {args.output}...")

        # Если есть метки, визуализируем матрицу ошибок и распределение классов
        if 'true_labels' in results:
            # Собираем все данные из загрузчика для t-SNE
            all_features = []
            all_predictions = []
            all_true_labels = []

            model = artifacts['model']
            device = artifacts['device']

            print("Сбор данных для t-SNE визуализации...")
            with torch.no_grad():
                for batch in test_data['test_loader']:
                    inputs, targets = batch
                    inputs = inputs.to(device)

                    # Сброс состояний RWKV
                    model.reset_states()

                    # Прямой проход
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)

                    # Собираем данные
                    all_features.append(inputs.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())
                    all_true_labels.append(targets.numpy())

            # Объединяем данные
            all_features = np.vstack(all_features)
            all_predictions = np.concatenate(all_predictions)
            all_true_labels = np.concatenate(all_true_labels)

            print(f"Собрано данных для t-SNE: {len(all_features)} образцов")

            # Визуализация с помощью t-SNE
            tsne_path = os.path.join(args.output, "tsne_visualization.png")
            visualize_tsne(
                all_features,
                all_predictions,
                all_true_labels,
                tsne_path,
                max_samples=args.tsne_samples
            )
            print(f"t-SNE визуализация сохранена в {tsne_path}")


            # Визуализация матрицы ошибок
            cm_path = os.path.join(args.output, "confusion_matrix.png")
            visualize_confusion_matrix(
                results['true_labels'],
                results['predictions'],
                artifacts['action_mapping'],
                cm_path
            )
            print(f"Матрица ошибок сохранена в {cm_path}")

            # Визуализация распределения классов
            dist_path = os.path.join(args.output, "class_distribution.png")
            visualize_class_distribution(
                results['true_labels'],
                results['predictions'],
                results['target_names'],
                dist_path
            )
            print(f"Распределение классов сохранено в {dist_path}")

            # Визуализация уверенности модели
            conf_path = os.path.join(args.output, "prediction_confidence.png")
            visualize_prediction_confidence(
                results['predictions'],
                results['probabilities'],
                results['true_labels'],
                results['target_names'],
                conf_path
            )
            print(f"График уверенности модели сохранен в {conf_path}")

            # Сохранение текстового отчета о классификации
            report_path = os.path.join(args.output, "classification_report.txt")
            with open(report_path, 'w') as f:
                f.write(f"Точность модели: {results['accuracy']:.4f}\n\n")
                f.write("Отчет о классификации:\n")
                f.write(classification_report(
                    results['true_labels'],
                    results['predictions'],
                    target_names=results['target_names'],
                    zero_division=0
                ))
            print(f"Отчет о классификации сохранен в {report_path}")
        else:
            # Если нет меток, собираем данные для t-SNE без меток
            all_features = []
            all_predictions = []

            model = artifacts['model']
            device = artifacts['device']

            print("Сбор данных для t-SNE визуализации...")
            with torch.no_grad():
                for batch in test_data['test_loader']:
                    inputs = batch
                    inputs = inputs.to(device)

                    # Сброс состояний RWKV
                    model.reset_states()

                    # Прямой проход
                    outputs = model(inputs)
                    _, predictions = torch.max(outputs, 1)

                    # Собираем данные
                    all_features.append(inputs.cpu().numpy())
                    all_predictions.append(predictions.cpu().numpy())

            # Объединяем данные
            all_features = np.vstack(all_features)
            all_predictions = np.concatenate(all_predictions)

            print(f"Собрано данных для t-SNE: {len(all_features)} образцов")

            # Визуализация с помощью t-SNE
            tsne_path = os.path.join(args.output, "tsne_visualization.png")
            visualize_tsne(
                all_features,
                all_predictions,
                None,
                tsne_path,
                max_samples=args.tsne_samples
            )
            print(f"t-SNE визуализация сохранена в {tsne_path}")

            # Если нет меток, сохраняем только предсказания
            # Создаем DataFrame с предсказаниями
            predictions = results['predictions']
            probas = results['probabilities']
            target_names = results['target_names']

            df_results = pd.DataFrame()
            df_results['Predicted_Class'] = predictions
            df_results['Predicted_Action'] = [target_names[p] for p in predictions]
            df_results['Confidence'] = np.max(probas, axis=1)

            # Добавляем вероятности для каждого класса
            for i, action in enumerate(target_names):
                df_results[f'Prob_{action}'] = probas[:, i]

            # Сохраняем предсказания в CSV
            pred_path = os.path.join(args.output, "predictions.csv")
            df_results.to_csv(pred_path, index=False)
            print(f"Предсказания сохранены в {pred_path}")

            # Визуализация распределения предсказанных классов
            dist_path = os.path.join(args.output, "predicted_distribution.png")
            visualize_predicted_distribution(
                results['predictions'],
                results['target_names'], dist_path
            )
            print(f"Распределение предсказанных классов сохранено в {dist_path}")

            # Визуализация уверенности модели без истинных меток
            conf_path = os.path.join(args.output, "prediction_confidence.png")
            visualize_prediction_confidence(
                results['predictions'],
                results['probabilities'],
                None,  # нет истинных меток
                results['target_names'],
                conf_path
            )
            print(f"График уверенности предсказаний сохранен в {conf_path}")

        print("Оценка модели завершена успешно.")
        return 0

    except Exception as e:
        print(f"Ошибка при оценке модели: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())