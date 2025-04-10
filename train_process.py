"""
train_process.py - Функции для обучения модели RWKV на покерных данных
"""
import os
import datetime
import pickle
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Импорт из локальных модулей
from model import SimpleRWKV
from data_preparation import load_and_prepare_data
from visualizations import visualize_data_distribution, plot_training_history

# ---------------------- Функции обучения модели ----------------------
def train_model_with_history(data_dict, hidden_dim=64, num_layers=2, epochs=10, lr=0.001):
    """
    Обучение модели RWKV с отслеживанием истории для визуализации
    
    Args:
        data_dict (dict): Словарь с подготовленными данными
        hidden_dim (int): Размерность скрытого состояния
        num_layers (int): Количество слоев RWKV
        epochs (int): Количество эпох обучения
        lr (float): Скорость обучения
        
    Returns:
        tuple: (модель, история обучения)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Используется устройство: {device}")

    # Получение данных из словаря
    train_loader = data_dict['train_loader']
    val_loader = data_dict['val_loader']
    input_dim = data_dict['input_dim']
    action_mapping = data_dict['action_mapping']

    # Создание модели
    num_classes = len(action_mapping)
    model = SimpleRWKV(input_dim, hidden_dim, num_classes, num_layers).to(device)

    # Функция потерь и оптимизатор
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # История обучения
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    # Обучение
    best_accuracy = 0.0
    print(f"Начало обучения на {epochs} эпох...")

    for epoch in range(epochs):
        # Обучение
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # Сброс состояний RWKV
            model.reset_states()

            # Обнуление градиентов
            optimizer.zero_grad()

            # Прямой проход
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # Обратный проход
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)  # Предотвращение взрыва градиентов
            optimizer.step()

            train_loss += loss.item()
            
            # Подсчет правильных предсказаний для тренировочной точности
            _, predicted = torch.max(outputs, 1)
            train_total += targets.size(0)
            train_correct += (predicted == targets).sum().item()

        # Оценка
        model.eval()
        val_correct = 0
        val_total = 0
        val_loss = 0.0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(device), targets.to(device)

                # Сброс состояний RWKV
                model.reset_states()

                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                val_total += targets.size(0)
                val_correct += (predicted == targets).sum().item()

        # Вычисление метрик
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        train_accuracy = train_correct / train_total
        val_accuracy = val_correct / val_total

        # Обновление истории
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['train_acc'].append(train_accuracy)
        history['val_acc'].append(val_accuracy)

        print(f'Эпоха {epoch+1}/{epochs}, Потери обучения: {avg_train_loss:.4f}, '\
              f'Потери валидации: {avg_val_loss:.4f}, '\
              f'Точность обучения: {train_accuracy*100:.2f}%, '\
              f'Точность валидации: {val_accuracy*100:.2f}%')

        # Сохранение лучшей модели
        if val_accuracy*100 > best_accuracy:
            best_accuracy = val_accuracy*100
            torch.save(model.state_dict(), 'best_rwkv_model.pth')
            print(f'Модель сохранена с точностью {best_accuracy:.2f}%')

    # Загрузка лучшей модели
    model.load_state_dict(torch.load('best_rwkv_model.pth'))
    print(f'Обучение завершено. Лучшая точность на валидации: {best_accuracy:.2f}%')

    return model, history

# ---------------------- Основная функция обучения ----------------------
def train_poker_model(file_path, output_dir=None, hidden_dim=128, num_layers=4, epochs=10, learning_rate=0.001, max_rows=None, data_preparation_fn=None):
    """
    Основная функция для обучения покерной модели
    
    Args:
        file_path (str): Путь к CSV-файлу с данными
        output_dir (str, optional): Директория для сохранения результатов
        hidden_dim (int): Размерность скрытого состояния
        num_layers (int): Количество слоев RWKV
        epochs (int): Количество эпох обучения
        learning_rate (float): Скорость обучения
        max_rows (int, optional): Максимальное число строк для загрузки
        data_preparation_fn (callable, optional): Функция для дополнительной подготовки данных
        
    Returns:
        dict: Результаты обучения
    """
    # Создаем директорию для сохранения результатов
    if output_dir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"poker_model_{timestamp}"
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Результаты будут сохранены в {output_dir}")
    
    # Подготовка данных с балансировкой классов
    print("Загрузка и подготовка данных...")
    if isinstance(file_path, str):
        df = pd.read_csv(file_path)
        if max_rows:
            df = df.head(max_rows)
    else:
        df = file_path
        if max_rows:
            df = df.head(max_rows)
    
    # Применяем пользовательскую функцию подготовки данных если она предоставлена
    if data_preparation_fn:
        df = data_preparation_fn(df)
    
    data_dict = load_and_prepare_data(df, balance_classes=True, is_dataframe=True)
    
    # Визуализация распределения классов
    print("Визуализация распределения классов...")
    class_dist_fig = visualize_data_distribution(
        data_dict['y_train'], 
        data_dict['y_val'], 
        data_dict['action_mapping']
    )
    class_dist_fig.savefig(os.path.join(output_dir, 'class_distribution.png'))
    plt.close(class_dist_fig)
    
    # Обучение модели с отслеживанием истории
    print("Начало обучения модели...")
    model, history = train_model_with_history(
        data_dict,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        epochs=epochs,
        lr=learning_rate
    )
    
    # Визуализация истории обучения
    print("Визуализация истории обучения...")
    history_fig = plot_training_history(
        history['train_loss'], 
        history['val_loss'],
        history['train_acc'], 
        history['val_acc']
    )
    history_fig.savefig(os.path.join(output_dir, 'training_history.png'))
    plt.close(history_fig)
    
    # Сохранение модели
    model_path = os.path.join(output_dir, "best_rwkv_model.pth")
    torch.save(model.state_dict(), model_path)
    print(f"Модель сохранена в {model_path}")
    
    # Сохранение метаданных модели
    model_info = {
        'input_dim': data_dict['input_dim'],
        'hidden_dim': hidden_dim,
        'num_layers': num_layers,
        'action_mapping': data_dict['action_mapping'],
        'feature_columns': data_dict['feature_columns'],
        'epochs': epochs,
        'learning_rate': learning_rate
    }
    
    # Сохранение скейлера для последующей нормализации признаков
    scaler_path = os.path.join(output_dir, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(data_dict['scaler'], f)
    print(f"Скейлер сохранен в {scaler_path}")
    
    # Сохранение метаданных модели
    model_info_path = os.path.join(output_dir, "model_info.pkl")
    with open(model_info_path, 'wb') as f:
        pickle.dump(model_info, f)
    print(f"Информация о модели сохранена в {model_info_path}")
    
    # Сохранение простого текстового описания модели
    with open(os.path.join(output_dir, "model_description.txt"), 'w') as f:
        f.write(f"Модель RWKV для покерных данных\n")
        f.write(f"Размерность входа: {data_dict['input_dim']}\n")
        f.write(f"Размерность скрытого состояния: {hidden_dim}\n")
        f.write(f"Количество слоев: {num_layers}\n")
        f.write(f"Количество классов: {len(data_dict['action_mapping'])}\n")
        f.write(f"Отображение действий: {data_dict['action_mapping']}\n")
        f.write(f"Количество эпох обучения: {epochs}\n")
        f.write(f"Скорость обучения: {learning_rate}\n")
    
    return {
        'model': model,
        'model_path': model_path,
        'scaler_path': scaler_path,
        'model_info_path': model_info_path,
        'action_mapping': data_dict['action_mapping'],
        'output_dir': output_dir,
        'history': history
    }

# Точка входа для запуска из командной строки
if __name__ == "__main__":
    import argparse
    
    # Аргументы командной строки с установленными значениями по умолчанию
    parser = argparse.ArgumentParser(description="Обучение RWKV модели на покерных данных")
    parser.add_argument("--file", type=str, default="/path/to/train_data.csv", help="Путь к CSV-файлу с данными")
    parser.add_argument("--output", type=str, default="./model_dir", help="Директория для сохранения результатов")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Размерность скрытого состояния")
    parser.add_argument("--num_layers", type=int, default=4, help="Количество слоев RWKV")
    parser.add_argument("--epochs", type=int, default=10, help="Количество эпох обучения")
    parser.add_argument("--lr", type=float, default=0.001, help="Скорость обучения")
    parser.add_argument("--max_rows", type=int, default=None, help="Максимальное число строк для загрузки")
    
    args = parser.parse_args()
    
    print("Запуск обучения с параметрами:")
    print(f"  Файл данных: {args.file}")
    print(f"  Директория вывода: {args.output}")
    print(f"  Размерность скрытого состояния: {args.hidden_dim}")
    print(f"  Количество слоев: {args.num_layers}")
    print(f"  Эпохи: {args.epochs}")
    print(f"  Скорость обучения: {args.lr}")
    print(f"  Максимальное число строк: {args.max_rows if args.max_rows else 'все'}")
    
    train_poker_model(
        file_path=args.file,
        output_dir=args.output,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_rows=args.max_rows
    )