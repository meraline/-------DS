
"""
Скрипт для обучения модели предсказания all-in решений
"""
import os
import torch
from data_preparation import load_and_prepare_data
from model_allin import RWKVAllInModel
from train_process import train_poker_model

def prepare_allin_data(df):
    """Подготовка данных для модели all-in"""
    # Вычисляем отношение ставки к банку
    df['BetToPot'] = (df['Bet'] / df['Pot'] * 100).fillna(0)
    
    # Создаем целевую переменную (1 для all-in, 0 для остальных)
    df['IsAllIn'] = ((df['BetToPot'] >= 200) | (df['Allin'] == 1)).astype(int)
    
    print("\nРаспределение all-in решений:")
    print(df['IsAllIn'].value_counts(normalize=True).round(3))
    return df

def train_allin_model(file_path, output_dir, **kwargs):
    """Обучение модели для предсказания all-in"""
    print("\n=== Запуск обучения модели RWKV для предсказания all-in ===\n")
    
    # Создаем директорию для сохранения результатов
    os.makedirs(output_dir, exist_ok=True)
    
    # Добавляем функцию подготовки данных
    kwargs['data_preparation_fn'] = prepare_allin_data
    
    # Обучаем модель
    result = train_poker_model(
        file_path=file_path,
        output_dir=output_dir,
        **kwargs
    )
    
    return result

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Обучение модели all-in")
    parser.add_argument("--file", type=str, default="./data/combined_data_all_processed_20250223_181119.csv")
    parser.add_argument("--output", type=str, default="./allin_model_dir")
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--max_rows", type=int, default=1500000)
    
    args = parser.parse_args()
    
    result = train_allin_model(
        file_path=args.file,
        output_dir=args.output,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        epochs=args.epochs,
        learning_rate=args.lr,
        max_rows=args.max_rows
    )
