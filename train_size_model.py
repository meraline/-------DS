"""
Обучение модели для предсказания размеров ставок
"""
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from data_preparation_size import prepare_bet_size_data
from train_process import train_poker_model

def calculate_bet_size_category(row):
    """Categorizes bet size."""
    bet = row['Bet']
    pot = row['Pot']
    stack = row['Stack']
    
    if bet / pot <= 0.25:
        return 'Small'
    elif bet / pot <= 0.5:
        return 'Medium'
    elif bet / pot <= 0.75:
        return 'Large'
    elif bet / pot <= 1:
        return 'Pot-Sized'
    elif bet / stack <= 0.25:
        return 'Small_Stack'
    elif bet / stack <= 0.5:
        return 'Medium_Stack'
    else:
        return 'Large_Stack'


def visualize_bet_size_distribution(df_size, output_dir):
    """Visualizes the distribution of bet size categories."""
    plt.figure(figsize=(12, 6))
    sns.countplot(x='BetSizeCategory', data=df_size)
    plt.title('Distribution of Bet Size Categories')
    plt.xlabel('Bet Size Category')
    plt.ylabel('Count')
    plt.savefig(f'{output_dir}/bet_size_distribution.png')
    plt.close()


def train_size_model(file_path, output_dir="./model_dir_size", **kwargs):
    """Обучение модели размеров ставок"""
    print("=== Запуск обучения модели RWKВ для предсказания размеров ставок ===")
    print("\nПодготовка данных для модели размеров ставок...")

    # Загружаем данные
    df = pd.read_csv(file_path)
    df_size = df[df['Action'].isin(['Bet', 'Raise'])].copy()

    # Добавляем категории размеров ставок
    df_size['BetSizeCategory'] = df_size.apply(calculate_bet_size_category, axis=1)

    # Визуализируем распределение
    visualize_bet_size_distribution(df_size, output_dir)

    # Показываем пример данных
    print("\nПример данных для обучения (первые 5 строк):")
    sample_cols = ['Action', 'Bet', 'Pot', 'Stack', 'Street_id', 'BetSizeCategory']
    print(df_size[sample_cols].head().to_string())

    # Обучаем модель
    kwargs.pop('data_preparation_fn', None)
    result = train_poker_model(
        file_path=file_path,
        output_dir=output_dir,
        **kwargs
    )

    return result

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Обучение модели размеров ставок")
    parser.add_argument("--file", type=str, default="./data/combined_data_all_processed_20250223_181119.csv")
    parser.add_argument("--output", type=str, default="./size_model_dir")
    parser.add_argument("--hidden_dim", type=int, default=192)
    parser.add_argument("--num_layers", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--max_rows", type=int, default=1500000)

    args = parser.parse_args()
    train_size_model(**vars(args))