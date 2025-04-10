
"""
Подготовка данных для второй модели RWKV (предсказание размеров ставок)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder

def visualize_bet_size_distribution(df, output_dir):
    """Визуализация распределения категорий размеров ставок"""
    plt.figure(figsize=(12, 6))
    counts = df['BetSizeCategory'].value_counts()
    sns.barplot(x=counts.index, y=counts.values)
    plt.title('Распределение категорий размеров ставок')
    plt.xlabel('Категория')
    plt.ylabel('Количество')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bet_size_distribution.png')
    plt.close()
    
    print("\nРаспределение категорий размеров ставок:")
    for category, count in counts.items():
        print(f"{category}: {count} ({count/len(df)*100:.1f}%)")

def calculate_bet_size_category(row):
    """Определяет категорию размера ставки"""
    if row['Allin'] == 1:
        return 'all-in'
    bet_percent = (row['Bet'] / row['Pot']) * 100
    if bet_percent < 26:
        return 'very_small'
    elif bet_percent < 44:
        return 'small'
    elif bet_percent < 58:
        return 'medium'
    elif bet_percent < 78:
        return 'medium_large'
    elif bet_percent < 92:
        return 'large'
    elif bet_percent < 200:
        return 'very_large'
    return 'all-in'

def prepare_bet_size_data(df):
    """Подготовка данных для модели размеров ставок"""
    # Фильтруем только Bet и Raise
    df_size = df[df['Action'].isin(['Bet', 'Raise'])].copy()
    
    # Добавляем категорию размера ставки
    df_size['BetSizeCategory'] = df_size.apply(calculate_bet_size_category, axis=1)
    
    # Добавляем признаки для второй модели
    df_size['PotToStack'] = df_size['Pot'] / df_size['Stack']
    df_size['BetToStack'] = df_size['Bet'] / df_size['Stack']
    
    return df_size
