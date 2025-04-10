
import pandas as pd
import numpy as np

def analyze_features(data_path):
    """Анализ всех признаков в датасете"""
    # Загружаем данные
    df = pd.read_csv(data_path)
    
    # Создаем таблицу с информацией о признаках
    feature_info = []
    
    for col in df.columns:
        info = {
            'Название': col,
            'Тип': str(df[col].dtype),
            'Уникальных значений': df[col].nunique(),
            'Пропущенные значения': df[col].isnull().sum(),
            'Пример значений': str(df[col].dropna().sample(n=3).tolist())[:50]
        }
        
        if df[col].dtype in ['int64', 'float64']:
            info.update({
                'Среднее': df[col].mean(),
                'Медиана': df[col].median(),
                'Стд. отклонение': df[col].std()
            })
        
        feature_info.append(info)
    
    # Создаем DataFrame с информацией о признаках
    feature_df = pd.DataFrame(feature_info)
    
    # Группируем признаки по категориям
    base_features = ['Hand', 'TournamentNumber', 'Street_id', 'Round', 'Action', 'Bet', 'Stack', 'Position']
    player_stats = [col for col in df.columns if any(x in col for x in ['VPIP', 'PFR', 'Frequency', 'Aggression'])]
    cards = [col for col in df.columns if 'Card' in col]
    position_features = [col for col in df.columns if 'Position' in col]
    derived_features = [col for col in df.columns if col not in base_features + player_stats + cards + position_features]
    
    # Сохраняем результаты
    with pd.ExcelWriter('feature_analysis.xlsx') as writer:
        feature_df.to_excel(writer, sheet_name='All Features', index=False)
        
        # Сохраняем отдельные категории
        pd.DataFrame({'Base Features': base_features}).to_excel(writer, sheet_name='Categories', startcol=0)
        pd.DataFrame({'Player Stats': player_stats}).to_excel(writer, sheet_name='Categories', startcol=2)
        pd.DataFrame({'Card Features': cards}).to_excel(writer, sheet_name='Categories', startcol=4)
        pd.DataFrame({'Position Features': position_features}).to_excel(writer, sheet_name='Categories', startcol=6)
        pd.DataFrame({'Derived Features': derived_features}).to_excel(writer, sheet_name='Categories', startcol=8)

if __name__ == "__main__":
    data_path = "./data/combined_data_all_processed_20250223_181119.csv"
    analyze_features(data_path)
    print("Анализ признаков сохранен в feature_analysis.xlsx")
