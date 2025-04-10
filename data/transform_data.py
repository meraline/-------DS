import pandas as pd
import numpy as np
import os
from datetime import datetime

def process_large_file(input_path, output_path, chunk_size=100000):
    """
    Обработка большого файла чанками с исправленным подходом для экономии памяти.
    
    Args:
        input_path (str): Путь к исходному CSV файлу
        output_path (str): Путь для сохранения обработанного файла
        chunk_size (int): Размер чанка (количество строк)
    """
    print(f"Начинаем обработку файла: {input_path}")
    print(f"Результаты будут сохранены в: {output_path}")
    
    # Создаем директорию для выходного файла, если её нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # ======== ШАГ 1: Подготовка маппинга игроков ========
    print("\nШАГ 1: Создание маппинга игроков...")
    
    # Читаем только колонку PlayerName для создания маппинга
    player_name_to_id = {}
    total_rows = 0
    
    for chunk_id, chunk in enumerate(pd.read_csv(input_path, usecols=['PlayerName'], chunksize=chunk_size)):
        chunk_id += 1
        print(f"  Обработка чанка {chunk_id}, {len(chunk)} строк...")
        
        # Извлекаем имена игроков без суффикса
        clean_names = chunk['PlayerName'].str.split('_').str[0].unique()
        
        # Добавляем новых игроков в маппинг
        for name in clean_names:
            if name not in player_name_to_id:
                player_name_to_id[name] = len(player_name_to_id)
        
        total_rows += len(chunk)
    
    print(f"  Всего обработано строк: {total_rows}")
    print(f"  Найдено уникальных игроков: {len(player_name_to_id)}")
    
    # ======== ШАГ 2: Обработка файла по частям ========
    print("\nШАГ 2: Последовательная обработка и запись данных...")
    
    # Открываем итератор для чтения чанков
    chunks = pd.read_csv(input_path, chunksize=chunk_size)
    
    # Обрабатываем первый чанк для получения списка колонок
    try:
        first_chunk = next(chunks)
        all_columns = list(first_chunk.columns)
        
        # Обрабатываем первый чанк
        processed_chunk = process_chunk(first_chunk, player_name_to_id)
        processed_columns = processed_chunk.columns
        
        # Записываем заголовки и первый чанк
        processed_chunk.to_csv(output_path, index=False, mode='w')
        
        # Счетчик для отслеживания прогресса
        rows_processed = len(processed_chunk)
        print(f"  Первый чанк обработан, {rows_processed} строк записано")
        
    except StopIteration:
        print("  Файл пуст, нет данных для обработки")
        return
    
    # Обрабатываем остальные чанки
    for chunk_id, chunk in enumerate(chunks, 2):
        # Обрабатываем чанк
        processed_chunk = process_chunk(chunk, player_name_to_id)
        
        # Добавляем в файл (append mode)
        processed_chunk.to_csv(output_path, index=False, header=False, mode='a')
        
        rows_processed += len(processed_chunk)
        if chunk_id % 5 == 0:
            print(f"  Обработан чанк {chunk_id}, всего {rows_processed} строк")
    
    print(f"\nГотово! Обработано всего {rows_processed} строк")
    print(f"Данные сохранены в: {output_path}")

def process_chunk(chunk, player_name_to_id):
    """
    Обрабатывает один чанк данных, применяя все необходимые преобразования.
    
    Args:
        chunk (DataFrame): Чанк данных для обработки
        player_name_to_id (dict): Словарь маппинга имен игроков на ID
        
    Returns:
        DataFrame: Обработанный чанк данных
    """
    # Создаем копию чанка
    df = chunk.copy()
    
    # ========== БАЗОВЫЕ ПРЕОБРАЗОВАНИЯ ==========
    
    # 1) КОРРЕКЦИЯ ID ИГРОКОВ
    # Извлекаем имя игрока без суффикса
    df['CleanPlayerName'] = df['PlayerName'].str.split('_').str[0]
    
    # Обновляем UniquePlayerId
    df['UniquePlayerId'] = df['CleanPlayerName'].map(player_name_to_id)
    
    # 2) КОРРЕКЦИЯ ACTION ДЛЯ ALLIN
    if 'Allin' in df.columns:
        df.loc[df['Allin'] == 1, 'Action'] = 'Allin'
    
    # 3) ПРИВЯЗКА ROUND К STREET_ID
    # Сохраняем оригинальный Round
    df['OriginalRound'] = df['Round'].copy()
    
    # Сортируем данные для обработки по группам
    df = df.sort_values(['Hand', 'Street_id', 'ActionOrder'])
    
    # Обновляем Round для каждой группы Hand и Street_id
    # Используем groupby для обновления Round
    df['Round'] = 0  # Инициализируем нулями
    
    # Вычисляем новые значения Round
    new_rounds = []
    for (hand, street), group in df.groupby(['Hand', 'Street_id']):
        # Создаем последовательные номера для каждой группы
        group_rounds = list(range(1, len(group) + 1))
        new_rounds.extend(group_rounds)
    
    # Сортируем DataFrame так же, как был выполнен groupby
    df = df.sort_values(['Hand', 'Street_id', 'ActionOrder']).reset_index(drop=True)
    
    # Присваиваем новые значения Round
    if len(new_rounds) == len(df):
        df['Round'] = new_rounds
    
    # ========== ДОПОЛНИТЕЛЬНЫЕ ПРИЗНАКИ ==========
    
    # Маппинг позиций
    pos_map = {
        "EP1": 0, "EP2": 1, "EP3": 2, "MP1": 2,
        "MP2": 3, "MP3": 4, "CO": 5, "BTN": 6,
        "SB": 7, "BB": 8
    }
    
    # 1) Преобразование позиций в числовые значения
    if 'Position' in df.columns:
        df["PositionNum"] = df["Position"].map(lambda x: pos_map.get(x, -1))
    else:
        df["PositionNum"] = -1
    
    # 2) Находим позицию последнего беттора/рейзера
    if 'Action' in df.columns and 'PositionNum' in df.columns:
        raiser_mask = df['Action'].isin(['Bet', 'Raise'])
        df['TempRaiserPos'] = np.where(raiser_mask, df['PositionNum'], np.nan)
        df["RaiserPosNum"] = df.groupby("Hand")['TempRaiserPos'].transform(
            lambda x: x.ffill().fillna(-1)
        )
        df = df.drop('TempRaiserPos', axis=1)
    else:
        df["RaiserPosNum"] = -1
    
    # 3) Определяем RelativePosition
    df["RelativePosition"] = np.where(
        df["RaiserPosNum"] == -1,
        "NoRaiser",
        np.where(
            df["PositionNum"] == df["RaiserPosNum"],
            "Self",
            np.where(
                df["PositionNum"] < df["RaiserPosNum"],
                "Before",
                "After"
            )
        )
    )
    
    # 4) Пот-оддсы с безопасным делением
    if 'Pot' in df.columns and 'Bet' in df.columns:
        df["PotOdds"] = np.where(
            df["Pot"] + df["Bet"] > 0,
            df["Bet"] / (df["Pot"] + df["Bet"]),
            0
        )
    else:
        df["PotOdds"] = 0
    
    # 5) Предыдущий размер ставки
    if 'Bet' in df.columns:
        df["PreviousBetSize"] = df.groupby("Hand")["Bet"].transform(lambda x: x.shift(1).fillna(0))
    else:
        df["PreviousBetSize"] = 0
    
    # 6) SPR с безопасным делением
    if 'Stack' in df.columns and 'Pot' in df.columns:
        df["SPR"] = np.where(
            df["Pot"] > 0,
            df["Stack"] / df["Pot"],
            0
        )
    else:
        df["SPR"] = 0
    
    # 7) Категоризация стека
    if 'Stack' in df.columns:
        df["StackCategory"] = df["Stack"].apply(categorize_stack)
    else:
        df["StackCategory"] = "Deep"
    
    # 8) Анализ текстуры борда
    card_columns = [f'Card{i}' for i in range(1, 6)]
    if all(col in df.columns for col in card_columns):
        df["BoardTexture"] = df.apply(get_board_texture, axis=1)
    else:
        df["BoardTexture"] = "Neutral"
    
    # 9) Уровень опасности борда
    df["BoardDangerLevel"] = df["BoardTexture"].map({
        "Monotone": "High",
        "Rainbow": "Low",
        "Neutral": "Medium"
    })
    
    # Удаляем временную колонку
    if 'CleanPlayerName' in df.columns:
        df = df.drop('CleanPlayerName', axis=1)
    
    return df

def categorize_stack(stack):
    """Категоризация размера стека."""
    if stack < 10:
        return "Short"
    elif stack < 30:
        return "Mid"
    return "Deep"

def get_board_texture(row):
    """
    Анализ текстуры борда на основе мастей карт.
    Учитываем только непустые карты (Card1..Card5).
    Масть определяется как последний символ ('9h' -> 'h').
    """
    suits = []
    for col in ["Card1", "Card2", "Card3", "Card4", "Card5"]:
        card = row[col]
        if pd.notna(card) and isinstance(card, str) and card != "":
            suits.append(card[-1].lower())
    
    if len(suits) == 0:
        return "Neutral"

    unique_suits = set(suits)
    if len(unique_suits) == 1:
        return "Monotone"
    if len(unique_suits) == len(suits):
        return "Rainbow"
    return "Neutral"

# Пример использования
if __name__ == "__main__":
    # Путь к исходному файлу
    input_path = "/home/tofan/data1/csv/LSTM/data_v2/processed_data/filtered_data.csv"
    
    # Путь для сохранения обработанного файла
    output_dir = os.path.join(os.path.dirname(input_path), 'processed_data')
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"filtered_data_fixed_{timestamp}.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Размер чанка (настраивается в зависимости от доступной памяти)
    chunk_size = 1000000
    
    # Запускаем обработку
    process_large_file(input_path, output_path, chunk_size)