"""
data_preparation.py - Функции для подготовки покерных данных
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# ---------------------- Вспомогательные функции ----------------------
def check_and_remove_column_duplicates(df, operation_name="операция"):
    """
    Проверяет наличие дубликатов столбцов и удаляет их при необходимости
    
    Args:
        df (pandas.DataFrame): Исходный датафрейм
        operation_name (str): Название операции для вывода в сообщениях
        
    Returns:
        pandas.DataFrame: Датафрейм с удаленными дубликатами столбцов
    """
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        print(f"Внимание: обнаружены дублирующиеся столбцы перед {operation_name}: {duplicate_columns}")
        df_clean = df.loc[:, ~df.columns.duplicated()]
        print(f"Дублирующиеся столбцы удалены, оставлены только первые экземпляры")
        return df_clean
    
    return df

# ---------------------- Классы и функции для подготовки данных ----------------------
class PokerDataset(Dataset):
    def __init__(self, features, targets):
        # Убедимся, что все данные числовые
        self.features = torch.tensor(features.astype(np.float32).values, dtype=torch.float32)
        self.targets = torch.tensor(targets.values, dtype=torch.long)
        
    def __len__(self):
        return len(self.features)
        
    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


def balance_classes_by_reducing_fold_preflop(df, target_proportion=None):
    """
    Сбалансировать классы путем удаления случайных строк с Action='Fold' на Street_id=0 (префлоп).
    
    Args:
        df (pandas.DataFrame): Исходный датафрейм с покерными данными.
        target_proportion (float, optional): Целевая доля класса Fold. 
                                           Если None, берется максимальная доля среди других классов.
    
    Returns:
        pandas.DataFrame: Сбалансированный датафрейм.
    """
    print("Начало балансировки классов...")
    # Копируем датафрейм, чтобы не изменять исходные данные
    df_balanced = df.copy()
    
    # Проверяем наличие необходимых столбцов
    if 'Action' not in df_balanced.columns or 'Street_id' not in df_balanced.columns:
        print("Ошибка: отсутствуют необходимые столбцы 'Action' или 'Street_id'")
        return df_balanced
    
    # Проверяем количество строк до балансировки
    original_count = len(df_balanced)
    print(f"Исходное количество строк: {original_count}")
    
    # Подсчитываем количество экземпляров каждого действия
    action_counts = df_balanced['Action'].value_counts()
    print("\nРаспределение классов до балансировки:")
    for action, count in action_counts.items():
        print(f"  {action}: {count} ({count/original_count*100:.1f}%)")
    
    # Если Fold нет в данных, возвращаем исходный датафрейм
    if 'Fold' not in action_counts:
        print("Действие Fold отсутствует в данных, балансировка не требуется")
        return df_balanced
    
    # Определяем целевую долю, если не задана явно
    if target_proportion is None:
        # Находим максимальную долю среди других классов (кроме Fold)
        other_actions_counts = {k: v for k, v in action_counts.items() if k != 'Fold'}
        if not other_actions_counts:
            print("Других действий кроме Fold не найдено, балансировка невозможна")
            return df_balanced
        
        # Находим действие с максимальным количеством (кроме Fold)
        max_other_action = max(other_actions_counts.items(), key=lambda x: x[1])
        target_proportion = max_other_action[1] / original_count
        print(f"\nЦелевая доля для Fold выбрана на основе {max_other_action[0]}: {target_proportion:.3f}")
    
    # Рассчитываем, сколько строк с Fold нужно оставить
    fold_count = action_counts['Fold']
    target_fold_count = int(original_count * target_proportion)
    
    # Если текущее количество Fold меньше целевого, ничего не делаем
    if fold_count <= target_fold_count:
        print("Текущее количество Fold уже меньше или равно целевому, балансировка не требуется")
        return df_balanced
    
    # Определяем количество строк с Fold на префлопе
    fold_preflop_mask = (df_balanced['Action'] == 'Fold') & (df_balanced['Street_id'] == 0)
    fold_preflop_count = fold_preflop_mask.sum()
    print(f"Найдено {fold_preflop_count} строк с Fold на префлопе")
    
    # Если нет Fold на префлопе, берем Fold с любой улицы
    if fold_preflop_count == 0:
        print("Fold на префлопе не найдены, используем Fold с любой улицы")
        fold_preflop_mask = (df_balanced['Action'] == 'Fold')
        fold_preflop_count = fold_preflop_mask.sum()
    
    # Определяем количество строк для удаления
    rows_to_remove = fold_count - target_fold_count
    
    # Если нужно удалить больше строк, чем есть Fold на префлопе, удаляем столько, сколько есть
    if rows_to_remove > fold_preflop_count:
        rows_to_remove = fold_preflop_count
        print(f"Внимание: можем удалить только {fold_preflop_count} строк с Fold на префлопе")
    
    # Отбираем индексы строк для удаления
    fold_preflop_indices = df_balanced[fold_preflop_mask].index
    indices_to_remove = np.random.choice(fold_preflop_indices, size=rows_to_remove, replace=False)
    
    # Удаляем отобранные строки
    df_balanced = df_balanced.drop(indices_to_remove)
    
    # Проверяем результаты балансировки
    new_count = len(df_balanced)
    new_action_counts = df_balanced['Action'].value_counts()
    
    print(f"\nУдалено {rows_to_remove} строк с Fold на префлопе")
    print(f"Новое количество строк: {new_count}")
    print("\nРаспределение классов после балансировки:")
    for action, count in new_action_counts.items():
        print(f"  {action}: {count} ({count/new_count*100:.1f}%)")
    
    return df_balanced


def add_sequence_features(df):
    """
    Добавляет признаки, отражающие последовательность действий в руке
    
    Args:
        df (pandas.DataFrame): Исходный датафрейм
        
    Returns:
        pandas.DataFrame: Датафрейм с дополнительными признаками
    """
    # Создаем копию датафрейма, чтобы избежать изменения исходных данных
    df_copy = df.copy()
    
    # Проверка наличия необходимых столбцов
    required_columns = ['Hand', 'Street_id', 'ActionOrder']
    missing_columns = [col for col in required_columns if col not in df_copy.columns]
    if missing_columns:
        print(f"Внимание: отсутствуют необходимые столбцы для add_sequence_features: {missing_columns}")
        print("Пропуск добавления последовательных признаков")
        return df_copy
    
    # Проверка дубликатов столбцов перед сортировкой
    duplicate_columns = df_copy.columns[df_copy.columns.duplicated()].tolist()
    if duplicate_columns:
        print(f"Внимание: обнаружены дублирующиеся столбцы в add_sequence_features: {duplicate_columns}")
        # Удаляем дубликаты, оставляя первый экземпляр каждого столбца
        df_copy = df_copy.loc[:, ~df_copy.columns.duplicated()]
        print("Дублирующиеся столбцы удалены, оставлены только первые экземпляры")
    
    try:
        # Сортируем по руке, улице и порядку действий
        df_copy = df_copy.sort_values(['Hand', 'Street_id', 'ActionOrder'])
        
        # Здесь можно добавить логику для создания последовательных признаков
        # Например, подсчет действий каждого типа внутри руки или улицы

        return df_copy
    except Exception as e:
        print(f"Ошибка при добавлении последовательных признаков: {e}")
        print("Возвращается датафрейм без дополнительных признаков")
        return df_copy


def verify_class_balance(y_train, action_mapping):
    """
    Проверяет баланс классов в тренировочной выборке
    
    Args:
        y_train: Метки тренировочной выборки
        action_mapping: Отображение действий в числовые метки
    """
    train_counts = pd.Series(y_train).value_counts(normalize=True)
    
    print("\nРаспределение классов в тренировочной выборке:")
    reverse_mapping = {v: k for k, v in action_mapping.items()}
    
    for class_id in sorted(action_mapping.values()):
        action_name = reverse_mapping[class_id]
        train_pct = train_counts.get(class_id, 0) * 100
        print(f"  {action_name}: {train_pct:.1f}%")


def load_and_prepare_data(file_path, max_rows=None, balance_classes=True, target_fold_proportion=None):
    """
    Загрузка и подготовка покерных данных
    
    Args:
        file_path (str): Путь к CSV-файлу с покерными данными
        max_rows (int, optional): Максимальное количество строк для загрузки
        balance_classes (bool): Выполнять ли балансировку классов
        target_fold_proportion (float, optional): Целевая доля класса Fold
        
    Returns:
        dict: Словарь с подготовленными данными и метаинформацией
    """
    print(f"Загрузка данных из {file_path}...")
    
    # Загрузка данных
    if max_rows:
        df = pd.read_csv(file_path, nrows=max_rows)
        df = check_and_remove_column_duplicates(df, "начальной обработкой")
    else:
        df = pd.read_csv(file_path)
    
    print(f"Загружено {len(df)} строк данных")
    
    # Проверка дубликатов столбцов
    duplicate_columns = df.columns[df.columns.duplicated()].tolist()
    if duplicate_columns:
        print(f"Внимание: обнаружены дублирующиеся столбцы: {duplicate_columns}")
        # Переименовываем дублирующиеся столбцы
        for col in duplicate_columns:
            col_indices = [i for i, c in enumerate(df.columns) if c == col]
            for i, idx in enumerate(col_indices[1:], 1):
                new_col_name = f"{col}_{i}"
                df.columns.values[idx] = new_col_name
                print(f"Переименован столбец '{col}' (индекс {idx}) в '{new_col_name}'")
 
    # Проверка наличия необходимых идентификаторов
    required_ids = ['TournamentNumber', 'Hand', 'Street_id', 'ActionOrder']
    missing_ids = [id_field for id_field in required_ids if id_field not in df.columns]
    
    if missing_ids:
        print(f"Внимание: отсутствуют важные идентификаторы: {missing_ids}")
        
        # Обработка отсутствующих идентификаторов
        if 'TournamentNumber' not in df.columns and 'Hand' not in df.columns:
            print("Критическая ошибка: отсутствуют идентификаторы турнира и раздачи!")
            print("Пытаемся создать суррогатные идентификаторы...")
            
            # Используем доступные поля для группировки данных
            grouping_fields = []
            for field in ['Table_id', 'Game_id', 'Date', 'Timestamp']:
                if field in df.columns:
                    grouping_fields.append(field)
            
            if grouping_fields:
                if 'Hand' not in df.columns:
                    print(f"Создаем суррогатный Hand на основе полей: {grouping_fields}")
                    df['Hand'] = df.groupby(grouping_fields).ngroup()
                
                if 'TournamentNumber' not in df.columns and len(grouping_fields) > 1:
                    # Используем подмножество полей для турнира
                    tournament_fields = grouping_fields[:len(grouping_fields)//2]
                    print(f"Создаем суррогатный TournamentNumber на основе полей: {tournament_fields}")
                    df['TournamentNumber'] = df.groupby(tournament_fields).ngroup()
            else:
                print("Невозможно создать суррогатные идентификаторы: недостаточно информации")
                # В крайнем случае, можем использовать индексы строк
                if 'Hand' not in df.columns:
                    df['Hand'] = (df.index // 10)  # Каждые 10 строк - новая рука (просто пример)
                if 'TournamentNumber' not in df.columns:
                    df['TournamentNumber'] = 0  # Предполагаем один турнир
        
        # Проверка и создание Street_id (улицы)
        if 'Street_id' not in df.columns:
            if 'Street' in df.columns:
                # Отображение текстовых названий улиц в числовые идентификаторы
                street_mapping = {'preflop': 0, 'flop': 1, 'turn': 2, 'river': 3}
                df['Street_id'] = df['Street'].map(street_mapping)
                print("Создан Street_id на основе поля Street")
            else:
                print("Внимание: невозможно определить улицу (Street_id)")
                df['Street_id'] = 0  # Предполагаем префлоп
        
        # Проверка и создание ActionOrder
        if 'ActionOrder' not in df.columns:
            if all(id_field in df.columns for id_field in ['Hand', 'Street_id']):
                # Создаем порядок действий внутри каждой руки и улицы
                df = df.sort_values(['Hand', 'Street_id'])
                df['ActionOrder'] = df.groupby(['Hand', 'Street_id']).cumcount()
                print("Создан ActionOrder на основе порядка строк внутри каждой руки и улицы")
            else:
                print("Внимание: невозможно определить порядок действий (ActionOrder)")
                df['ActionOrder'] = df.groupby('Hand').cumcount()
    
    # Балансировка классов
    if balance_classes:
        # Используем функцию для балансировки классов
        df = balance_classes_by_reducing_fold_preflop(df, target_proportion=target_fold_proportion)
    
    # Отбор нужных колонок для модели
    features = [
        'Level', 'StackCategory', 'ContinuationTendency', 'Card1', 'Card2', 'Card3','Card4','Card5',
        'Street_id', 'Round', 'Pot', 'Stack', 'SPR', 'PotOdds', 'PreviousBetSize',
        'Position', 'PositionNum', 'VPIP', 'PFR',
        'BetFrequency', 'RaiseFrequency', 'CallFrequency', 'FoldFrequency',
        'BoardTexture', 'BoardDangerLevel','AggressionFactor_Street0', 'AggressionFactor_Street1', 
        'AggressionFactor_Street2','AggressionFactor_Street3',
        'BetSize_mean','RelativePosition', 'TypeBuyIn'
    ]
    
    # Проверка наличия колонок
    available_features = [f for f in features if f in df.columns]
    print(f"Используются признаки: {available_features}")
    
    # Копируем только нужные колонки плюс служебные идентификаторы
    id_columns = ['TournamentNumber', 'Hand', 'Street_id', 'Round', 'ActionOrder']
    df_model = df[available_features + ['Action'] + id_columns].copy()
    
    # Заполнение пропущенных значений в числовых признаках
    numeric_cols = df_model.select_dtypes(include=['number']).columns
    for col in numeric_cols:
        # Заполняем медианой по каждой руке (чтобы не было утечки между руками)
        df_model[col] = df_model.groupby('Hand')[col].transform(
            lambda x: x.fillna(x.median() if not pd.isna(x.median()) else df_model[col].median())
        )
    
    # Заполнение пропущенных значений в категориальных признаках
    categorical_cols = ['Position', 'BoardTexture', 'BoardDangerLevel', 'ActionHistory']
    cat_cols_available = [c for c in categorical_cols if c in df_model.columns]
    
    for col in cat_cols_available:
        df_model[col] = df_model[col].fillna('Unknown')
    
    # Добавление признаков, отражающих последовательность действий
    df_model = check_and_remove_column_duplicates(df_model, "добавлением последовательных признаков")
    df_model = add_sequence_features(df_model)
    
    # Добавляем составной признак Street_Round для более полного представления о структуре игры
    df_model = check_and_remove_column_duplicates(df_model, "созданием Street_Round")
    if 'Round' in df_model.columns and 'Street_id' in df_model.columns:
        # Проверяем наличие дубликатов столбцов перед созданием признака
        if df_model.columns.duplicated().any():
            print("Обнаружены дублирующиеся столбцы перед созданием Street_Round, удаляем их")
            df_model = df_model.loc[:, ~df_model.columns.duplicated()]
        
        try:
            # Создаем составной признак
            df_model['Street_Round'] = df_model['Street_id'].astype(str) + '_' + df_model['Round'].astype(str)
            print("Добавлен составной признак Street_Round")
        except Exception as e:
            print(f"Ошибка при создании признака Street_Round: {e}")
            print("Продолжаем без этого признака")
    
    # Добавляем признак количества агрессивных действий в предыдущих раундах на той же улице
    if 'Round' in df_model.columns:
        try:
            # Сортируем по руке, улице и раунду
            df_model = df_model.sort_values(['Hand', 'Street_id', 'Round', 'ActionOrder'])
            
            # Функция для подсчета агрессивных действий в предыдущих раундах
            def count_previous_rounds_aggression(group):
                aggressive_actions = (group['Action'].isin(['Bet', 'Raise'])).astype(int)
                result = [0] * len(group)
                
                for i in range(1, len(group)):
                    # Если начался новый раунд, добавляем к счетчику агрессивных действий из предыдущего раунда
                    if group.iloc[i]['Round'] > group.iloc[i-1]['Round']:
                        result[i] = result[i-1] + aggressive_actions.iloc[i-1]
                    else:
                        result[i] = result[i-1]
                
                return result
            
            # Применяем функцию к каждой группе (рука, улица)
            df_model['PreviousRoundsAggression'] = df_model.groupby(['Hand', 'Street_id']).apply(
                lambda x: pd.Series(count_previous_rounds_aggression(x), index=x.index)
            ).reset_index(level=[0, 1], drop=True)
            
            print("Добавлен признак PreviousRoundsAggression")
        except Exception as e:
            print(f"Ошибка при создании признака PreviousRoundsAggression: {e}")
    
    # Кодирование целевой переменной
    action_mapping = {
        'Fold': 0, 'Check': 1, 'Call': 2, 'Bet': 3, 'Raise': 4
    }
    df_model['Action_encoded'] = df_model['Action'].map(action_mapping)
    df_model = df_model.dropna(subset=['Action_encoded'])
    
    # Определяем все категориальные (строковые/объектные) столбцы
    obj_columns = df_model.select_dtypes(include=['object']).columns.tolist()
    print(f"Обнаружены строковые столбцы: {obj_columns}")
    
    # Все строковые столбцы, кроме 'Action', считаем категориальными и требуют кодирования
    cat_columns = [col for col in obj_columns if col != 'Action']
    
    # One-hot кодирование категориальных признаков
    df_encoded = pd.get_dummies(df_model, columns=cat_columns, dummy_na=False)
    
    # Проверим, остались ли нечисловые столбцы после кодирования
    remaining_obj_cols = df_encoded.select_dtypes(include=['object']).columns.tolist()
    
    # Исключаем 'Action', так как этот столбец не будет использоваться для обучения
    remaining_obj_cols = [col for col in remaining_obj_cols if col != 'Action']
    
    if remaining_obj_cols:
        print(f"ВНИМАНИЕ: После кодирования остались нечисловые столбцы: {remaining_obj_cols}")
        print("Применяем Label Encoding для этих столбцов")
        
        for col in remaining_obj_cols:
            le = LabelEncoder()
            # Добавляем неизвестную категорию для тестовых данных
            unique_values = df_encoded[col].unique().tolist()
            # Обучаем и трансформируем
            df_encoded[col] = le.fit_transform(df_encoded[col])
            print(f"Столбец {col} закодирован в числа от 0 до {len(unique_values)-1}")
    
    # Разделение данных с учетом иерархии
    
    # Инициализация переменных по умолчанию
    train_tournaments = None
    val_tournaments = None
    
    # Уникальные турниры
    unique_tournaments = df_encoded['TournamentNumber'].unique()
    print(f"Всего уникальных турниров: {len(unique_tournaments)}")

    # Инициализация переменных по умолчанию
    train_tournaments = None
    val_tournaments = None

    # Разделяем на тренировочные и валидационные ТУРНИРЫ
    if len(unique_tournaments) > 1:
        train_tournaments, val_tournaments = train_test_split(
            unique_tournaments, 
            test_size=0.1,
            random_state=42
        )
        
        # Создаем маски на основе идентификаторов турниров
        train_mask = df_encoded['TournamentNumber'].isin(train_tournaments)
        val_mask = df_encoded['TournamentNumber'].isin(val_tournaments)
    else:
        # Запоминаем единственный турнир в обе переменные
        train_tournaments = val_tournaments = unique_tournaments
        
        # Если у нас только один турнир, сразу разделяем по рукам
        print("Только один турнир в данных. Разделение выполняется по рукам.")
        unique_hands = df_encoded['Hand'].unique()
        print(f"Всего уникальных рук: {len(unique_hands)}")
        
        # Если много рук, используем обычное разделение
        if len(unique_hands) > 10:  # Убедимся, что у нас достаточно рук для разделения
            train_hands, val_hands = train_test_split(
                unique_hands, 
                test_size=0.1, 
                random_state=42
            )
            
            train_mask = df_encoded['Hand'].isin(train_hands)
            val_mask = df_encoded['Hand'].isin(val_hands)
        else:
            # Если мало рук, разделяем данные напрямую
            print("Мало рук для разделения по рукам. Выполняем прямое разделение строк.")
            train_indices, val_indices = train_test_split(
                np.arange(len(df_encoded)),
                test_size=0.1,
                random_state=42
            )
            
            train_mask = pd.Series(False, index=df_encoded.index)
            train_mask.iloc[train_indices] = True
            
            val_mask = pd.Series(False, index=df_encoded.index)
            val_mask.iloc[val_indices] = True
    
    # Проверка корректности разделения
    assert train_mask.sum() + val_mask.sum() == len(df_encoded), "Потеряны некоторые примеры при разделении!"
    
    # Разделение признаков и целевой переменной
    id_cols = ['TournamentNumber', 'Hand', 'Street_id', 'Round', 'ActionOrder', 'Action']
    X_train = df_encoded.loc[train_mask].drop(id_cols + ['Action_encoded'], axis=1)
    y_train = df_encoded.loc[train_mask, 'Action_encoded']
    X_val = df_encoded.loc[val_mask].drop(id_cols + ['Action_encoded'], axis=1)
    y_val = df_encoded.loc[val_mask, 'Action_encoded']
    
    # Определяем количество уникальных турниров и рук в каждой выборке
    train_tournaments_count = df_encoded.loc[train_mask, 'TournamentNumber'].nunique()
    val_tournaments_count = df_encoded.loc[val_mask, 'TournamentNumber'].nunique()
    train_hands_count = df_encoded.loc[train_mask, 'Hand'].nunique()
    val_hands_count = df_encoded.loc[val_mask, 'Hand'].nunique()
    
    print(f"Тренировочная выборка: {len(X_train)} примеров из {train_hands_count} рук в {train_tournaments_count} турнирах")
    print(f"Валидационная выборка: {len(X_val)} примеров из {val_hands_count} рук в {val_tournaments_count} турнирах")
    
    # Проверка на наличие нечисловых столбцов
    non_numeric_cols = X_train.select_dtypes(exclude=['number']).columns.tolist()
    if non_numeric_cols:
        print(f"ВНИМАНИЕ: Обнаружены нечисловые столбцы после кодирования: {non_numeric_cols}")
        print("Эти столбцы будут преобразованы с помощью label encoding")
        
        for col in non_numeric_cols:
            le = LabelEncoder()
            # Создаем копии данных для преобразования
            train_labels = X_train[col].copy().fillna('Unknown')
            val_labels = X_val[col].copy().fillna('Unknown')
            
            # Объединяем для обучения энкодера
            all_labels = pd.concat([train_labels, val_labels])
            le.fit(all_labels)
            
            # Преобразуем данные
            X_train[col] = le.transform(train_labels)
            X_val[col] = le.transform(val_labels)
    
    # Нормализация числовых признаков
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), 
        columns=X_train.columns, 
        index=X_train.index
    )
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val), 
        columns=X_val.columns, 
        index=X_val.index
    )
    
    # Преобразование в float32 для PyTorch
    X_train_float32 = X_train_scaled.astype(np.float32)
    X_val_float32 = X_val_scaled.astype(np.float32)
    
    # Создание датасетов и загрузчиков
    train_dataset = PokerDataset(X_train_float32, y_train)
    val_dataset = PokerDataset(X_val_float32, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    print(f"Данные подготовлены. X_train: {X_train_float32.shape}, y_train: {y_train.shape}")
    
    # Проверка баланса классов
    verify_class_balance(y_train, action_mapping)
    
    # Сохраняем колонки для последующего использования при инференсе
    feature_columns = X_train.columns.tolist()
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'X_train': X_train_float32,
        'X_val': X_val_float32,
        'y_train': y_train,
        'y_val': y_val,
        'action_mapping': action_mapping,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'input_dim': X_train_float32.shape[1],
        'train_tournaments': train_tournaments,
        'val_tournaments': val_tournaments
    }