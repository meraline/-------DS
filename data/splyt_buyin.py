import pandas as pd
import os
import time

def split_data_by_buyin_type(file_path, output_dir="split_data"):
    """
    Считывает большой файл покерных данных чанками, группирует записи по 'TypeBuyIn'
    и сохраняет каждую группу в отдельный файл.
    
    Args:
        file_path (str): Путь к большому CSV файлу
        output_dir (str): Папка для сохранения результатов
    """
    print(f"Начинаем разделение файла: {file_path}")
    
    # Создаём директорию для сохранения результатов, если её нет
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Создана директория для вывода: {output_dir}")
    
    # Размер чанка - регулируйте в зависимости от доступной памяти
    chunk_size = 500000
    
    # Словарь для хранения DataFrame'ов для каждого типа buy-in
    buyin_dataframes = {}
    
    # Для отслеживания прогресса
    total_rows_processed = 0
    start_time = time.time()
    
    # Получим общее количество строк для оценки прогресса
    print("Подсчёт общего количества строк в файле...")
    total_rows = sum(1 for _ in open(file_path)) - 1  # вычитаем строку заголовка
    print(f"Всего строк в файле: {total_rows:,}")
    
    # Считываем файл по частям
    for chunk_num, chunk in enumerate(pd.read_csv(file_path, chunksize=chunk_size)):
        chunk_start = time.time()
        print(f"Обработка чанка {chunk_num+1}, строки {total_rows_processed+1:,}-{min(total_rows_processed+chunk_size, total_rows):,}...")
        
        # Проверяем наличие столбца 'TypeBuyIn'
        if 'TypeBuyIn' not in chunk.columns:
            print("Ошибка: Столбец 'TypeBuyIn' не найден в данных!")
            print(f"Доступные столбцы: {chunk.columns.tolist()}")
            return
        
        # Группируем данные по типу buy-in
        for buyin_type, group in chunk.groupby('TypeBuyIn'):
            # Преобразуем значение buy-in в строку для использования в имени файла
            buyin_str = str(buyin_type).replace("/", "_").replace("\\", "_").replace(" ", "_")
            
            # Если это первый чанк с таким типом buy-in, инициализируем DataFrame
            if buyin_str not in buyin_dataframes:
                buyin_dataframes[buyin_str] = group
            else:
                # Иначе добавляем данные к существующему DataFrame
                buyin_dataframes[buyin_str] = pd.concat([buyin_dataframes[buyin_str], group])
            
            # Сохраняем промежуточные результаты, если размер DataFrame стал большим
            if len(buyin_dataframes[buyin_str]) > chunk_size * 2:
                output_file = os.path.join(output_dir, f"buyin_type_{buyin_str}.csv")
                mode = 'a' if os.path.exists(output_file) else 'w'
                header = not os.path.exists(output_file)
                
                buyin_dataframes[buyin_str].to_csv(output_file, mode=mode, header=header, index=False)
                print(f"Промежуточное сохранение для типа buy-in '{buyin_str}', строк: {len(buyin_dataframes[buyin_str]):,}")
                
                # Очищаем память
                buyin_dataframes[buyin_str] = pd.DataFrame()
        
        # Обновляем счётчик обработанных строк
        total_rows_processed += len(chunk)
        
        # Выводим информацию о прогрессе
        elapsed = time.time() - start_time
        rows_per_sec = total_rows_processed / elapsed
        est_remaining = (total_rows - total_rows_processed) / rows_per_sec if rows_per_sec > 0 else 0
        
        print(f"Прогресс: {total_rows_processed:,}/{total_rows:,} строк ({total_rows_processed/total_rows*100:.1f}%)")
        print(f"Скорость: {rows_per_sec:.0f} строк/сек")
        print(f"Примерное оставшееся время: {est_remaining/60:.1f} минут")
        print(f"Время обработки чанка: {time.time() - chunk_start:.1f} секунд\n")
    
    # Сохраняем оставшиеся данные
    for buyin_str, df in buyin_dataframes.items():
        if len(df) > 0:
            output_file = os.path.join(output_dir, f"buyin_type_{buyin_str}.csv")
            mode = 'a' if os.path.exists(output_file) else 'w'
            header = not os.path.exists(output_file)
            
            df.to_csv(output_file, mode=mode, header=header, index=False)
            print(f"Финальное сохранение для типа buy-in '{buyin_str}', строк: {len(df):,}")
    
    # Выводим итоговую статистику
    print("\nОбработка завершена!")
    print(f"Всего обработано строк: {total_rows_processed:,}")
    print(f"Общее время выполнения: {(time.time() - start_time) / 60:.1f} минут")
    
    # Собираем информацию о созданных файлах
    print("\nСозданные файлы:")
    total_files = 0
    for filename in os.listdir(output_dir):
        if filename.startswith("buyin_type_") and filename.endswith(".csv"):
            file_path = os.path.join(output_dir, filename)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # в МБ
            
            # Подсчитаем количество строк в файле (быстрый способ)
            with open(file_path, 'r') as f:
                line_count = sum(1 for _ in f) - 1  # вычитаем заголовок
            
            buyin_type = filename.replace("buyin_type_", "").replace(".csv", "")
            print(f"{filename}: {line_count:,} строк, {file_size:.1f} МБ, тип buy-in: {buyin_type}")
            total_files += 1
    
    print(f"\nВсего создано {total_files} файлов в директории {output_dir}")
    return output_dir

# Запуск функции
if __name__ == "__main__":
    file_path = "/home/tofan/data1/csv/LSTM/data_v2/processed_data/filtered_data.csv"
    output_directory = split_data_by_buyin_type(file_path)
    print(f"Результаты сохранены в директории: {output_directory}")