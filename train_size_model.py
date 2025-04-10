
"""
Обучение модели для предсказания размеров ставок
"""
import argparse
from data_preparation_size import prepare_bet_size_data
from train_process import train_poker_model

def train_size_model(file_path, output_dir="./size_model_dir", **kwargs):
    """Обучение модели размеров ставок"""
    print("=== Запуск обучения модели RWKV для предсказания размеров ставок ===")
    
    # Подготавливаем данные
    result = train_poker_model(
        file_path=file_path,
        output_dir=output_dir,
        data_preparation_fn=prepare_bet_size_data,
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
