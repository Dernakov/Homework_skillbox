import os
import json
import pandas as pd
import dill
from datetime import datetime


class TestDataProcessor:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.test_files = [f for f in os.listdir(data_dir) if f.endswith('.json')]

    def extract_features(self, data):
        """
        Извлекаем нужные признаки из данных
        """
        features = {
            'model': data.get('model', ''),
            'year': data.get('year', 0),
            'odometer': data.get('odometer', 0),
            'fuel': data.get('fuel', ''),
            'price': data.get('price', 0),
            'transmission': data.get('transmission', ''),
        }
        return features

    def read_test_data(self):
        """
        Чтение всех тестовых файлов из папки и преобразование их в DataFrame
        """
        if not self.test_files:
            print(f"No test files found in {self.data_dir}")
            return None

        data_frames = []
        for file in self.test_files:
            file_path = os.path.join(self.data_dir, file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Чтение данных как JSON
                    data = json.load(f)
                    print(f"Reading file: {file}")

                    # Преобразуем данные в признаки
                    features = self.extract_features(data)
                    data_frames.append(features)
            except ValueError as e:
                print(f"Error reading {file}: {e}")
            except Exception as e:
                print(f"Unexpected error reading {file}: {e}")

        if not data_frames:
            print("No valid test data found.")
            return None

        return pd.DataFrame(data_frames)


# Функция для загрузки модели
def load_model(path):
    model_files = [f for f in os.listdir(path) if f.endswith('.pkl')]
    if not model_files:
        print("No model files found.")
        return None

    latest_model = max(model_files, key=lambda x: os.path.getmtime(os.path.join(path, x)))
    model_path = os.path.join(path, latest_model)

    with open(model_path, 'rb') as file:
        model = dill.load(file)

    print(f"Loaded model: {latest_model}")
    return model


# Функция для сохранения предсказаний в CSV
def save_predictions(predictions, predictions_path):
    output_filename = f"predictions_{datetime.now().strftime('%Y%m%d%H%M')}.csv"
    predictions_filepath = os.path.join(predictions_path, output_filename)

    predictions.to_csv(predictions_filepath, index=False)
    print(f"Predictions saved to {predictions_filepath}")


# Основная функция для выполнения предсказаний
def predict():
    # Путь к проекту
    path = os.environ.get('PROJECT_PATH', '.')

    # Папка с тестовыми данными и папка для сохранения предсказаний
    test_data_path = os.path.join(path, 'data', 'test')
    predictions_path = os.path.join(path, 'data', 'predictions')

    # Загрузка модели
    model = load_model(os.path.join(path, 'data', 'models'))
    if not model:
        print("No model loaded. Exiting.")
        return

    # Обработка тестовых данных с использованием класса TestDataProcessor
    processor = TestDataProcessor(test_data_path)
    test_data = processor.read_test_data()

    if test_data is None:
        print("No test data to predict on.")
        return

    # Проверяем, что данные имеют правильный формат
    print(f"Test data shape: {test_data.shape}")

    # Применение модели для предсказания
    predictions = model.predict(test_data)

    # Создание DataFrame с предсказаниями
    predictions_df = pd.DataFrame(predictions, columns=['predicted_price_category'])

    # Сохранение предсказаний в CSV
    save_predictions(predictions_df, predictions_path)


if __name__ == '__main__':
    predict()
