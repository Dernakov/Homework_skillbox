import datetime as dt
import os
import sys

from airflow.models import DAG
from airflow.operators.python import PythonOperator

# Указание пути к проекту
path = os.path.expanduser('~/airflow_hw')
os.environ['PROJECT_PATH'] = path
sys.path.insert(0, path)

# Импорт функций
from modules.pipeline import pipeline
from modules.predict import predict  # Добавляем импорт функции predict

args = {
    'owner': 'airflow',
    'start_date': dt.datetime(2022, 6, 10),
    'retries': 1,
    'retry_delay': dt.timedelta(minutes=1),
    'depends_on_past': False,
}

with DAG(
        dag_id='car_price_prediction',
        schedule_interval="00 15 * * *",  # Запускать DAG каждый день в 15:00
        default_args=args,
) as dag:
    # Шаг 1: Pipeline — обучение модели
    train_model = PythonOperator(
        task_id='train_model',
        python_callable=pipeline,  # Вызов функции pipeline, которая обучает модель
    )

    # Шаг 2: Predict — предсказания
    make_predictions = PythonOperator(
        task_id='make_predictions',
        python_callable=predict,  # Вызов функции predict для предсказания
    )

    # Указываем порядок выполнения задач: сначала обучение модели, затем предсказания
    train_model >> make_predictions

