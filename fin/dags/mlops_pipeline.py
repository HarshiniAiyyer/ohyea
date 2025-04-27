from create_model import train_model

from datetime import datetime
from airflow import DAG

from airflow.operators.python_operator import PythonOperator

default_args = {
    'owner' : 'harsh',
    'start_date' : datetime(2025,4,27),
    'catchup' : False
}

dag = DAG('ml_pipeline', default_args = default_args, schedule_interval = None)

train_op = PythonOperator(
    task_id = 'train',
    python_callable = train_model,
    dag = dag
)
