import pandas as pd  
import mlflow

from sklearn.neural_network import MLPClassifier

def train_model():
    clf = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    
    x = pd.read_csv('/home/harshini/my_airflow_home/fin/x_train.csv').iloc[:, 1:]
    y = pd.read_csv('/home/harshini/my_airflow_home/fin/y_train.csv').iloc[:, 1]

    

    clf.fit(x,y)

    with mlflow.start_run():
        mlflow.set_tracking_uri('http://127.0.0.1:5000')
        mlflow.log_param('hidden_layer_sizes', (150, 100, 50))
        mlflow.log_param('max_iter', 300)
        mlflow.log_param('activation', 'relu')
        mlflow.log_param('solver', 'adam')
        mlflow.sklearn.log_model(clf, 'MLP1')
    
    print("Task Complete!")

train_model()





