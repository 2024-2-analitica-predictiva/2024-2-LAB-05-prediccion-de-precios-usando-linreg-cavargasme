#
# En este dataset se desea pronosticar el precio de vhiculos usados. El dataset
# original contiene las siguientes columnas:
#
# - Car_Name: Nombre del vehiculo.
# - Year: Año de fabricación.
# - Selling_Price: Precio de venta.
# - Present_Price: Precio actual.
# - Driven_Kms: Kilometraje recorrido.
# - Fuel_type: Tipo de combustible.
# - Selling_Type: Tipo de vendedor.
# - Transmission: Tipo de transmisión.
# - Owner: Número de propietarios.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# pronostico están descritos a continuación.
#
#
# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

import pandas as pd 
from sklearn.model_selection import GridSearchCV 
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error
import pickle
import numpy as np
import os
import time
import gzip



# Paso 1.
# Preprocese los datos.
# - Cree la columna 'Age' a partir de la columna 'Year'.
#   Asuma que el año actual es 2021.
# - Elimine las columnas 'Year' y 'Car_Name'.
#

def preprocess_data(data):
    df=data.copy()
    df['Age']=2021-df['Year']
    df.drop(columns=['Year','Car_Name'],inplace=True)
    return df

#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#

def get_features_target(data, target_column):
    x = data.drop(columns=target_column)
    y = data[target_column]
    return x, y

#
# Paso 3.
# Cree un pipeline para el modelo de regresión. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Escala las variables numéricas al intervalo [0, 1].
# - Selecciona las K mejores entradas.
# - Ajusta un modelo de regresion lineal.
#

def make_pipeline(df):
    categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
    numerical_features = [col for col in df.columns if col not in categorical_features]

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('feature_selection', SelectKBest(score_func=f_regression)),
            ('regressor', LinearRegression())
        ]
    )

    return pipeline
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use el error medio absoluto
# para medir el desempeño modelo.
#

def optimize_hyperparameters(pipeline, x_train, y_train):
    param_grid = {
        'feature_selection__k': [i for i in range(1, 12)]
    }

    model = GridSearchCV(pipeline, param_grid, cv=10, scoring='neg_mean_absolute_error', n_jobs=-1, verbose=1)
    model.fit(x_train, y_train)

    return model


#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#

def save_model(model):
    if not os.path.exists('../files/models'):
        os.makedirs('../files/models')
    with gzip.open('../files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)


# Paso 6.
# Calcule las metricas r2, error cuadratico medio, y error absoluto medio
# para los conjuntos de entrenamiento y prueba. Guardelas en el archivo
# files/output/metrics.json. Cada fila del archivo es un diccionario con
# las metricas de un modelo. Este diccionario tiene un campo para indicar
# si es el conjunto de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'metrics', 'dataset': 'train', 'r2': 0.8, 'mse': 0.7, 'mad': 0.9}
# {'type': 'metrics', 'dataset': 'test', 'r2': 0.7, 'mse': 0.6, 'mad': 0.8}
#

def calculate_metrics(model, x_train, y_train, x_test, y_test):
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)

    metrics_train = {
        'type': 'metrics',
        'dataset': 'train',
        'r2': float(r2_score(y_train, y_train_pred)),
        'mse': float(mean_squared_error(y_train, y_train_pred)),
        'mad': float(median_absolute_error(y_train, y_train_pred))
    }

    metrics_test = {
        'type': 'metrics',
        'dataset': 'test',
        'r2': float(r2_score(y_test, y_test_pred)),
        'mse': float(mean_squared_error(y_test, y_test_pred)),
        'mad': float(median_absolute_error(y_test, y_test_pred))
    }

    return metrics_train, metrics_test

if __name__ == '__main__':
    
    train_data_zip = '../files/input/train_data.csv.zip'
    test_data_zip = '../files/input/test_data.csv.zip'

    train_data=pd.read_csv(
        train_data_zip,
        index_col=False,
        compression='zip')

    test_data=pd.read_csv(
        test_data_zip,
        index_col=False,
        compression='zip')
    
    train_data=preprocess_data(train_data)
    test_data=preprocess_data(test_data)

    x_train, y_train = get_features_target(train_data, 'Present_Price')
    x_test, y_test = get_features_target(test_data, 'Present_Price')

    pipeline = make_pipeline(x_train)

    start = time.time()
    model = optimize_hyperparameters(pipeline, x_train, y_train)
    end = time.time()

    save_model(model)

    metrics_train, metrics_test = calculate_metrics(model, x_train, y_train, x_test, y_test)

    if not os.path.exists('../files/output'):
        os.makedirs('../files/output')

    metrics = [metrics_train, metrics_test]
    pd.DataFrame(metrics).to_json('../files/output/metrics.json', orient='records', lines=True)