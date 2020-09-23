""" SEGUNDO MODELO TEORICO DE KAGGLE """

### METODO DE RANDOM FORESTS ###

# Importar librerías
import pandas as pd
from sklearn.ensemble import RandomForestRegressor    # Metodo de Random Forests
from sklearn.metrics import mean_absolute_error       # Calculo del error promedio absoluto
from sklearn.model_selection import train_test_split  # Permite separar los datos en subconjuntos de entrenamiento y evaluacion


### LECTURA DEL ARCHIVO ###

# Camino al archivo
melbourne_file_path = '.\melb\melb_data.csv'

# Almacenar los datos en un Data Frame
melbourne_data = pd.read_csv(melbourne_file_path) 


### LIMPIANDO LOS DATOS ###

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)


### TOMANDO DATOS EN ESPECIFICO ###

# Queremos establecer un objetivo de predicción, conocido como "y". En este caso será el precio.
y = melbourne_data.Price

# Ahora escogemos que Features usar para predecir nuestra y.
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude', 'YearBuilt']

# Convencionalmente a este se le conoce como el vector "x"
X = melbourne_data[melbourne_features]

# Separandolos en datos de entrenamiento y evaluacion
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) # Separacion en train y evaluation


### IMPLEMENTACION DE UN MODELO ###

# Definir un modelo.
melbourne_model = RandomForestRegressor(random_state=1)

# Fit el modelo
melbourne_model.fit(train_X, train_y)


### REALIZAR PREDICCIONES ###

melb_preds = melbourne_model.predict(val_X)


### MOSTRAR EL ERROR ###

print("El Error Medio Absoluto es de: " + str(mean_absolute_error(val_y, melb_preds)))


