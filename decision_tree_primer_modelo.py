""" MODELO DE ML SIGUIENDO LA TEORÍA DE KAGGLE """

# Importando librerías
import pandas as pd
from sklearn.tree import DecisionTreeRegressor  # Tipo de ML
from sklearn.metrics import mean_absolute_error # Para calcular el error de las predicciones
from sklearn.model_selection import train_test_split  # Permite separar los datos en subconjuntos de entrenamiento y evaluacion

### LECTURA DEL ARCHIVO ###

# save filepath to variable for easier access
melbourne_file_path = '.\melb\melb_data.csv'

# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path) 

### DESCRIPCION DEL ARCHIVO ###

# print a summary of the data in Melbourne data
melbourne_data.describe()

# Observar los nombres las columnas o Features del dataset
print(melbourne_data.columns)

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

# Veamos que contiene la X
print(X.describe())
print(X.head()) # Identico a R
print(X.tail()) # Identico a R

### IMPLEMENTACION DE UN MODELO ###

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
melbourne_model.fit(X, y)

### MOSTRANDO RESULTADOS ###

print("PRIMERA PREDICCION")
print("Realizando predicciones para las siguientes 5 casas:")
print(X.head())
print("Las predicciones son: ")
print(melbourne_model.predict(X.head()))
print("Comparando a los precios reales")
print(y.head().tolist())
print("")

print("SEGUNDA PREDICCION")
print("Realizando predicciones para las siguientes 5 casas:")
print(X.tail())
print("Las predicciones son: ")
print(melbourne_model.predict(X.tail()))
print("Comparando a los precios reales")
print(y.tail().tolist())
print("")

### OBSERVANDO ERROR IN-SAMPLE ###

prediccion_precios = melbourne_model.predict(X)
error_medio_absoluto = mean_absolute_error(y, prediccion_precios)
print("El MAE es de: " + str(error_medio_absoluto))

### REHAGAMOS EL MODELO USANDO BUENAS PRACTICAS ###

"""Esencialmente, de los datos que tenemos crear un subconjunto que sea para el entrenamiento y otro para validacion"""

train_X, val_X, train_y, val_y = train_test_split(X, y, random_state = 0) # Separacion en train y evaluation

melbourne_better_model = DecisionTreeRegressor() # Creamos el modelo

melbourne_better_model.fit(train_X, train_y) # Fit the model

# Obtener predicciones del precio

val_predicciones = melbourne_better_model.predict(val_X)
print("El Error Medio Absoluto es de: " + str(mean_absolute_error(val_y, val_predicciones)))

### COMPARAR LA EFICIENCIA DEL MODELO A DISTINTOS NIVELES DE PROFUNDIDAD ###

# De esta manera podemos ver cuántos nodos (usando este modelo de regresión por árbol) nos dan los mejores resultados

def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    """ Funcion para obtener el error promedio con distintos niveles de nodos """
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# Comparar el Error Promedio con 6 distintos niveles de nodos y devolver el mas apropiado
leaf_nodes_candidates = [5, 50, 500, 1000, 2500, 5000]
maes_obtenidos = []

for max_leaf_nodes in leaf_nodes_candidates:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    maes_obtenidos.append(my_mae)
    print("Máximos nodos de hojas: %d  \t\t Mean Absolute Error:  %d" %(max_leaf_nodes, my_mae))

print("Un menor MAE es mejor.")

# Guardar la posición del mejor índice
mejor_indice = maes_obtenidos.index(min(maes_obtenidos))

print("")
print("Los mejores resultados se obtuvieron con " + str(leaf_nodes_candidates[mejor_indice]) + " nodos.")



