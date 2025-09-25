import os
import sys
import time
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

q = False

def Control_De_Datos(filname: str):
    # = datasets.load_base_proyecto()
    df = pd.read_csv(filname, sep = ",", index_col=0)

    #volvemos matrices los datos que sacamos del dataset

    #construyo la matiz x
    X_generador = df.iloc[:,[2,4]]

    #construyo el vector Y
    y_generador = df.iloc[:,1]

    X, x_descartado, y, y_descartado = train_test_split( X_generador,y_generador, test_size = 0.95, random_state = 0) 
    return X, y        

def Generacion_Val(filname: str):
    h = 0.2 
    c = 1.0

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    X,y = Control_De_Datos(filname)
    #Separar las matrices en una parte para entrenar y otra para povar el modelo
    #Usar los mismos datos es un error metodologico
    x_entrenador, x_provador, y_entrenador, y_provador = train_test_split( X,y, test_size = 0.40, random_state = 0)
    return x_entrenador, x_provador, y_entrenador, y_provador, c

def FRegresioLineal(filname:str):
    x_entrenador, x_provador, y_entrenador, y_provador, c = Generacion_Val(filename)
    lineal_svc = SVC(kernel= "linear", C=c)
    #entrenamos el modelo
    lineal_svc.fit(x_entrenador,y_entrenador)
    return lineal_svc

def FRegresioRBF():
    pass
def FNeurona():
    pass

if __name__ == '__main__':
    sys.argv = 2 
    if sys.argv == 2: 
        filename = "base_proyecto.csv"
        while(q != True):
            regresion =  input("Por favor indicar \nque tipo de entrenamineto de maquina \nquiere(LINEAL,RBF,NEURONA):")  
            if(regresion == "LINEAL"):
                print("La regresion resultante es ")
                print(FRegresioLineal(filename))
            elif(regresion == "RBF"):
                print("adios")
                q = True
            elif(regresion == "NEURONA"):
                print("adios")
                q = True
            else:
                print("Colocaste un entrenamineto no valido")
    else:
        print('Incorrect number of arguments! Usage:')
        print('python Core_file.py <filename>')