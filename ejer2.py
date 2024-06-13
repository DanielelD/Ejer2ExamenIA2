# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 01:13:09 2024

@author: ddiaz
"""
import numpy as np

def funcion_escalon(x):
    return np.where(x >= 0, 1, 0)

np.random.seed(42)

pesos_en = np.random.randn(2, 2)
bias_o= np.random.randn(2)
pesos_os = np.random.randn(2, 1)
bias_s = np.random.randn(1)

print("Pesos entrada-oculta:\n", pesos_en)
print("Bias oculta:\n", bias_o)
print("Pesos oculta-salida:\n", pesos_os)
print("Bias salida:\n", bias_s)

def prop(X):
    z_oculta = np.dot(X, pesos_en) + bias_o
    activacion_o = funcion_escalon(z_oculta)
    
    z_salida = np.dot(activacion_o, pesos_os) + bias_s
    salida = funcion_escalon(z_salida)
    
    return activacion_o, salida

def act_pesos(X, y, activacion_o, salida, tasa_aprendizaje=0.2):
    global pesos_en, bias_o, pesos_os, bias_s
    
    error_s = y - salida
    
    delta_salida = error_s
    pesos_os += tasa_aprendizaje * np.dot(activacion_o.T, delta_salida)
    bias_s += tasa_aprendizaje * np.sum(delta_salida, axis=0)
    
    error_o = np.dot(delta_salida, pesos_os.T)
    
   
    delta_o = error_o * (activacion_o * (1 - activacion_o)) 
    pesos_en += tasa_aprendizaje * np.dot(X.T, delta_o)
    bias_o += tasa_aprendizaje * np.sum(delta_o, axis=0)

def entrenar(X, y, epocas=1000, tasa_a=0.2):
    for _ in range(epocas):
        act_oculta, salida = prop(X)
        act_pesos(X, y, act_oculta, salida, tasa_a)

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]]) 

entrenar(X, y)

_, salida_final = prop(X)
print("Salida final:\n", salida_final)

