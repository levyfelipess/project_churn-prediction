import numpy as np
import pandas as pd
import joblib

# Retorna uma função que realça as células que contém os n-ésimos valores máximos
# Utilizar com Stylers de exibição do pandas
def highlight_nthmax(nth_max=1):
    def highlight_max(s, props=""):
        a = np.array([''] * np.size(s), dtype='object')
        a[np.argsort(s.values)[-nth_max]] = props
        return a
    return highlight_max

def save_models(models_dict, path='../models/', format='.pkl'):
    for key in models_dict.keys():
        joblib.dump(models_dict[key], path + key + format)

def load_models(models_names, path='../models/', format='.pkl'):
    models = {}
    for name in models_names:
        models[name] = joblib.load(path + name + format)
    return models
