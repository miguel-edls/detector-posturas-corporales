from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import pandas as pd
from sklearn.model_selection import KFold
from itertools import combinations

scorings = {
    'r2': 'r2',
    'mea': 'neg_mean_absolute_error',
}

def kfold_cv(data, dependent = 'cva', independents = ['smpn', 'nch', 'chth', 'smpth'], rs = 1, degree = 2, model = 'linear', scoring = 'r2'):

    df = data.copy()

    #df = pd.DataFrame(data)

    X = df[independents]
    y = df[dependent]  

    if model == 'linear':
        model = LinearRegression()
    else:
        model = make_pipeline(PolynomialFeatures(degree=degree, include_bias=False), LinearRegression())

    
    #model = LinearRegression()

    
    kf = KFold(n_splits=5, shuffle=True, random_state=rs)

    
    scores = cross_val_score(model, X, y, cv=kf, scoring=scorings[scoring])

    #print(scoring, scores.mean())

    return  scores.mean()

def get_model(data, dependent = 'cva', independents = ['smpth']):
    df = data.copy()

    X = df[independents]
    y = df[dependent]  

    
    model = LinearRegression()

    model.fit(X, y)

    return model

#Esta funcion obtiene los r2 scores de todas las combinaciones posibles de las variables independientes
def get_all_scores(data, dependent = 'cva', independents = ['smpn', 'nch', 'chth', 'smpth'], show_top = 4, model = 'linear'):
    scores = []
    data = data.copy()
    if independents == "all":
        independents = list(data[0].keys())
        independents.remove(dependent)
        independents.remove("name")

    for i in range(len(independents)):
        for elem in combinations(independents, i+1):
            r2 = kfold_cv(data, dependent, list(elem), model= model)
            print(f"Dependent: {dependent} / Independents: {elem} / R2: {r2.mean()}")
            scores.append([elem, r2.mean()])
    scores = sorted(scores, key = lambda x: x[1], reverse = True)
    if model == 'linear':
        model = 'Linear Regression'
    else:
        model = f'Polynomial Regression (degree = 2)'
    
    #print(f"Model: {model}")
    for i in range(show_top):
        print(f"Dependent: {dependent} / Independents: {scores[i][0]} / R2: {scores[i][1]}")

