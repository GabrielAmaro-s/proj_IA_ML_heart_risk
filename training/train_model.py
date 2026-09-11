"""Treina KNN e árvore de decisão e salva somente o vencedor."""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent.parent
NUMERICAS = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
CATEGORICAS = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope']


def carregar_dados():
    dados = pd.read_csv(ROOT / 'data/heart.csv').drop_duplicates()
    if dados['HeartDisease'].isna().any() or set(dados['HeartDisease'].unique()) != {0, 1}:
        raise ValueError('HeartDisease deve conter somente 0 e 1, sem ausências.')
    # Zero nesses campos representa uma medida não disponível, não um valor real.
    dados[['RestingBP', 'Cholesterol']] = dados[['RestingBP', 'Cholesterol']].replace(0, np.nan)
    return dados[NUMERICAS + CATEGORICAS], dados['HeartDisease']


def montar_pipeline(classificador):
    numericas = Pipeline([
        ('imputacao', SimpleImputer(strategy='median')),
        ('escala', MinMaxScaler()),
    ])
    preprocessador = ColumnTransformer([
        ('numericas', numericas, NUMERICAS),
        ('categoricas', OneHotEncoder(handle_unknown='ignore'), CATEGORICAS),
    ])
    return Pipeline([('preprocessador', preprocessador), ('modelo', classificador)])


def main():
    X, y = carregar_dados()
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    candidatos = {
        'KNN': (KNeighborsClassifier(), {
            'modelo__n_neighbors': [3, 5, 7, 11, 15],
            'modelo__weights': ['uniform', 'distance'],
        }),
        'Árvore de Decisão': (DecisionTreeClassifier(random_state=42), {
            'modelo__max_depth': [3, 5, 8, None],
            'modelo__min_samples_leaf': [1, 5, 10],
        }),
    }
    buscas = {}
    for nome, (modelo, parametros) in candidatos.items():
        busca = GridSearchCV(montar_pipeline(modelo), parametros, cv=cv,
                            scoring='balanced_accuracy', n_jobs=2, error_score='raise')
        busca.fit(Xtr, ytr)
        buscas[nome] = busca

    # Escolhe na validação; o teste não participa da escolha do vencedor.
    vencedor = max(buscas, key=lambda nome: buscas[nome].best_score_)
    print(f'Registros: {len(X)} | Treino: {len(Xtr)} | Teste: {len(Xte)}')
    print(f'{"Modelo":<20} {"CV balanceada":>14} {"Acurácia teste":>15} {"Recall teste":>13}')
    for nome, busca in buscas.items():
        previsto = busca.predict(Xte)
        print(f'{nome:<20} {busca.best_score_:>14.1%} {accuracy_score(yte, previsto):>15.1%} {recall_score(yte, previsto):>13.1%}')
        print(f'  Parâmetros: {busca.best_params_}')
        print(f'  Acurácia balanceada no teste: {balanced_accuracy_score(yte, previsto):.1%}')
    joblib.dump(buscas[vencedor].best_estimator_, ROOT / 'training/modelo.joblib')
    print(f'Salvo: {vencedor} -> training/modelo.joblib')


if __name__ == '__main__':
    main()
