"""API do classificador acadêmico de doença cardíaca."""
from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

CAMINHO_MODELO = Path(__file__).resolve().parent.parent / 'training/modelo.joblib'
if not CAMINHO_MODELO.exists():
    raise RuntimeError('Execute: py training/train_model.py')
modelo = joblib.load(CAMINHO_MODELO)

app = FastAPI(title='Heart Risk Triage', version='4.0.0')
app.add_middleware(CORSMiddleware, allow_origins=['*'], allow_methods=['GET', 'POST'], allow_headers=['*'])


class DadosPaciente(BaseModel):
    Age: int = Field(ge=1, le=120)
    Sex: Literal['M', 'F']
    ChestPainType: Literal['TA', 'ATA', 'NAP', 'ASY']
    RestingBP: float | None = Field(default=None, gt=0, le=300)
    Cholesterol: float | None = Field(default=None, gt=0, le=1000)
    FastingBS: Literal[0, 1]
    RestingECG: Literal['Normal', 'ST', 'LVH']
    MaxHR: int = Field(ge=20, le=250)
    Oldpeak: float = Field(ge=-10, le=10)
    ExerciseAngina: Literal['Y', 'N']
    ST_Slope: Literal['Up', 'Flat', 'Down']
    model_config = {'extra': 'forbid', 'allow_inf_nan': False}


@app.get('/')
def saude():
    return {'status': 'ok'}


@app.post('/prever')
def prever(paciente: DadosPaciente):
    valores = paciente.model_dump()
    for campo in ['RestingBP', 'Cholesterol']:
        if valores[campo] is None:
            valores[campo] = np.nan
    tabela = pd.DataFrame([valores])
    classe = int(modelo.predict(tabela)[0])
    probabilidade = float(modelo.predict_proba(tabela)[0][list(modelo.classes_).index(1)])
    return {
        'classe_risco': classe,
        'rotulo_risco': 'Doença cardíaca indicada pelo modelo' if classe else 'Doença cardíaca não indicada pelo modelo',
        'probabilidade': round(probabilidade, 3),
        'recomendacao': 'Resultado acadêmico; não confirma nem exclui doença e não substitui avaliação médica.',
    }
