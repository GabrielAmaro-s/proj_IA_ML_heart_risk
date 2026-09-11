# Heart Risk Triage

Projeto acadêmico simples: utiliza **KNN (K-Nearest Neighbors)** para classificar presença de doença cardíaca.

## Executar

```powershell
py -m pip install -r requirements.txt
py training/train_model.py
py -m uvicorn api.main:app --reload
```

Abra `frontend/index.html`. Documentação da API: http://localhost:8000/docs.

## Arquivos

- `data/heart.csv`: 918 registros e 11 atributos.
- `training/train_model.py`: prepara dados, ajusta o KNN e salva o modelo.
- `training/modelo.joblib`: único arquivo gerado pelo treinamento.
- `api/main.py`: endpoint `POST /prever`.
- `frontend/index.html`: formulário HTML, CSS e JavaScript.

## Dados e resultados

Fonte: [Heart Failure Prediction Dataset — fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
Alvo `HeartDisease`: 1 = doença; 0 = ausência no rótulo. Não prevê infarto futuro.

Zeros em pressão e colesterol são tratados como ausências, preenchidas pela mediana aprendida dentro de cada treino. Na interface esses campos podem ficar vazios. Oldpeak negativo é preservado.

Divisão estratificada 80/20, semente 42: 734 registros de treino e 184 de teste. Pipeline com imputação, MinMaxScaler e OneHotEncoder. Busca com validação cruzada de 5 folds; os parâmetros do KNN são escolhidos pela maior acurácia balanceada na validação, sem usar o teste nessa escolha.

| Modelo | Acurácia balanceada na validação | Acurácia no teste | Recall no teste |
|---|---:|---:|---:|
| **KNN** | **85,4%** | **89,7%** | **92,2%** |

O KNN classifica cada exemplo usando os 15 vizinhos mais próximos no conjunto de treino. Os votos são ponderados pela distância: vizinhos mais próximos têm maior influência. O modelo salvo permanece ajustado somente no treino. Métricas aparecem no terminal, sem relatórios adicionais.

Resultados acadêmicos deste split, sem validação clínica. O escore não é uma probabilidade clínica calibrada.

# Perguntas

# Qual dataset e problema?
-- Heart Failure Prediction Dataset, usado para classificar presença de doença cardíaca.

# Qual variável-alvo?	
-- HeartDisease.

# Quais classes?	
-- 0: ausência de doença no rótulo; 1: presença de doença.

# Quais entradas?	
-- 11 atributos, incluindo idade, sexo, pressão, colesterol, dor torácica e resultados de exames.

# Quem utilizaria e para quê?	
-- É um protótipo acadêmico de apoio à análise de dados cardíacos. O usuário (profissional da saúde) informa características clínicas e recebe uma classificação sobre presença de doença cardíaca, acompanhada do escore do modelo. A proposta é organizar uma consulta ao modelo em uma interface simples, como informação complementar à avaliação profissional.

# O que faz com a classificação?	
-- Exibe a classe prevista, um indicador visual e o escore do modelo.

# Como é a interface?	
-- Formulário web que envia os dados à API e apresenta o resultado na própria página.

