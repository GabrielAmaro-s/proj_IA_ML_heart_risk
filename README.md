# Heart Risk Triage

Projeto acadêmico simples: compara **KNN e Árvore de Decisão** para classificar presença de doença cardíaca.

## Executar

```powershell
py -m pip install -r requirements.txt
py training/train_model.py
py -m uvicorn api.main:app --reload
```

Abra `frontend/index.html`. Documentação da API: http://localhost:8000/docs.

## Arquivos

- `data/heart.csv`: 918 registros e 11 atributos.
- `training/train_model.py`: prepara dados, compara modelos e salva o vencedor.
- `training/modelo.joblib`: único arquivo gerado pelo treinamento.
- `api/main.py`: endpoint `POST /prever`.
- `frontend/index.html`: formulário HTML, CSS e JavaScript.

## Dados e resultados

Fonte: [Heart Failure Prediction Dataset — fedesoriano](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).
Alvo `HeartDisease`: 1 = doença; 0 = ausência no rótulo. Não prevê infarto futuro.

Zeros em pressão e colesterol são tratados como ausências, preenchidas pela mediana aprendida dentro de cada treino. Na interface esses campos podem ficar vazios. Oldpeak negativo é preservado.

Divisão estratificada 80/20, semente 42: 734 registros de treino e 184 de teste. Pipeline com imputação, MinMaxScaler e OneHotEncoder. Busca com validação cruzada de 5 folds; vence a maior acurácia balanceada na validação, sem escolher pelo teste.

| Modelo | Acurácia balanceada na validação | Acurácia no teste | Recall no teste |
|---|---:|---:|---:|
| **KNN** | **85,4%** | **89,7%** | **92,2%** |
| Árvore | 82,9% | 82,1% | 79,4% |

KNN: 15 vizinhos e pesos por distância. Árvore: profundidade 3 e mínimo de 5 registros por folha. O modelo salvo permanece ajustado somente no treino. Métricas aparecem no terminal, sem relatórios adicionais.

Resultados acadêmicos deste split, sem validação clínica. O escore não é uma probabilidade clínica calibrada.
