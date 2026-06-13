# 🎓 Graduate Predictor — FT-Transformer vs MLP

Demo en Streamlit para predecir si un estudiante se graduará (`graduate`) a partir de datos de aprendizaje en línea (IEEE Online Learning dataset). Permite entrenar y comparar dos modelos: un **FT-Transformer** (implementado en PyTorch puro) y un **MLP base**.

## Características

- Carga de CSV (`ieee_online_learning_balanced.csv`)
- Análisis exploratorio: distribución del target, boxplots, correlaciones
- Preprocesamiento: LabelEncoding de categóricas, escalado de continuas, split 70/15/15
- Entrenamiento configurable (épocas, batch size, embed dim, heads, bloques, learning rate)
- Métricas: Accuracy, Precision, Recall, F1, AUC-ROC
- Visualizaciones: curvas de loss/accuracy, matriz de confusión, ROC, Precision-Recall
- Comparación lado a lado de ambos modelos

## Estructura del proyecto

```
graduate-predictor/
├── app.py            # App principal de Streamlit
├── model_utils.py     # Preprocesamiento, modelos y métricas
├── requirements.txt   # Dependencias
└── README.md
```

## Instalación

```bash
git clone https://github.com/TU_USUARIO/graduate-predictor.git
cd graduate-predictor
pip install -r requirements.txt
```

## Ejecución

```bash
streamlit run app.py
```

Abre `http://localhost:8501`, sube el CSV, ajusta los hiperparámetros en el sidebar y presiona **▶ Iniciar entrenamiento**.

## Dataset esperado

El CSV debe contener al menos las siguientes columnas:

| Tipo          | Columnas                                                                                     |
| ------------- | -------------------------------------------------------------------------------------------- |
| Identificador | `student_id`                                                                                 |
| Categóricas   | `academic_year`, `online_access_count`, `final_grade`, `semester`                            |
| Continuas     | `test_scores`, `project_grades`, `assignment_completion`, `final_points`, `engagement_score` |
| Target        | `graduate` (0/1)                                                                             |

## Tecnologías

- [Streamlit](https://streamlit.io/) — interfaz web
- [PyTorch](https://pytorch.org/) — FT-Transformer y MLP
- [scikit-learn](https://scikit-learn.org/) — preprocesamiento y métricas
- [Matplotlib](https://matplotlib.org/) / [Seaborn](https://seaborn.pydata.org/) — visualizaciones

## Licencia

MIT
