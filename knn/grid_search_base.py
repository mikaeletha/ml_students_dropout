import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

param = [
    {
        'n_neighbors': range(1, 3),
        'p': [1, 2, 3]
    }
]

file_path = "pre_processed/students_dropout_train.csv"

dataset = pd.read_csv(file_path)
x_train = dataset.drop(columns=['Dropout'])
t_train = dataset['Dropout']

# Inicializar o GridSearchCV
gs = GridSearchCV(
    KNeighborsClassifier(),
    param,
    scoring='recall',
    verbose=True
)

# Rodar o GridSearch nos dados de treino
gs.fit(x_train, t_train)

# Melhor parâmetro encontrado
print(f"Best parameters: ", gs.best_params_)
print(f"Best score: ", gs.best_score_)
