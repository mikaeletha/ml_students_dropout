import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

# Função para perguntar ao usuário a escolha do dataset


def get_dataset_type():
    print("Escolha o tipo de dataset:")
    print("1 - Completo (students_dropout_train.csv)")
    print("2 - Limpo (cm_students_dropout_train.csv)")
    choice = input("Digite sua escolha (1 ou 2): ")

    if choice == "1":
        return "students_dropout_train.csv"
    elif choice == "2":
        return "cm_students_dropout_train.csv"
    else:
        print("Escolha inválida! Carregando o dataset completo por padrão.")
        return "students_dropout_train.csv"


# Obter o nome do arquivo com base na escolha do usuário
train_file = get_dataset_type()
file_path = f"pre_processed/{train_file}"

# Hiperparâmetros para o GridSearch
param = [
    {
        'n_estimators': list(np.arange(290, 305, 5)),
        'max_features': ["sqrt", "log2", 10, 15]
    },
]

# Inicializar o GridSearchCV
gs = GridSearchCV(
    RandomForestClassifier(),
    param,
    scoring='recall',
    verbose=True
)

# Carregar o dataset
output_var = "Dropout"
dataset = pandas.read_csv(file_path)
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var]

# Ajustar o modelo
gs.fit(x_train, t_train)

# Exibir os melhores parâmetros e score
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
