import pandas
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC


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

# Configuração dos parâmetros para GridSearchCV
param = [
    {
        # Valores de C entre 0.25 e 0.40 com incrementos de 0.05
        'C': list(np.arange(0.25, 0.40, 0.05)),
        'kernel': ['linear', 'poly', 'sigmoid', 'rbf']  # Tipos de kernel
    },
]

# Configuração do modelo SVC com ajuste para classes desbalanceadas
gs = GridSearchCV(
    SVC(class_weight='balanced'),
    param,
    scoring='recall',  # Usando recall como métrica de avaliação
    verbose=True
)

# Carregar o dataset escolhido
dataset = pandas.read_csv(file_path)
x_train = dataset.drop(columns=['Dropout'])  # Features
t_train = dataset['Dropout']  # Target

# Treinar o modelo usando GridSearch
gs.fit(x_train, t_train)

# Exibir os melhores parâmetros e o melhor score
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
