import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

# Parâmetros do GridSearch
param = [
    {
        'n_neighbors': range(1, 3),
        'p': [1, 2, 3]
    }
]

# Função para perguntar ao usuário a escolha do dataset


def get_file_path():
    print("Escolha o tipo de dataset para carregar:")
    print("1 - Completo (students_dropout_train.csv)")
    print("2 - Limpo (cm_students_dropout_train.csv)")
    choice = input("Digite sua escolha (1 ou 2): ")

    if choice == "1":
        return "pre_processed/students_dropout_train.csv"
    elif choice == "2":
        return "pre_processed/cm_students_dropout_train.csv"
    else:
        print("Escolha inválida! Carregando o dataset completo por padrão.")
        return "pre_processed/students_dropout_train.csv"


# Obter o caminho do arquivo com base na escolha do usuário
file_path = get_file_path()

# Carregar o dataset escolhido
dataset = pd.read_csv(file_path)

# Separar as features (X) e o target (y)
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
print(f"Melhores parâmetros: {gs.best_params_}")
print(f"Melhor score: {gs.best_score_}")
