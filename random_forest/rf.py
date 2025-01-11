import matplotlib.pyplot as plt
import pandas
from numpy import sort
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
import joblib

# Função para perguntar ao usuário a escolha do dataset


def get_dataset_type():
    print("Escolha o tipo de dataset:")
    print("1 - Completo (students_dropout_train.csv e students_dropout_test.csv)")
    print("2 - Limpo (cm_students_dropout_train.csv e cm_students_dropout_test.csv)")
    choice = input("Digite sua escolha (1 ou 2): ")

    if choice == "1":
        return "students_dropout_train.csv", "students_dropout_test.csv", 'models/rf_students.pkl'
    elif choice == "2":
        return "cm_students_dropout_train.csv", "cm_students_dropout_test.csv", 'models/rf_students_limited.pkl'
    else:
        print("Escolha inválida! Carregando os datasets completos por padrão.")
        return "students_dropout_train.csv", "students_dropout_test.csv", 'models/rf_students.pkl'


# Obter os nomes dos arquivos com base na escolha do usuário
train_file, test_file, model_path = get_dataset_type()
train_path = f"pre_processed/{train_file}"
test_path = f"pre_processed/{test_file}"

# Variável target
output_var = "Dropout"

# Carregar os datasets de treino e teste
dataset = pandas.read_csv(train_path)
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var]

test_dataset = pandas.read_csv(test_path)
x_test = test_dataset.drop(columns=[output_var])
t_test = test_dataset[output_var]

# Criar o modelo Random Forest
rf = RandomForestClassifier(
    max_features=10,
    n_estimators=295
)
rf.fit(x_train, t_train)

# Salvar o modelo no caminho especificado
joblib.dump(rf, model_path)
print(f"Modelo Random Forest salvo em: {model_path}")

# Previsões do modelo
y_train = rf.predict(x_train)
y_test = rf.predict(x_test)

# Classes únicas do target
classes = dataset[output_var].unique()


# Função para exibir matriz de confusão
def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()


# Matriz de confusão para os dados de treino e teste
display_confusion_matrix(t_train, y_train, classes,
                         "Matriz de Confusão (Treinamento)")
display_confusion_matrix(t_test, y_test, classes, "Matriz de Confusão (Teste)")

# Relatórios de classificação para os dados de treino e teste
print("Relatório - Dados de Treinamento")
train_report = classification_report(t_train, y_train, digits=4)
print(train_report)

print("Relatório - Dados de Teste")
test_report = classification_report(t_test, y_test, digits=4)
print(test_report)
