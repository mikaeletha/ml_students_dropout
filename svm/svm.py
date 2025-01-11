import matplotlib.pyplot as plt
import pandas
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC
import joblib

# Função para perguntar ao usuário a escolha do dataset


def get_dataset_type():
    print("Escolha o tipo de dataset:")
    print("1 - Completo (students_dropout_train.csv e students_dropout_test.csv)")
    print("2 - Limpo (cm_students_dropout_train.csv e cm_students_dropout_test.csv)")
    choice = input("Digite sua escolha (1 ou 2): ")

    if choice == "1":
        return "students_dropout_train.csv", "students_dropout_test.csv", 'models/svm_students.pkl'
    elif choice == "2":
        return "cm_students_dropout_train.csv", "cm_students_dropout_test.csv", 'models/svm_students_limited.pkl'
    else:
        print("Escolha inválida! Carregando os datasets completos por padrão.")
        return "students_dropout_train.csv", "students_dropout_test.csv", 'models/svm_students.pkl'


# Obter os nomes dos arquivos com base na escolha do usuário
train_file, test_file, model_path = get_dataset_type()
train_file_path = f"pre_processed/{train_file}"
test_file_path = f"pre_processed/{test_file}"

target = 'Dropout'

# Separar features e target do conjunto de treino
dataset = pandas.read_csv(train_file_path)
x_train = dataset.drop(columns=[target])
t_train = dataset[target]

# Separar features e target do conjunto de teste
test_dataset = pandas.read_csv(test_file_path)
x_test = test_dataset.drop(columns=[target])
t_test = test_dataset[target]

# Hiperparâmetros do modelo SVM
c = 0.070
kernel = "poly"

# Criar modelo SVM
svm = SVC(C=c, kernel=kernel)

# Treinar modelo usando os dados de treinamento
svm.fit(x_train, t_train)

# Salvar o modelo no caminho especificado
joblib.dump(svm, model_path)
print(f"Modelo SVM salvo em: {model_path}")

# Saídas previstas pelo modelo - Previsões
y_train = svm.predict(x_train)
y_test = svm.predict(x_test)

# Obter classes/target do conjunto de dados
classes = dataset[target].unique()


def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()


# Exibir matrizes de confusão
display_confusion_matrix(t_train, y_train, classes,
                         "Matriz de Confusão (Treinamento)")
display_confusion_matrix(t_test, y_test, classes, "Matriz de Confusão (Teste)")

# Relatórios de classificação
print("Relatório - Dados de Treinamento")
train_report = classification_report(t_train, y_train, digits=4)
print(train_report)

print("Relatório - Dados de Teste")
test_report = classification_report(t_test, y_test, digits=4)
print(test_report)
