import matplotlib.pyplot as plt
import pandas
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, recall_score
from sklearn.neighbors import KNeighborsClassifier
import joblib


def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()


# Função para perguntar ao usuário a escolha do dataset
def get_dataset_type():
    print("Escolha o tipo de dataset:")
    print("1 - Completo (students_dropout_train.csv e students_dropout_test.csv)")
    print("2 - Limpo (cm_students_dropout_train.csv e cm_students_dropout_test.csv)")
    choice = input("Digite sua escolha (1 ou 2): ")

    if choice == "1":
        return "students_dropout_train.csv", "students_dropout_test.csv", 'models/knn_students.pkl'
    elif choice == "2":
        return "cm_students_dropout_train.csv", "cm_students_dropout_test.csv", 'models/knn_students_limited.pkl'
    else:
        print("Escolha inválida! Carregando os datasets completos por padrão.")
        return "students_dropout_train.csv", "students_dropout_test.csv", 'models/knn_students.pkl'


# Obter os nomes dos arquivos com base na escolha do usuário
train_file, test_file, model_path = get_dataset_type()

# Caminho dos arquivos
train_file_path = f"pre_processed/{train_file}"
test_file_path = f"pre_processed/{test_file}"

# Carregar os datasets escolhidos
dataset = pandas.read_csv(train_file_path)
x_train = dataset.drop(columns=['Dropout'])
t_train = dataset['Dropout']

test_dataset = pandas.read_csv(test_file_path)
x_test = test_dataset.drop(columns=['Dropout'])
t_test = test_dataset['Dropout']

# Configuração do modelo KNN
n_neighbors = 1
p = 3
knn = KNeighborsClassifier(n_neighbors, p=p)
knn.fit(x_train, t_train)

# Salvar o modelo no caminho especificado
joblib.dump(knn, model_path)
print(f"Modelo KNN salvo em: {model_path}")

# Previsões
y_train = knn.predict(x_train)
y_test = knn.predict(x_test)

quality = dataset['Dropout'].unique()

# Exibir métricas para os dados de treino
print("Dados de treinamento:")
print(f"Accuracy: {accuracy_score(t_train, y_train) * 100:.2f}%")
print(f"Recall: {recall_score(t_train, y_train) * 100:.2f}%")

# Exibir métricas para os dados de teste
print("Dados de teste:")
print(f"Accuracy: {accuracy_score(t_test, y_test) * 100:.2f}%")
print(f"Recall: {recall_score(t_test, y_test) * 100:.2f}%")

# Relatório detalhado para treino e teste
print("Relatório de treino:")
train_report = classification_report(t_train, y_train, digits=4)
print(train_report)

print("Relatório de teste:")
test_report = classification_report(t_test, y_test, digits=4)
print(test_report)

# Exibir matrizes de confusão
display_confusion_matrix(t_train, y_train, quality,
                         "Matriz de confusão (treinamento)")
display_confusion_matrix(t_test, y_test, quality, "Matriz de confusão (teste)")
