import matplotlib.pyplot as plt
import pandas
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, classification_report
from sklearn.svm import SVC

target = 'Dropout'

# SEPERAR FEATURES E TARGET TREINO
dataset = pandas.read_csv(
    "pre_processed/students_dropout_train.csv")
x_train = dataset.drop(columns=[target])
t_train = dataset[target]

# SEPERAR FEATURES E TARGET TESTE
test_dataset = pandas.read_csv(
    "pre_processed/students_dropout_test.csv")
x_test = test_dataset.drop(columns=[target])
t_test = test_dataset[target]

# HIPERPARAMETROS
c = 0.35
kernel = "linear"

# CRIAR MODELO SVM
svm = SVC(C=c, kernel=kernel)

# TREINAR MODELO USANDO OS DADOS DE TREINAMENTO
svm.fit(x_train, t_train)

# SAÍDA PREVISTAS PELO MODELO - PREVISÕES
y_train = svm.predict(x_train)
y_test = svm.predict(x_test)

# OBTER CLASSES/TARGET DO CONJUNTO DE DADOS (SPAM/NÃO SPAM)
classes = dataset[target].unique()


def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()


display_confusion_matrix(t_train, y_train, classes,
                         "Training data confusion matrix")
display_confusion_matrix(t_test, y_test, classes, "Test data confusion matrix")

# print("Training data:")
# print(f"accuracy: {accuracy_score(t_train, y_train) * 100:.2f}%")

# print("Testing data:")
# print(f"accuracy: {accuracy_score(t_test, y_test) * 100:.2f}%")

print("Train report")
train_report = classification_report(t_train, y_train, digits=4)
print(train_report)

print("Test report")
test_report = classification_report(t_test, y_test, digits=4)
print(test_report)
