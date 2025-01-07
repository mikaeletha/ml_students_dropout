import matplotlib.pyplot as plt
import pandas
from numpy import sort
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

output_var = "Dropout"

dataset = pandas.read_csv(
    "pre_processed/students_dropout_train.csv")
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var]

test_dataset = pandas.read_csv(
    "pre_processed/students_dropout_test.csv")
x_test = test_dataset.drop(columns=[output_var])
t_test = test_dataset[output_var]

rf = RandomForestClassifier(
    max_features=10,
    n_estimators=603
)
rf.fit(x_train, t_train)

# model predicted outputs
y_train = rf.predict(x_train)
y_test = rf.predict(x_test)

classes = dataset[output_var].unique()


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
