import joblib
import pandas as pd
from matplotlib import pyplot as plt
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def read_dataset(filename):
    columns_to_keep = ["Relocated", "HasDebt", "PayTuition", "Gender",
                       "HasScholarship", "X2", "X4", "X5", "X8", "X9", "X10", "X11", "Dropout"]

    dataset = pd.read_csv(filename)
    return dataset[columns_to_keep]


def encode_features(data):
    label_encoders = {}
    for column in data.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        data[column] = le.fit_transform(data[column])
        label_encoders[column] = le
    return data


def scale_inputs(x):
    # scaler = joblib.load('models/students_dropout_less_correl_scaler.pkl')
    scaler = joblib.load('models/students_scaler_limited.pkl')
    x_scaled = scaler.transform(x)
    return pd.DataFrame(x_scaled, columns=x.columns)


def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(
        targets, predicted, labels=sort(classes))
    cm.ax_.set_title(title)
    plt.show()


def present_results(algorithm, model_filename, x, t):
    print('---------------------')
    print(algorithm)
    print("---------------------")

    model = joblib.load(model_filename)

    y = model.predict(x)  # model predicted outputs

    print(f"accuracy: {accuracy_score(t, y) * 100:.2f}%")

    report = classification_report(t, y, digits=4, zero_division=0)
    print(report)

    classes = t.unique()
    cm = ConfusionMatrixDisplay.from_predictions(t, y, labels=sort(classes))
    cm.ax_.set_title(f'{algorithm} confusion matrix')
    plt.show()


def present_model_results(dataset_filename, scaler_filename):
    dataset = read_dataset(dataset_filename)

    x = dataset.drop(columns=['Dropout'])
    t = dataset['Dropout']

    # Encode categorical features
    x = encode_features(x)

    if scaler_filename:
        x = scale_inputs(x)

    present_results('KNN', 'models/knn_students.pkl', x, t)
    present_results('SVM', 'models/svm_students.pkl', x, t)
    present_results('Random forest',
                    'models/rf_students.pkl', x, t)


# Defina diretamente os parâmetros para o arquivo e scaler
# dataset_filename = "pre_processed/students_dropout_less_correl_train.csv"
dataset_filename = "pre_processed/cm_students_dropout_train.csv"
# scaler_filename = 'models/students_dropout_less_correl_scaler.pkl'
scaler_filename = 'models/students_scaler_limited.pkl'

# Chame a função para apresentar os resultados
present_model_results(dataset_filename, scaler_filename)
