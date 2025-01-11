import argparse

import joblib
import pandas
from matplotlib import pyplot as plt
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report

# dataset = pandas.read_csv('pre_processed/cm_students_dropout_train.csv')
# print(dataset.columns)


def read_dataset(filename):
    # columns_to_keep = [
    #     "Priority", "Nacionality", "AdmissionGrade", "Relocated", "SpecialNeeds",
    #     "HasDebt", "PayTuition", "Gender", "HasScholarship", "X1", "X2", "X3", "X4", "X5",
    #     "X6", "X7", "X8", "X9", "X10", "X11", "X12", "X13", "X14", "X15", "Relationship",
    #     "MothersHigherEducation", "FathersHigherEducation", "Dropout"
    # ]

    columns_to_keep = ['Relocated', 'HasDebt', 'PayTuition', 'Gender', 'HasScholarship',
                       'X2', 'X4', 'X5', 'X8', 'X9', 'X10', 'X11', 'Dropout']

    dataset = pandas.read_csv(filename)

    return dataset[columns_to_keep]


def scale_inputs(x):
    scaler = joblib.load('models/students_scaler.pkl')
    x_scaled = scaler.transform(x)

    return pandas.DataFrame(x_scaled, columns=x.columns)


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

    report = classification_report(t, y, digits=4)
    print(report)

    classes = t.unique()
    cm = ConfusionMatrixDisplay.from_predictions(t, y, labels=sort(classes))
    cm.ax_.set_title(f'{algorithm} confusion matrix')
    plt.show()


def present_model_results(dataset_filename, scaler_filename):
    dataset = read_dataset(dataset_filename)

    x = dataset.drop(columns=['Dropout'])
    t = dataset['Dropout']

    if scaler_filename:
        x = scale_inputs(x)

    present_results('KNN', 'models/knn_students.pkl', x, t)
    present_results('SVM', 'models/svm_students.pkl', x, t)
    present_results('Random forest', 'models/rf_students.pkl', x, t)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Present the Wiscosin model results for a given dataset file.")
    parser.add_argument("filename", type=str,
                        help="Path to the dataset CSV file")
    parser.add_argument("--scaler", type=str, help="Path to the scaler file")

    # Parse the arguments
    args = parser.parse_args()
    present_model_results(args.filename, args.scaler)


if __name__ == "__main__":
    main()

# dataset_filename = "pre_processed/students_dropout_less_correl_train.csv"
# scaler_filename = 'models/students_dropout_less_correl_scaler.pkl'
    # python wisconsin_model_results.py pre_processed/cm_students_dropout_train.csv --scaler models/students_scaler.pkl
