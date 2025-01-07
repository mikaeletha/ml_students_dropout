import argparse

import joblib
import pandas
from matplotlib import pyplot as plt
from numpy import sort
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report


def read_dataset(filename):
    columns_to_keep = ['Diagnosis', 'Mean_Radius', 'Mean_Texture', 'Mean_Perimeter', 'Mean_Area', 'Mean_Smoothness',
                    'Mean_Compactness', 'Mean_Concavity', 'Mean_Concave_Points', 'Mean_Symmetry', 'Radius_SE',
                    'Perimeter_SE', 'Area_SE', 'Compactness_SE', 'Concavity_SE', 'Concave_Points_SE', 'Worst_Radius',
                    'Worst_Texture', 'Worst_Perimeter', 'Worst_Area', 'Worst_Smoothness', 'Worst_Compactness',
                    'Worst_Concavity', 'Worst_Concave_Points', 'Worst_Symmetry', 'Worst_Fractal_Dimension']

    dataset = pandas.read_csv(filename)

    return dataset[columns_to_keep]


def scale_inputs(x):
    scaler = joblib.load('models/wiscosin_scaler.pkl')
    x_scaled = scaler.transform(x)

    return pandas.DataFrame(x_scaled, columns=x.columns)


def display_confusion_matrix(targets, predicted, classes, title):
    cm = ConfusionMatrixDisplay.from_predictions(targets, predicted, labels=sort(classes))
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

    x = dataset.drop(columns=['Diagnosis'])
    t = dataset['Diagnosis']

    if scaler_filename:
        x = scale_inputs(x)

    present_results('KNN', 'models/knn_wiscosin.pkl', x, t)
    present_results('SVM', 'models/svm_wiscosin.pkl', x, t)
    present_results('Random forest', 'models/rf_wiscosin.pkl', x, t)


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Present the Wiscosin model results for a given dataset file.")
    parser.add_argument("filename", type=str, help="Path to the dataset CSV file")
    parser.add_argument("--scaler", type=str, help="Path to the scaler file")

    # Parse the arguments
    args = parser.parse_args()
    present_model_results(args.filename, args.scaler)

if __name__ == "__main__":
    main()