import pandas
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

param = [
    {
        # 'C': [0.25, 0.30, 0.35, 0.4],
        'C': list(np.arange(0.25, 0.40, 0.05)),
        'kernel': ['linear', 'poly', 'sigmoid', 'rbf']
    },
]


gs = GridSearchCV(
    # SVC(),
    # Ajuste para lidar com classes desbalanceadas
    SVC(class_weight='balanced'),
    param,
    # scoring='precision',
    scoring='recall',
    verbose=True
)
file_path = "pre_processed/students_dropout_train.csv"
dataset = pandas.read_csv(file_path)
x_train = dataset.drop(columns=['Dropout'])
t_train = dataset['Dropout']


# Fit the model
gs.fit(x_train, t_train)

# Print the best parameters
print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
