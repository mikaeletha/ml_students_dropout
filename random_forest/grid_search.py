import pandas
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

param = [
    {
        # 'n_estimators': [602, 603, 604],
        'n_estimators': list(np.arange(603, 610, 1)),
        'max_features': ["sqrt", "log2", 10, 15]
    },
]

gs = GridSearchCV(
    RandomForestClassifier(),
    param,
    scoring='recall',
    verbose=True
)

output_var = "Dropout"
file_path = "pre_processed/students_dropout_train.csv"
dataset = pandas.read_csv(file_path)
x_train = dataset.drop(columns=[output_var])
t_train = dataset[output_var]

gs.fit(x_train, t_train)

print("Best parameters found: ", gs.best_params_)
print("Best score: ", gs.best_score_)
