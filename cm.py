import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import joblib

# Carregar os dados originais
dataset = pd.read_csv('pre_processed/students_dropout.csv')

# Calcular a matriz de correlação
correlation_matrix = dataset.corr()

# Converter para valores absolutos
absolute_correlation_matrix = correlation_matrix.abs()

# Filtrar variáveis com correlação menor que 0.1 com a variável-alvo
target_variable = 'Dropout'
relevant_variables = absolute_correlation_matrix[target_variable]
less_relevant_variables = relevant_variables[relevant_variables < 0.1]

# Exibir no terminal as variáveis menos relevantes e suas pontuações
print("Variáveis menos relevantes com a variável-alvo (spam):")
print(less_relevant_variables)

# Remover as variáveis com correlação menor que 0.1 com a variável-alvo
variables_to_remove = less_relevant_variables.index
dataset_cleaned = dataset.drop(columns=variables_to_remove)

# Dividir os dados em treino e teste (por exemplo, 80% treino, 20% teste)
X = dataset_cleaned.drop(columns=['Dropout'])  # Características (features)
y = dataset_cleaned['Dropout']  # Variável alvo (target)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.9, random_state=42)

# Juntar features e target para treino e teste
train_data = pd.concat([X_train, y_train], axis=1)
test_data = pd.concat([X_test, y_test], axis=1)

scaler = MinMaxScaler((0, 1)).fit(X_train)

# Salvar o scaler
joblib.dump(scaler, 'models/students_scaler_limited.pkl')

# Salvar os conjuntos de dados de treino e teste combinados
train_data.to_csv(
    'pre_processed/cm_students_dropout_train.csv', index=False)
test_data.to_csv('pre_processed/cm_students_dropout_test.csv', index=False)

# Verificar os novos arquivos
print("Arquivos de treino e teste combinados e salvos com as colunas removidas.")
