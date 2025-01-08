import pandas
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pandas.read_csv("data/students_dropout.csv")

# MAPEIA VARIAVEIS PARA VALORES BINARIOS
dataset['Nacionality'] = dataset['Nacionality'].map(
    {'Other': 0, 'Portuguese': 1})
dataset['Relocated'] = dataset['Relocated'].map({'No': 0, 'Yes': 1})
dataset['SpecialNeeds'] = dataset['SpecialNeeds'].map({'No': 0, 'Yes': 1})
dataset['HasDebt'] = dataset['HasDebt'].map({'No': 0, 'Yes': 1})
dataset['PayTuition'] = dataset['PayTuition'].map({'No': 0, 'Yes': 1})
dataset['Gender'] = dataset['Gender'].map({'Male': 0, 'Female': 1})
dataset['HasScholarship'] = dataset['HasScholarship'].map({'No': 0, 'Yes': 1})
dataset['Relationship'] = dataset['Relationship'].map(
    {'Divorced': 0, 'Facto Union': 1, 'Legally Separated': 2, 'Married': 3, 'Single': 4, 'Widower': 5})
dataset['MothersHigherEducation'] = dataset['MothersHigherEducation'].map({
                                                                          'No': 0, 'Yes': 1})
dataset['FathersHigherEducation'] = dataset['FathersHigherEducation'].map({
                                                                          'No': 0, 'Yes': 1})
dataset['Dropout'] = dataset['Dropout'].map({'No': 0, 'Yes': 1})

# REMOVE COLUNA ID
dataset = dataset.drop(columns=['StudentID'])
# REMOVE LINHA DUPLICADAS
dataset = dataset.drop_duplicates()
print('REMOVE LINHA DUPLICADAS, OK')

# REMOVE LINHHAS VAZIAS
dataset = dataset.dropna()
print('REMOVE LINHHAS VAZIAS, OK')

# SALVA DADOS TRATADOS
dataset.to_csv(
    "pre_processed/students_dropout.csv", index=False)
print('SALVA DADOS TRATADOS, OK')

# INFORMAÇÕES DO DATASET
print('INFORMAÇÕES DO DATASET')
print(dataset.describe(include="all"))

# SEPERAR FEATURES E TARGET
x = dataset.drop(columns=['Dropout'])  # FEATURES - O QUE VAI SER PREVISTO
t = dataset['Dropout']  # TARGET - CARACTERISTICAS

# SEPARAR EM TREINAMENTO E TESTE
x_train, x_test, t_train, t_test = (
    train_test_split(x, t, train_size=0.9, stratify=t, random_state=42))

# COLOCAR OS DADOS NUMÉRICOS NA MESMA ESCALA
scaller = MinMaxScaler((0, 1)).fit(x_train)

x_train_scaled = scaller.transform(x_train)
x_test_scaled = scaller.transform(x_test)

# CRIAR DATAFRAME
x_train_scaled = pandas.DataFrame(
    x_train_scaled, columns=x_train.columns, index=x_train.index)
x_test_scaled = pandas.DataFrame(
    x_test_scaled, columns=x_test.columns, index=x_test.index)

train = pandas.concat([x_train_scaled, t_train], axis='columns', join='inner')
test = pandas.concat([x_test_scaled, t_test], axis='columns', join='inner')

# SALVAR EM CSV OS DADOS DE TREINO E TESTE
train.to_csv(
    'pre_processed/students_dropout_train.csv', index=False)
test.to_csv(
    'pre_processed/students_dropout_test.csv', index=False)

print("Processamento concluído! Os dados de treino e teste foram salvos com sucesso.")
