#Guilherme Bertinati Silvestre NUSP-14575683


import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import openml

dataset_id = 1457
dataset = openml.datasets.get_dataset(dataset_id)

# X - data frame com as instâncias
# y - array com a variável alvo / target
X, y, _, _ = dataset.get_data(
    dataset_format="dataframe",
    target=dataset.default_target_attribute
)

#Print inicial para verificar os dados como estão destribuidos, ele retorna quatas vezes a carácterística foi usada

#juntar todas as reviews de cada usuário
grupos = y.groupby(y.index // 30)

#formatar o dataframe, juntando 30 linhas, que equivalem a todas as revisões feitas por cada usuário
def reformat_dataframe(df, a):
    # Agrupando as linhas
    row_groups = np.arange(len(df)) // a
    df_row_reduced = df.groupby(row_groups).sum()

    return df_row_reduced

#novo dataframe separado para cada usuário
X_reduced = reformat_dataframe(X, 30)


nomes = grupos.first().values

assert len(nomes) == len(X_reduced)

X_reduced.index = nomes

print(X_reduced)

# Defina o número de colunas que você deseja considerar
num_cols = 40

X_reduced_10 = X_reduced.iloc[:, :num_cols]

# Calcule a soma total das colunas selecionadas
total = X_reduced_10.sum(axis=1)

# Crie um novo DataFrame para armazenar o resultado
X_normalized_10 = pd.DataFrame()

# Normalizar cada valor pelo total da soma das colunas
for col in X_reduced_10.columns:
    X_normalized_10[col] = X_reduced_10[col].reset_index(drop=True) / total.reset_index(drop=True) * 100

# Agora, vamos criar o histograma para cada usuário
for i, row in X_normalized_10.iterrows():
    # Crie uma nova figura para cada usuário
    plt.figure()

    # Crie um histograma com os valores
    plt.bar(X_normalized_10.columns, row, width= 0.5)

    # Defina o título para o nome do usuário (índice)
    plt.title(X_reduced.index[i])

    # Defina os rótulos dos eixos
    plt.xlabel('Colunas')
    plt.ylabel('Porcentagem de Uso (%)')

    # Mostre o gráfico
    plt.show()



X_subset = X.iloc[:, 0:100]
#verificação de semelhanças carácterísticas a partir de um gráfico de calor

correlation_matrix = X_subset.corr()
sns.heatmap(correlation_matrix)
plt.title('Mapa de calor da matriz de correlação')
plt.show() #imprime o mapa de calor

threshold = 0.4 # limite de corelação a ser mostrado
similar_columns = {}

for i in range(len(correlation_matrix.columns)):
    for j in range(i + 1 , len(correlation_matrix.columns)):
        if np.abs(correlation_matrix.iloc[i, j]) >= threshold:
            col_name_i = correlation_matrix.columns[i]
            col_name_j = correlation_matrix.columns[j]
            if col_name_i not in similar_columns:
                similar_columns[col_name_i] = []
            similar_columns[col_name_i].append(col_name_j)

# Imprime as colunas com alta correlação
for key, values in similar_columns.items():
    print(f'Coluna {key} tem alta correlação com as colunas: {values}')



print (X.describe())
#valores descritivos, com os valores máximos de uso até o uso por porcentagens, é possível concluir pela variancia a dispesão de uso


count = 0

for column in X.columns:
    if count < 5:
      #normalização dos dados para uma frequência de 0 a 100
      weights = np.ones_like(X[column])/float(len(X[column]))*100
      plt.hist(X[column], bins=60, weights=weights)
      plt.title('histograma da Características' + column)
      plt.xlabel('valor')
      plt.ylabel('frequencia')
      plt.show()
      count += 1




X.isnull().sum() #verificação se existe algum valor ausente no dataset, o que retorna nenhum valor ausente
#a saída mostra que no dataset não há valores faltantes


# Criando um novo DataFrame para armazenar os dados sem outliers
X_out = X.copy()

# Criando um dicionário para armazenar o número de outliers para cada coluna
outliers_dict = {}

total_outliers = 0

for column in X_out.columns:
    # Calculando a média e o desvio padrão da coluna
    mean = X_out[column].mean()
    std = X_out[column].std()

    # Identificando os outliers usando a média de cada coluna
    outliers = (np.abs(X_out[column] - mean) > 3 * std)

    # Contando o número de outliers
    num_outliers = outliers.sum()

    # contando todos os outliers existentes
    total_outliers += num_outliers

    # Adicionando o número de outliers ao dicionário
    outliers_dict[column] = num_outliers

# Convertendo o dicionário em um DataFrame
outliers_df = pd.DataFrame(list(outliers_dict.items()), columns=['Coluna', 'Número de Outliers'])

print(outliers_df)

print(f"número total de: {total_outliers}.")

#no caso específico do dataset a limpeza não é bem vinda, os outliers entregam carácterísticas específicas de cada usuário

i = 0
for column in X.columns:
  if i < 10:

    plt.boxplot(X[column])

    plt.title('Boxplot para a coluna: ' + column)

    plt.show()

    i += 1

#demonstração gráfica do outlaier usando outro método IDQ




