import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv('data/vgsales.csv')

descricao = df.describe()
print("Descrição:\n", descricao)
nulos = df.isnull().sum()
removed = df.dropna()
vendas = removed['Global_Sales']

print("Número de valores nulos:", nulos)
media = vendas.mean()
mediana = vendas.median()
moda = vendas.mode()
print("Media:", media)
print("Mediana:", mediana)
print("Moda:", moda)

maximo = vendas.max()
minimo = vendas.min()
print("Amplitude:", maximo - minimo)

desvio_padrao = vendas.std()
print("Desvio padrão:", desvio_padrao)
coeficiente_variacao = desvio_padrao / media
print("Coeficiente de variação:", coeficiente_variacao)
Q1 = vendas.quantile(0.25)
Q3 = vendas.quantile(0.75)
IQR = Q3 - Q1
print("Intervalo interquartil (IQR):", IQR)
assimetria = vendas.skew()
print("Assimetria:", assimetria)
curtose = vendas.kurtosis()
print("Curtose:", curtose)
frequencia_absoluta = vendas.value_counts().sort_index()
frequencia_relativa = vendas.value_counts(normalize=True).sort_index()
print("Frequência absoluta:\n", frequencia_absoluta)
print("Frequência relativa:\n", frequencia_relativa)
frequencia_acumulada = frequencia_absoluta.cumsum()
print("Frequência acumulada:\n", frequencia_acumulada)
correlacao = removed.corr(numeric_only=True)
print("Correlação:\n", correlacao)

boxplot = vendas.plot.box()
plt.title('Boxplot de Vendas Globais')
plt.show()

histograma = vendas.plot.hist(bins=50)
plt.title('Histograma de Vendas Globais')
plt.show()

heatmap_correlacao = plt.matshow(correlacao)
plt.xticks(range(len(correlacao.columns)), correlacao.columns, rotation=90)
plt.yticks(range(len(correlacao.columns)), correlacao.columns)
plt.title('Mapa de Calor da Correlação')
plt.colorbar(heatmap_correlacao)
plt.show()
