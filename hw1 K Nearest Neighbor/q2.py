from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

"""  2a  """
iris = load_iris()

data = pd.DataFrame(data=iris.data, columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
target = pd.DataFrame(data=iris.target, columns=['species'])
df = pd.concat([data, target], axis=1)
df.species.replace({0: 'setosa', 1: 'versicolor', 2: 'virginica'}, inplace=True)


""""  2b  """
box = sns.boxplot(x="species", y="sepal length (cm)", data=df)
plt.show()

box = sns.boxplot(x="species", y="sepal width (cm)", data=df)
plt.show()

box = sns.boxplot(x="species", y="petal length (cm)", data=df)
plt.show()

box = sns.boxplot(x="species", y="petal width (cm)", data=df)
plt.show()

"""  2c  """

sns.FacetGrid(df, hue="species", size=6).map(plt.scatter, "sepal length (cm)", "sepal width (cm)").add_legend()
plt.show()
