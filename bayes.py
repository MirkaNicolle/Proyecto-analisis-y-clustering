import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import make_scorer, accuracy_score,precision_score,recall_score,f1_score
import seaborn as sns

dftotal = pd.read_csv("nacimientos17a19.csv", encoding='latin-1')

#dftotal = dftotal.dropna()

#dftotal = dftotal.sample(n=100000, random_state=35)

#dftotal = dftotal.drop(['TipoIns','Depreg','Mupreg','Mesreg','Añoreg','Sexo','Tipar','ViaPar','Libras','Onzas','Asisrec','Sitioocu','Mupocu','Diaocu','Mesocu','Añoocu','Paisrep', 'Deprep', 'Muprep', 'PuebloPP', 'Paisnacp', 'Depnap', 'Munpnap', 'Paisrem', 'Deprem', 'Muprem', 'PuebloPM', 'Paisnacm', 'Depnam', 'Mupnam', 'Tohite', 'Tohinm', 'Tohivi','Unnamed: 0', 'Escivp', 'Escolap', 'Ocupap', 'Escivm', 'Escolam',
       #'Ocupam'], axis=1)

dftotal = dftotal.drop(['TipoIns','Depreg','Mupreg','Mesreg','Añoreg','Sexo','Tipar','ViaPar','Libras','Onzas','Asisrec','Sitioocu','Mupocu','Diaocu','Mesocu','Añoocu','Paisrep', 'Deprep', 'Muprep', 'PuebloPP', 'Paisnacp', 'Depnap', 'Munpnap', 'Paisrem', 'Deprem', 'Muprem', 'PuebloPM', 'Paisnacm', 'Depnam', 'Mupnam', 'Tohite', 'Tohinm', 'Tohivi','Unnamed: 0'], axis=1)

corr_df = dftotal.corr(method='pearson')

plt.figure(figsize=(8, 6))
sns.heatmap(corr_df, annot=True)
plt.show()

target = dftotal.pop('Depocu')

print(dftotal.columns)


''' from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dftotal,target,test_size=0.3)

from sklearn.naive_bayes import GaussianNB
model = GaussianNB()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

y_predT = model.predict(X_train)

accuracy_Entre = accuracy_score(y_train, y_predT)
accuracy=accuracy_score(y_test,y_pred)
precision =precision_score(y_test, y_pred,average='micro')
recall =  recall_score(y_test, y_pred,average='micro')
f1 = f1_score(y_test,y_pred,average='micro')

print('Accuracy del Test: ',accuracy)
print('Accuracy del train:', accuracy_Entre) '''