
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import scipy.stats as stats

plt.rcParams['figure.figsize'] = (16, 9)
plt.style.use('ggplot')

matri = pd.read_csv('matrimonios17a19.csv', encoding='latin-1')
naci = pd.read_csv('nacimientos17a19.csv', encoding='latin-1')

normal = naci.select_dtypes(include = np.number)
CN = normal.columns.values
normal = normal.dropna()
r = ''

fig = plt.figure()
g = 0
for i in CN:
    estadistico1, p_value1 = stats.kstest(normal[i], 'norm')

    if p_value1 > 0.5:
        r = 'Es normal'
    else:
        r = 'no es normal'

    plt.subplot(7,7,g+1)
    sns.distplot(normal[i])
    plt.xlabel(i)
    g+= 1

    print(i, ": ", r)

plt.tight_layout()
plt.show()

