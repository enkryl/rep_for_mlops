# import dvc.api
import joblib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

plt.style.use(["seaborn-darkgrid"])
plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams["font.family"] = "DejaVu Sans"

# %matplotlib inline
# %config InlineBackend.figure_format="retina"

RANDOM_STATE = 42

# your code here

X_test = pd.read_csv("samsung_test.txt", sep="\\s+", header=None)

y_test = pd.read_csv("samsung_test_labels.txt", header=None)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_test)
X_scaled = pd.DataFrame(X_scaled)

pca = PCA(0.9, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

pca.explained_variance_ratio_.sum()

plt.figure(figsize=(15, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_test, s=15, cmap="viridis")

model = joblib.load("model.pkl")

model.fit(X_pca)

y_pred = model.predict(X_test)

accuracy_train = metrics.accuracy_score(y_test.transpose().values[0], y_pred)

np.savetxt("predictions.txt", y_pred)
np.savetxt("metrics.txt", [accuracy_train])
