import logging

import hydra
import joblib
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="infering")
def testing_model(cfg):
    X_test = pd.read_csv(cfg.data.X_test_path, sep="\\s+", header=None)
    y_test = pd.read_csv(cfg.data.y_test_path, header=None)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_test)
    X_scaled = pd.DataFrame(X_scaled)

    pca = PCA(
        n_components=cfg.preprocessing.pca.n_components,
        random_state=cfg.preprocessing.pca.random_state,
    )
    X_pca = pca.fit_transform(X_scaled)

    model = joblib.load(cfg.model.my_model)

    y_test = np.array(y_test).ravel()

    y_pred = model.predict(X_pca)

    acc_test = metrics.accuracy_score(y_test, y_pred)

    np.savetxt("predictions.txt", y_pred)
    np.savetxt("metrics.txt", [acc_test])


if __name__ == "__main__":
    testing_model()
