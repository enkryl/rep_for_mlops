import logging

import hydra
import joblib
import numpy as np
import pandas as pd
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="training")
def train_model(cfg):
    X_train = pd.read_csv(cfg.data.X_train_path, sep="\\s+", header=None)
    y_train = pd.read_csv(cfg.data.y_train_path, header=None)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    X_scaled = pd.DataFrame(X_scaled)

    pca = PCA(
        n_components=cfg.preprocessing.pca.n_components,
        random_state=cfg.preprocessing.pca.random_state,
    )
    X_pca = pca.fit_transform(X_scaled)

    y_train = np.array(y_train).ravel()

    model = svm.SVC(kernel="linear")
    model.fit(X_pca, y_train)

    joblib.dump(model, "model.pkl")


if __name__ == "__main__":
    train_model()
