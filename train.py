# import dvc.api
import joblib
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# folder_path = "//wsl.localhost/Ubuntu/home/kate/mlops_hw/rep_for_mlops/.dvc"

# Загрузка данных с помощью DVC API
# with dvc.api.open(folder_path, remote = 'myrem') as files_list:
#    for file_path in files_list:
#        print(file_path)
# Делаем что-то с каждым файлом, например, читаем его содержимое
#        with dvc.api.open(file_path,
#                          repo='.',  #Путь к корневому каталогу DVCrep
#                          remote='myrem',
#                          mode='r'  # Режим чтения
#                          ) as file_content:
# Считываем содержимое файла
#                data = file_content.read()
# Делаем что-то с данными
#                print(data)


plt.style.use(["seaborn-darkgrid"])
plt.rcParams["figure.figsize"] = (12, 9)
plt.rcParams["font.family"] = "DejaVu Sans"

# %matplotlib inline
# %config InlineBackend.figure_format="retina"

RANDOM_STATE = 42

# your code here
X_train = pd.read_csv("samsung_train.txt", sep="\\s+", header=None)

y_train = pd.read_csv("samsung_train_labels.txt", header=None)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_scaled = pd.DataFrame(X_scaled)

pca = PCA(0.9, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_scaled)

pca.explained_variance_ratio_.sum()

plt.figure(figsize=(15, 10))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y_train, s=15, cmap="viridis")

model = svm.SVC(kernel="linear")
model.fit(X_train, y_train)

# сохранение модели
joblib.dump(model, "model.pkl")
