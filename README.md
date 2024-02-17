##  Цель: сделать многоклассовую классификацию на основе имеющихся признаков на train и проверим на test.


Дан датасет с набором данных Samsung Human Activity Recognition.
Gо ссылке https://drive.google.com/drive/u/0/folders/1yrWAo_RcJbpfOoxrSYyxf1oEbxt8NYK_.
Данные поступают с акселерометров и гироскопов мобильных телефонов Samsung Galaxy S3, также известен вид активности человека с телефоном в кармане – ходил ли он, стоял, лежал, сидел или шел вверх/вниз по лестнице.
Имеющиеся метки соответствуют:

1 - ходьбе,
2 - подъему вверх по лестнице,
3 - спуску по лестнице,
4 - сидению,
5 - стоянию,
6 - лежанию

# Перечень команд для запуска проекта
1. Склонировать репозиторий: git clone https://github.com/enkryl/rep_for_mlops.git
2. Создать и активировать виртуальное окружение: python3 -m venv .venv, source .venv/bin/activate.
3. Установить poetry и pre-commit: pip install poetry, pip install pre-commit.
4. poetry install
5. pre-commit install
6. Запустить файл train.py: python train.py
7. Запустить файл infer.py: python infer.py
8. Результат работы проекта - обученная модель model.pkl, файл с метриками metrics.txt и файл с предсказаниями predictions.txt.
