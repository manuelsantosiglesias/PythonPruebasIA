import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split

def get_random_ops(rows=100):
    data = []
    for i in range(rows):
        a = random.randint(1, 100)
        b = random.randint(1, 100)
        suma, resta = random.choice([
            [1, 0],
            [0, 1],
        ])
        if suma == 1: y = a + b;
        if resta == 1: y = a - b;
        data.append({
            "a": a,
            "b": b,
            "suma": suma,
            "resta": resta,
            "y": y
        })
    return data


data = pd.DataFrame(get_random_ops(rows=10000))
data[["a", "b", "suma", "resta", "y"]].head()

X_train, X_test, y_train, y_test = train_test_split(
    data[["a", "b", "suma", "resta"]], data["y"],
    test_size=0.30, random_state=42
)

model = MLPRegressor(
    # Paso 1
    max_iter=800,
    # Para reentrenar
    hidden_layer_sizes=(100, 100, 100),
    learning_rate_init=0.0001,
)
model.fit(X_train, y_train)
predict = model.predict(X_test)

data_check = pd.DataFrame(predict, columns=["predict"])
data_check["y"] = list(y_test)
data_check.set_index(["y"], drop=False, inplace=True)
data_check.sort_values(by=["y"], inplace=True)

data_check.plot()
plt.show()