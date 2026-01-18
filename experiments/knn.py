import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

k_values = range(1, 21)
val_scores = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    val_scores.append(accuracy_score(y_val, knn.predict(X_val)))

plt.plot(k_values, val_scores, marker="o")
plt.xlabel("K")
plt.ylabel("Validation Accuracy")
plt.savefig("../plots/k_vs_accuracy.png")
plt.close()
