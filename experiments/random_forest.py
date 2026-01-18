import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

trees = [10, 50, 100, 200]
val_scores = []

for n in trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42)
    rf.fit(X_train, y_train)
    val_scores.append(accuracy_score(y_val, rf.predict(X_val)))

plt.plot(trees, val_scores, marker="o")
plt.xlabel("Number of Trees")
plt.ylabel("Validation Accuracy")
plt.savefig("../plots/bias_variance.png")
plt.close()
