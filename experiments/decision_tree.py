import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

df = pd.read_csv("../data/heart.csv")

X = df.drop("target", axis=1)
y = df["target"]

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# No depth limit
dt_full = DecisionTreeClassifier(random_state=42)
dt_full.fit(X_train, y_train)

train_acc_full = accuracy_score(y_train, dt_full.predict(X_train))
val_acc_full = accuracy_score(y_val, dt_full.predict(X_val))

# Tuned depth
dt_tuned = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_tuned.fit(X_train, y_train)

train_acc_tuned = accuracy_score(y_train, dt_tuned.predict(X_train))
val_acc_tuned = accuracy_score(y_val, dt_tuned.predict(X_val))

print("Decision Tree (No Limit)")
print(train_acc_full, val_acc_full, train_acc_full - val_acc_full)

print("Decision Tree (max_depth=5)")
print(train_acc_tuned, val_acc_tuned, train_acc_tuned - val_acc_tuned)
