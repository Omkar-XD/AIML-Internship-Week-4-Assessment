import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

df = pd.read_csv("../data/heart.csv")

X = df.drop("target", axis=1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertia = []
silhouette = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X_scaled)
    inertia.append(km.inertia_)
    silhouette.append(silhouette_score(X_scaled, labels))

plt.plot(k_range, inertia, marker="o")
plt.xlabel("K")
plt.ylabel("Inertia")
plt.savefig("../plots/elbow.png")
plt.close()
