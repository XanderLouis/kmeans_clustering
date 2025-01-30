import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib
from sklearn.preprocessing import StandardScaler

# Load and preprocess data
df = pd.read_csv("data/data_clustering.csv")

# Assign values to X before scaling
X = df.iloc[:, :].values

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Using the elbow method to find the optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++", random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss)
plt.title("The Elbow Method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# Train the K-Means model on the scaled data
kmeans = KMeans(n_clusters=5, init="k-means++", random_state=42)
y_pred = kmeans.fit_predict(X_scaled)

# Save the trained model
joblib.dump((kmeans, scaler), "model/model.pkl")

# Predictions
plt.scatter(X_scaled[y_pred == 0, 0], X_scaled[y_pred == 0, 1], s=100, c="red", label="Prudent Spenders")
plt.scatter(X_scaled[y_pred == 1, 0], X_scaled[y_pred == 1, 1], s=100, c="blue", label="Generous Spenders")
plt.scatter(X_scaled[y_pred == 2, 0], X_scaled[y_pred == 2, 1], s=100, c="green", label="Extravagant Spenders")
plt.scatter(X_scaled[y_pred == 3, 0], X_scaled[y_pred == 3, 1], s=100, c="cyan", label="Wise Spenders")
plt.scatter(X_scaled[y_pred == 4, 0], X_scaled[y_pred == 4, 1], s=100, c="magenta", label="Loose Spenders")
plt.scatter(
    kmeans.cluster_centers_[:, 0],
    kmeans.cluster_centers_[:, 1],
    s=300,
    c="yellow",
    label="Centroids",
)
plt.title("Clusters of customers")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.show()
