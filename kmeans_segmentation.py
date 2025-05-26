import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("customer_data.csv")

# Rename columns if needed
df.columns = [col.strip().replace(' ', '_').lower() for col in df.columns]

# Select features for clustering
X = df[['age', 'annual_income_(k$)', 'spending_score_(1-100)']]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Use Elbow method to find optimal k
inertia = []
k_range = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

# Plot Elbow
plt.plot(k_range, inertia, marker='o')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.savefig("elbow_plot.png")
plt.show()

# Fit final KMeans
kmeans = KMeans(n_clusters=5, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Save clustered data
df.to_csv("clustered_customers.csv", index=False)
print("Clustering complete! Output saved to clustered_customers.csv")
