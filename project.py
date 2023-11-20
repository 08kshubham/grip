import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Load iris dataset from CSV
iris_df = pd.read_csv('iris.csv')

# Extracting features from the dataset
data = iris_df.iloc[:, :-1].values  

# Finding the optimum number of clusters using the Elbow Method
wcss = []  # Within-Cluster Sum of Squares

for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(data)
    wcss.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(8, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters')
plt.ylabel('WCSS')  # Within-Cluster Sum of Squares
plt.show()

# From the Elbow Method graph, choose the optimum number of clusters
# Let's say it's 3 clusters (as there's an "elbow" at 3)

# Applying KMeans to the dataset with the optimum number of clusters
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
kmeans.fit(data)

# Adding cluster labels to the dataset
iris_df['cluster'] = kmeans.labels_

# Visualizing the clusters using pairplot
sns.pairplot(iris_df, hue='cluster', palette='Dark2')
plt.show()
