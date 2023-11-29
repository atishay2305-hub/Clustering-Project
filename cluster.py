import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load Git comments from CSV file with specified encoding
data = pd.read_csv('Code_Review_Project.csv', encoding='latin1')
data = data.head(500)

# Text preprocessing
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', str(text))
    text = text.lower()
    return text

data['processed_comments'] = data['subject'].apply(preprocess_text)

# Feature extraction
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(data['processed_comments']).toarray()

# K-Means Clustering
k = 5  # Adjust based on your specific needs
kmeans = KMeans(n_clusters=k, random_state=42)
data['comment_category'] = kmeans.fit_predict(X)

# Assign meaningful labels to clusters
category_labels = {
    0: 'Documentation',
    1: 'Test Coverage',
    2: 'Security',
    3: 'General Improvement',
    4: 'Bug Fix',
    5: 'Other'
}

data['comment_category_label'] = data['comment_category'].map(category_labels)

# Count the number of subjects in each cluster
cluster_counts = data.groupby('comment_category_label')['subject'].count().reset_index()

# Save the cluster counts to a CSV file
output_file = 'cluster_counts.csv'
cluster_counts.to_csv(output_file, index=False)
print(f"Cluster counts saved to '{output_file}'")

# Data Visualization - Pie Chart
plt.figure(figsize=(8, 8))
plt.pie(cluster_counts['subject'], labels=cluster_counts['comment_category_label'], autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Comments in Clusters')
plt.show()
