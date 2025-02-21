import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from itertools import product
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("data/smartphone_data.csv")

# 🔹 1️⃣ Descriptive Analysis
print("🔹 Descriptive Statistics:\n", df.describe())

# Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df["Price"], bins=10, kde=True, color="blue")
plt.title("Price Distribution of Smartphones")
plt.xlabel("Price (INR)")
plt.ylabel("Count")
plt.show()

# 🔹 2️⃣ Correlation Analysis
correlation_matrix = df.corr(numeric_only=True)
print("\n🔹 Correlation Matrix:\n", correlation_matrix)

# Heatmap of Correlations
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Feature Correlation Heatmap")
plt.show()

# 🔹 3️⃣ Conjoint Analysis (Basic Simulation)
attributes = {
    "Brand": df["Brand"].unique(),
    "Storage": df["Storage"].unique(),
    "RAM": df["RAM"].unique()
}

profiles = list(product(*attributes.values()))
profiles_df = pd.DataFrame(profiles, columns=attributes.keys())

np.random.seed(42)
profiles_df["Preference_Score"] = np.random.randint(1, 100, len(profiles_df))

print("\n🔹 Sample Conjoint Analysis Data:\n", profiles_df.head())

# Boxplot for Brand Preference
plt.figure(figsize=(8, 5))
sns.boxplot(x="Brand", y="Preference_Score", data=profiles_df)
plt.title("Brand Preference in Conjoint Analysis")
plt.xlabel("Brand")
plt.ylabel("Preference Score")
plt.xticks(rotation=45)
plt.show()

# 🔹 4️⃣ Feature Importance (Simple Analysis)
# Encode categorical data
encoder = LabelEncoder()
df["Brand"] = encoder.fit_transform(df["Brand"])
df["Storage"] = df["Storage"].str.replace("GB", "").astype(int)
df["RAM"] = df["RAM"].str.replace("GB", "").astype(int)
df["Camera"] = df["Camera"].str.replace("MP", "").astype(int)

X = df.drop(columns=["Price", "Processor"])  # Features
y = df["Price"]  # Target

# Train Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Display Feature Importance (Model Coefficients)
feature_importance = pd.DataFrame({"Feature": X.columns, "Importance": model.coef_})
feature_importance = feature_importance.sort_values(by="Importance", ascending=False)
print("\n🔹 Feature Importance:\n", feature_importance)

# 🔹 5️⃣ Predictive Model (Price Prediction)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Scatter Plot of Predictions
plt.figure(figsize=(8, 5))
plt.scatter(y_test, y_pred, color="green", alpha=0.5)
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.title("Actual vs Predicted Price")
plt.show()

# 🔹 6️⃣ Customer Segmentation (K-Means Clustering)
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X)

# Cluster Visualization
plt.figure(figsize=(8, 5))
sns.scatterplot(x=df["Battery"], y=df["Price"], hue=df["Cluster"], palette="viridis")
plt.title("Customer Segmentation based on Battery & Price")
plt.xlabel("Battery Life (hrs)")
plt.ylabel("Price (INR)")
plt.legend(title="Cluster")
plt.show()
