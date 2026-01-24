"""
EDA: Movies Dataset
Author: Khyati Sharma
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("datasets/movie_titles_metadata.csv")

# Overview
print(df.head())
print(df.info())

# Missing Values
print("\nMissing Values:")
print(df.isnull().sum())

# Movie Release Year Distribution
plt.figure()
sns.histplot(df["title_year"].dropna(), bins=20)
plt.title("Movie Release Year Distribution")
plt.show()

# Budget vs Gross
plt.figure()
sns.scatterplot(x="budget", y="gross", data=df)
plt.title("Budget vs Gross Revenue")
plt.show()

# Language Count
plt.figure()
df["language"].value_counts().head(10).plot(kind="bar")
plt.title("Top Movie Languages")
plt.show()

print("""
INSIGHTS:
- Higher budget movies tend to earn higher gross revenue
- Majority of movies are released post-2000
- English dominates movie production
""")
# Recommendations
print("""RECOMMENDATIONS: 
- Invest in higher budget films for better revenue potential
- Focus on English language films for wider audience reach  
""")
# Save cleaned data
df.to_csv("datasets/movie_titles_metadata_Cleaned.csv", index=False)
