
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
data = pd.read_csv('ORAB_Annotation_MIMIC.csv')

# Distribution of numerical data
data.hist(bins=15, figsize=(15, 10), layout=(4, 3))

# Boxplots for numerical data to check for outliers
plt.figure(figsize=(15, 10))
data.boxplot()

# Distribution of categorical data
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
data['gender'].value_counts().plot(kind='bar', ax=ax[0]).set_title('Gender Distribution')
data['anchor_age'].plot(kind='hist', ax=ax[1], bins=20).set_title('Age Distribution')

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, fmt=".2f")

plt.show()
