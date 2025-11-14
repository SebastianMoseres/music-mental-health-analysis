# =============================================================================
# 1. SETUP AND DATA LOADING
# =============================================================================
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# Create a directory to save plots
if not os.path.exists('plots'):
    os.makedirs('plots')

print("Downloading dataset from KaggleHub...")
# Download latest version of the dataset
path = kagglehub.dataset_download("catherinerasgaitis/mxmh-survey-results")
print(f"Path to dataset files: {path}")

# Construct the full path to the CSV file and load it
csv_file_path = os.path.join(path, "mxmh_survey_results.csv")
df = pd.read_csv(csv_file_path)

print("\nDataset Info:")
df.info()

print("\nDataset Description:")
print(df.describe())


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================================
print("\nStarting Exploratory Data Analysis...")

# Plot 1: Distribution of Anxiety and Depression
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.histplot(df['Anxiety'], kde=True)
plt.title('Distribution of Anxiety Scores')
plt.subplot(1, 2, 2)
sns.histplot(df['Depression'], kde=True)
plt.title('Distribution of Depression Scores')
plt.tight_layout()
plt.savefig('plots/01_anxiety_depression_dist.png')
plt.show()

# Plot 2: Anxiety and Depression by Music Effect
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.boxplot(data=df, x='Music effects', y='Anxiety')
plt.title('Anxiety Scores by Music Effect')
plt.xticks(rotation=15)
plt.subplot(1, 2, 2)
sns.boxplot(data=df, x='Music effects', y='Depression')
plt.title('Depression Scores by Music Effect')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('plots/02_scores_by_music_effect.png')
plt.show()

# Plot 3: Count of Respondents by Favorite Genre
plt.figure(figsize=(12, 8))
sns.countplot(data=df, y='Fav genre', order=df['Fav genre'].value_counts().index, color='steelblue')
plt.title('Number of Respondents by Favorite Genre')
plt.xlabel('Count')
plt.ylabel('Favorite Genre')
plt.tight_layout()
plt.savefig('plots/03_genre_counts.png')
plt.show()

# Plot 4: Anxiety Scores by Favorite Genre
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Fav genre', y='Anxiety')
plt.title('Anxiety Scores by Favorite Genre')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/04_anxiety_by_genre.png')
plt.show()

# Plot 5: Depression Scores by Favorite Genre
plt.figure(figsize=(14, 8))
sns.boxplot(data=df, x='Fav genre', y='Depression')
plt.title('Depression Scores by Favorite Genre')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('plots/05_depression_by_genre.png')
plt.show()

# Plot 6: Correlation Heatmap of Mental Health Metrics
health_df = df[['Anxiety', 'Depression', 'Insomnia', 'OCD']]
correlation_matrix = health_df.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Health Metrics')
plt.savefig('plots/06_health_metrics_heatmap.png')
plt.show()

# Plot 7: Anxiety vs. Listening Hours by Age Group
df['Age Group'] = pd.cut(df['Age'], bins=, labels=['<18', '18-25', '25-40', '40+'])
plt.figure(figsize=(12, 8))
sns.scatterplot(data=df, x='Hours per day', y='Anxiety', hue='Age Group', alpha=0.7, s=50)
plt.title('Anxiety vs. Listening Hours, by Age Group')
plt.savefig('plots/07_anxiety_vs_hours_by_age.png')
plt.show()

# Plot 8: Percentage of High Anxiety Respondents by Favorite Genre
df['High Anxiety_EDA'] = df['Anxiety'] >= 8
genre_anxiety_dist = pd.crosstab(df['Fav genre'], df['High Anxiety_EDA'], normalize='index') * 100
genre_anxiety_dist = genre_anxiety_dist.sort_values(by=True, ascending=False)
genre_anxiety_dist.plot(kind='bar', stacked=True, figsize=(14, 8), color=['skyblue', 'salmon'])
plt.title('Percentage of High Anxiety Respondents by Favorite Genre')
plt.ylabel('Percentage')
plt.xlabel('Favorite Genre')
plt.xticks(rotation=45, ha='right')
plt.legend(title='High Anxiety (>=8)', labels=['False', 'True'])
plt.tight_layout()
plt.savefig('plots/08_high_anxiety_percentage_by_genre.png')
plt.show()

# Plot 9: Distribution of All Health Metrics by Perceived Music Effect
df_melted = df.melt(id_vars=['Music effects'], value_vars=['Anxiety', 'Depression', 'Insomnia', 'OCD'],
                    var_name='Metric', value_name='Score')
plt.figure(figsize=(15, 8))
sns.violinplot(data=df_melted, x='Music effects', y='Score', hue='Metric')
plt.title('Distribution of All Health Metrics by Perceived Music Effect')
plt.savefig('plots/09_all_metrics_by_music_effect.png')
plt.show()


# =============================================================================
# 3. STATISTICAL HYPOTHESIS TESTING
# =============================================================================
print("\nPerforming ANOVA Test for Anxiety Scores by Music Effect...")

# Separate the anxiety scores for each group
anxiety_improve = df[df['Music effects'] == 'Improve']['Anxiety'].dropna()
anxiety_no_effect = df[df['Music effects'] == 'No effect']['Anxiety'].dropna()
anxiety_worsen = df[df['Music effects'] == 'Worsen']['Anxiety'].dropna()

# Perform the ANOVA test
f_statistic, p_value = stats.f_oneway(anxiety_improve, anxiety_no_effect, anxiety_worsen)

print(f"F-statistic: {f_statistic:.4f}")
print(f"P-value: {p_value:.4f}")

# Interpret the result
alpha = 0.05
if p_value < alpha:
    print("\nConclusion: We reject the null hypothesis.")
    print("There is a statistically significant difference in mean anxiety scores between the groups.")
else:
    print("\nConclusion: We fail to reject the null hypothesis.")
    print("There is no statistically significant difference in mean anxiety scores between the groups.")


# =============================================================================
# 4. PREDICTIVE MODELING
# =============================================================================
print("\nStarting Predictive Modeling for High Anxiety...")

# --- 4.1. Data Preparation ---
# Define the binary target variable. A threshold of >= 7 creates a balanced dataset.
df['High Anxiety'] = df['Anxiety'] >= 7

# Define features (X) and target (y)
features = ['Fav genre', 'Music effects', 'Age', 'Hours per day', 'Instrumentalist', 'Composer']
target = 'High Anxiety'

# Drop rows with any missing values in the selected columns for simplicity
df_model = df.dropna(subset=features + [target])

X = df_model[features]
y = df_model[target]

# One-Hot Encode categorical features. drop_first=True to avoid multicollinearity.
X_encoded = pd.get_dummies(X, columns=['Fav genre', 'Music effects', 'Instrumentalist', 'Composer'], drop_first=True)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.3, random_state=42, stratify=y)
print(f"\nTraining set size: {len(X_train)}, Test set size: {len(X_test)}")

# --- 4.2. Model 1: Logistic Regression (Baseline) ---
print("\n--- Training Logistic Regression Model ---")
log_reg = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
log_reg.fit(X_train, y_train)
y_pred_lr = log_reg.predict(X_test)

print("\n--- Logistic Regression Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_lr))
plt.savefig('plots/10_logistic_regression_report.png') # Note: This saves an empty plot, for consistency.

# --- 4.3. Model 2: Random Forest Classifier ---
print("\n--- Training Random Forest Model ---")
rf_model = RandomForestClassifier(random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("\n--- Random Forest Performance ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred_rf))
plt.savefig('plots/11_random_forest_report.png') # Note: This saves an empty plot, for consistency.

# --- 4.4. Feature Importance Analysis ---
print("\n--- Analyzing Feature Importance ---")
importances = rf_model.feature_importances_
feature_names = X_train.columns

# Create a pandas series for easier plotting and analysis
feature_importance_series = pd.Series(importances, index=feature_names)

# Plot the top 15 most important features
plt.figure(figsize=(12, 8))
feature_importance_series.nlargest(15).sort_values().plot(kind='barh')
plt.title('Top 15 Most Important Features in Predicting High Anxiety')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('plots/12_feature_importance.png')
plt.show()

print("\nAnalysis complete. All plots have been saved to the /plots directory.")