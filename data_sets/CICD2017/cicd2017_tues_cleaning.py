from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

###############################
# Load CICIDS2017 file
###############################
file_path = Path(r"C:\Users\Kaye\OneDrive\Documentos\CS406\data_sets\CICD2017\Tuesday-WorkingHours.pcap_ISCX.csv")

cicd_DF = pd.read_csv(file_path)

print("Dataset loaded.")
print("Initial shape:", cicd_DF.shape)
print(cicd_DF.head())
cicd_DF.columns = cicd_DF.columns.str.strip().str.replace(" ", "_", regex=False)

###############################
# Basic inspection
###############################
print("\n**** CICD2017 tuesday INFO ****")
print(cicd_DF.info())

print("\n**** MISSING VALUES ****")
print(cicd_DF.isnull().sum().sort_values(ascending=False).head(20))

print("\n**** DUPLICATE ROWS ****")
print(cicd_DF.duplicated().sum())

###############################
# Handle inconsistent records
###############################
# Code will take some time to load due to size
print("\nNow Cleaning data...")

# Replace infinite values with NaN
cicd_DF.replace([np.inf, -np.inf], np.nan, inplace=True)

# Standardize label formatting
if "Label" not in cicd_DF.columns:
    raise ValueError("The dataset does not contain a 'Label' column.")

cicd_DF["Label"] = cicd_DF["Label"].astype(str).str.strip().str.upper()

# Drop rows with missing values
rows_before = cicd_DF.shape[0]
cicd_DF.dropna(inplace=True)
rows_after_na = cicd_DF.shape[0]
print(f"Removed {rows_before - rows_after_na} rows with missing or infinite values.")

# Remove duplicate rows
rows_before_dup = cicd_DF.shape[0]
cicd_DF.drop_duplicates(inplace=True)
rows_after_dup = cicd_DF.shape[0]
print(f"Removed {rows_before_dup - rows_after_dup} duplicate rows.")

print("Shape after cleaning:", cicd_DF.shape)

###############################
# Transform target variable
###############################
cicd_DF["Label"] = cicd_DF["Label"].apply(lambda x: 0 if x == "BENIGN" else 1)

print("\n**** LABEL DISTRIBUTION ****")
print(cicd_DF["Label"].value_counts())

###############################
# Exploratory Data Analysis (EDA)
###############################
saved_imgs = file_path.parent / "saved_images"
saved_imgs.mkdir(exist_ok=True)

# Class distribution
plt.figure(figsize=(6, 4))
cicd_DF["Label"].value_counts().plot(kind="bar")
plt.title("Class Distribution")
plt.xlabel("Label (0 = Benign, 1 = Attack)")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(saved_imgs / "class_distribution_tuesday.png")
plt.show()

# Select numeric columns
num_colmns = cicd_DF.select_dtypes(include=[np.number]).columns.tolist()
num_colmns_no_label = [col for col in num_colmns if col != "Label"]

# Histograms
samp_colmns = num_colmns_no_label[:10]
if samp_colmns:
    cicd_DF[samp_colmns].hist(figsize=(14, 10))
    plt.suptitle("Feature Distributions (Sample)")
    plt.tight_layout()
    plt.savefig(saved_imgs / "feature_histograms_tuesday.png")
    plt.show()

# Correlation matrix
if len(samp_colmns) > 1:
    corr_sample = cicd_DF[samp_colmns + ["Label"]].corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(corr_sample, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(corr_sample.columns)), corr_sample.columns, rotation=90)
    plt.yticks(range(len(corr_sample.columns)), corr_sample.columns)
    plt.title("Correlation Matrix (Sample)")
    plt.tight_layout()
    plt.savefig(saved_imgs / "correlation_matrix_tuesday.png")
    plt.show()

# Boxplot
if num_colmns_no_label:
    plt.figure(figsize=(8, 4))
    plt.boxplot(cicd_DF[num_colmns_no_label[0]].dropna())
    plt.title(f"Boxplot of {num_colmns_no_label[0]}")
    plt.tight_layout()
    plt.savefig(saved_imgs / "boxplot_sample_feature_tuesday.png")
    plt.show()

print(f"\nEDA plots saved to: {saved_imgs}")

###############################
# Feature engineering
###############################
X = cicd_DF.drop(columns=["Label"])
y = cicd_DF["Label"]

# Remove highly correlated features
corr_matrix = X.corr().abs()
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop_corr = [col for col in upper.columns if any(upper[col] > 0.90)]

print("\n**** REMOVED CORRELATED COLUMNS ****")
if to_drop_corr:
    for col in to_drop_corr:
        print(col)
else:
    print("No columns removed.")

print("\nThreshold used: correlation > 0.90")

# Drop them
X = X.drop(columns=to_drop_corr)

print(f"\nTotal removed: {len(to_drop_corr)}")

print("\n**** KEPT INFO ****")
print(cicd_DF.info())

print("Shape after correlation filtering:", X.shape)

# Normalize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("Scaled feature matrix shape:", X_scaled.shape)

# Remove low-variance features
variance_selector = VarianceThreshold(threshold=0.01)
X_prepared = variance_selector.fit_transform(X_scaled)

print("Shape after low-variance filtering:", X_prepared.shape)

# Save cleaned dataset
cleaned_df = X.copy()
cleaned_df["Label"] = y.values

cleaned_output_file = file_path.parent / "cicids2017_tuesday_cleaned.csv"
cleaned_df.to_csv(cleaned_output_file, index=False)

print(f"\nCleaned dataset saved to: {cleaned_output_file}")

# Final summary
print("\n**** FINAL SUMMARY ****")
print("Final cleaned dataframe shape:", cleaned_df.shape)
print("Prepared feature matrix shape:", X_prepared.shape)
print("Pipeline complete.")