from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = Path(r"C:\Users\Kaye\OneDrive\Documentos\CS406")
DATA_SET = (ROOT_DIR /
                "data_sets" /
                "CICD2017" /
                "cicids2017_tues_wed_fri_cleaned.csv"
                )
SAVED_IMGS = (ROOT_DIR /
             "decisionTrees_randomForest" /
             "saved_images"
             )
SAVED_IMGS.mkdir(parents=True, exist_ok=True)
print("Currently loading dataset.")
print("\nDue to size of set, may take sometime")
df = pd.read_csv(DATA_SET)
print(df.head())

#########################
# DATA CLEANING / PREPARATION
#############################

print("\n********DATA CLEANING***********")

# Remove extra spaces from column names
df.columns = df.columns.str.strip()
# Replace infi values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
# Convert blank to NaN
df.replace(r"^\s*$", np.nan, regex=True, inplace=True)
# Ensure Label column exists
if "Label" not in df.columns:
    raise ValueError("Label column not found.")
# remove empty colms
df.dropna(axis=1, how="all", inplace=True)
# Show Dataset_Day counts if present
if "Dataset_Day" in df.columns:

    print("\nDataset Day Counts:")
    print(df["Dataset_Day"].value_counts())

# Remove Dataset_Day from modeling
if "Dataset_Day" in df.columns:
    df.drop(columns=["Dataset_Day"], inplace=True)
# Remove rows with missing labels
df.dropna(subset=["Label"], inplace=True)
# Convert all feature columns to numeric
ft_colmns = [col for col in df.columns if col != "Label"]

for col in ft_colmns:
    df[col] = pd.to_numeric(df[col], errors="coerce")
    
# Fill missing values
df[ft_colmns] = df[ft_colmns].fillna(0)
# Remove duplicates
df.drop_duplicates(inplace=True)

print("\nDataset After Cleaning:")
print(df.shape)
print(df.head())

#####################
# PREP
#####################

label_col = "Label"

X = df.drop(label_col, axis=1)
y = df[label_col]

y = y.astype(str)
X = X.select_dtypes(include=[np.number])

print("\nFeature Matrix Shape:")
print(X.shape)

print("Unique Labels:")
print(y.unique())

#########################
# TRAIN / TEST SPLIT
#########################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("Training Shape:")
print(X_train.shape)

print("Testing Shape:")
print(X_test.shape)

#####################
# TREE 1
# GINI + SHALLOW TREE
#####################
tree1 = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

tree1.fit(X_train, y_train)
pred1 = tree1.predict(X_test)
acc1 = accuracy_score(y_test, pred1)

print("\n*************TREE 1 RESULTS*************")
print(f"Accuracy: {acc1:.4f}")
print("\nClassification:")
print(classification_report(y_test, pred1))

######################
# TREE 1 CONFUSION MATRIX
######################
cm1 = confusion_matrix(y_test, pred1)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm1,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=tree1.classes_,
    yticklabels=tree1.classes_
)

plt.title("Tree 1 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

tree1_cm_path = SAVED_IMGS / "cicd_tree1_confusion_matrix.png"
plt.savefig(
    tree1_cm_path,
    bbox_inches='tight',
    dpi=300
)

plt.close()

#####################
# TREE 1 VISUALIZATION
######################

plt.figure(figsize=(28, 14))
plot_tree(
    tree1,
    feature_names=X.columns,
    class_names=[str(c) for c in tree1.classes_],
    filled=True,
    fontsize=7,
    max_depth=4
)

plt.title("Decision Tree 1 (Gini, max_depth=4)")

tree1_plot_path = SAVED_IMGS / "cicd_tree1_visualization.png"

plt.savefig(
    tree1_plot_path,
    bbox_inches='tight',
    dpi=300
)

plt.close()


########################
# TREE 2
# ENTROPY + DEEPER TREE
########################

tree2 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=8,
    min_samples_split=20,
    random_state=42
)

tree2.fit(X_train, y_train)
pred2 = tree2.predict(X_test)
acc2 = accuracy_score(y_test, pred2)

print("\n*************TREE 2 RESULTS******************")
print(f"Accuracy: {acc2:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, pred2))

############################
# TREE 2 CONFUSION MATRIX
############################

cm2 = confusion_matrix(y_test, pred2)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm2,
    annot=True,
    fmt='d',
    cmap='Greens',
    xticklabels=tree2.classes_,
    yticklabels=tree2.classes_
)

plt.title("Tree 2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

tree2_cm_path = SAVED_IMGS / "cicd_tree2_confusion_matrix.png"

plt.savefig(
    tree2_cm_path,
    bbox_inches='tight',
    dpi=300
)

plt.close()

####################
# TREE 2 VISUAL
####################

plt.figure(figsize=(30, 16))
plot_tree(
    tree2,
    feature_names=X.columns,
    class_names=[str(c) for c in tree2.classes_],
    filled=True,
    fontsize=6,
    max_depth=4
)

plt.title("Decision Tree 2 (Entropy, max_depth=8)")
tree2_plot_path = SAVED_IMGS / "cicd_tree2_visualization.png"

plt.savefig(
    tree2_plot_path,
    bbox_inches='tight',
    dpi=300
)

plt.close()

############################
# RANDOM FOREST MODEL
############################

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=8,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("\n************* RANDOM FOREST RESULTS *************")
print(f"Accuracy: {rf_acc:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred))

############################
# RANDOM FOREST CONFUSION MATRIX
############################
rf_cm = confusion_matrix(y_test, rf_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(
    rf_cm,
    annot=True,
    fmt='d',
    cmap='Purples',
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

rf_cm_path = (
    SAVED_IMGS /
    "cicd_random_forest_confusion_matrix.png"
)

plt.savefig(
    rf_cm_path,
    bbox_inches='tight',
    dpi=300
)

plt.close()

############################
# RANDOM FOREST FEATURE IMPORTANCE
############################

importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
})

importance_df = (
    importance_df
    .sort_values(by='Importance', ascending=False)
    .head(10)
)

plt.figure(figsize=(12, 6))

sns.barplot(
    data=importance_df,
    x='Importance',
    y='Feature',
    palette='Purples'
)

plt.title("Top 10 Random Forest Feature Importances")
plt.xlabel("Importance Score")
plt.ylabel("Feature")

rf_importance_path = (
    SAVED_IMGS /
    "cicd_random_forest_feature_importance.png"
)

plt.savefig(
    rf_importance_path,
    bbox_inches='tight',
    dpi=300
)

plt.close()