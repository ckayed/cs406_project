from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.ensemble import RandomForestClassifier

ROOT_DIR = Path(r"C:\Users\Kaye\OneDrive\Documentos\CS406")

TRAIN_SET = (ROOT_DIR /
            "data_sets" /
            "UNSWNB15" /
            "unsw_nb15_training-set.csv"
            )

TEST_SET = (ROOT_DIR /
            "data_sets" /
            "UNSWNB15" /
            "unsw_nb15_testing-set.csv"
            )

SAVED_IMGS = (ROOT_DIR /
            "decisionTrees_randomForest" /
            "saved_images"
            )

SAVED_IMGS.mkdir(parents=True, exist_ok=True)

print("Loading UNSW-NB15 datasets.")

train_df = pd.read_csv(TRAIN_SET)
test_df = pd.read_csv(TEST_SET)

print("\nTrain Dataset:")
print(train_df.head())

print("\nTest Dataset:")
print(test_df.head())

###########################
# CLEAN TRAIN DATA
###########################
print("\n******** CLEANING TRAIN DATA ********")

train_df.columns = train_df.columns.str.strip()

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)

train_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

if "label" not in train_df.columns:
    raise ValueError("label column not found in train data.")

train_df.dropna(axis=1, how="all", inplace=True)

if "id" in train_df.columns:
    train_df.drop(columns=["id"], inplace=True)

if "attack_cat" in train_df.columns:

    print("\nTrain Attack Categories:")
    print(train_df["attack_cat"].value_counts())

    train_df.drop(columns=["attack_cat"], inplace=True)

train_df.dropna(subset=["label"], inplace=True)

# Convert ALL feature columns to numeric FIRST
train_features = [
    col for col in train_df.columns if col != "label"
]

for col in train_features:

    train_df[col] = pd.to_numeric(
        train_df[col],
        errors="coerce"
    )

# Fill missing numeric values
train_df[train_features] = (
    train_df[train_features].fillna(0)
)

# Remove duplicates
train_df.drop_duplicates(inplace=True)

print("\nTrain Shape After Cleaning:")
print(train_df.shape)

print("\nTrain First Cleaned Rows:")
print(train_df.head())

###########################
# CLEAN TEST DATA
###########################
print("\n******** CLEANING TEST DATA ********")

test_df.columns = test_df.columns.str.strip()

test_df.replace([np.inf, -np.inf], np.nan, inplace=True)

test_df.replace(r"^\s*$", np.nan, regex=True, inplace=True)

if "label" not in test_df.columns:
    raise ValueError("label column not found in test data.")

test_df.dropna(axis=1, how="all", inplace=True)

if "id" in test_df.columns:
    test_df.drop(columns=["id"], inplace=True)

if "attack_cat" in test_df.columns:

    print("\nTest Attack Categories:")
    print(test_df["attack_cat"].value_counts())

    test_df.drop(columns=["attack_cat"], inplace=True)

test_df.dropna(subset=["label"], inplace=True)

# Convert ALL feature columns to numeric FIRST
test_features = [
    col for col in test_df.columns if col != "label"
]

for col in test_features:

    test_df[col] = pd.to_numeric(
        test_df[col],
        errors="coerce"
    )

# Fill missing numeric values
test_df[test_features] = (
    test_df[test_features].fillna(0)
)

# Remove duplicates
test_df.drop_duplicates(inplace=True)

print("\nTest Shape After Cleaning:")
print(test_df.shape)

print("\nTest First Cleaned Rows:")
print(test_df.head())

###########################
# PREP TRAIN AND TEST DATA
###########################

label_col = "label"

X_train = train_df.drop(label_col, axis=1)
y_train = train_df[label_col]

X_test = test_df.drop(label_col, axis=1)
y_test = test_df[label_col]

# Convert labels to string
y_train = y_train.astype(str)
y_test = y_test.astype(str)

# Keep only numeric columns AFTER conversion
X_train = X_train.select_dtypes(include=[np.number])
X_test = X_test.select_dtypes(include=[np.number])

# Match train/test columns
X_train, X_test = X_train.align(
    X_test,
    join="inner",
    axis=1
)

print("\nColumns in X_train:")
print(X_train.columns)

print("\nColumns in X_test:")
print(X_test.columns)

print("\nTraining Feature Shape:")
print(X_train.shape)

print("\nTesting Feature Shape:")
print(X_test.shape)

print("\nTraining Labels:")
print(y_train.unique())

print("\nTesting Labels:")
print(y_test.unique())

###########################
# TREE 1
# GINI + SHALLOW TREE
###########################

tree1 = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    random_state=42
)

tree1.fit(X_train, y_train)

pred1 = tree1.predict(X_test)

acc1 = accuracy_score(y_test, pred1)

print("\n************* TREE 1 RESULTS *************")

print(f"Accuracy: {acc1:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred1))

###########################
# TREE 1 CONFUSION MATRIX
############################

cm1 = confusion_matrix(y_test, pred1)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm1,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=tree1.classes_,
    yticklabels=tree1.classes_
)

plt.title("UNSW-NB15 Tree 1 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

tree1_cm_path = (
    SAVED_IMGS /
    "unsw_tree1_confusion_matrix.png"
)

plt.savefig(
    tree1_cm_path,
    bbox_inches="tight",
    dpi=300
)

plt.close()


###########################
# TREE 2
###########################

tree2 = DecisionTreeClassifier(
    criterion="entropy",
    max_depth=8,
    min_samples_split=20,
    random_state=42
)

tree2.fit(X_train, y_train)

pred2 = tree2.predict(X_test)

acc2 = accuracy_score(y_test, pred2)

print("\n************* TREE 2 RESULTS *************")

print(f"Accuracy: {acc2:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, pred2))

###########################
# TREE 2 CONFUSION MATRIX
###########################

cm2 = confusion_matrix(y_test, pred2)

plt.figure(figsize=(10, 8))

sns.heatmap(
    cm2,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=tree2.classes_,
    yticklabels=tree2.classes_
)

plt.title("UNSW-NB15 Tree 2 Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

tree2_cm_path = (
    SAVED_IMGS /
    "unsw_tree2_confusion_matrix.png"
)

plt.savefig(
    tree2_cm_path,
    bbox_inches="tight",
    dpi=300
)

plt.close()

############################
# RANDOM FOREST MODEL
###########################

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
########################

rf_cm = confusion_matrix(y_test, rf_pred)

plt.figure(figsize=(10, 8))

sns.heatmap(
    rf_cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=rf_model.classes_,
    yticklabels=rf_model.classes_
)

plt.title("UNSW-NB15 Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

rf_cm_path = (
    SAVED_IMGS /
    "unsw_random_forest_confusion_matrix.png"
)

plt.savefig(
    rf_cm_path,
    bbox_inches="tight",
    dpi=300
)

plt.close()