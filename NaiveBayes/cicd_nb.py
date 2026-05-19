from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix)
from sklearn.model_selection import train_test_split

root_dir = Path(r"C:\Users\Kaye\OneDrive\Documentos\CS406")

cicd_dataset = (
    root_dir /
    "data_sets" /
    "CICD2017" /
    "cicids2017_tues_wed_fri_cleaned.csv"
)

saved_imgs = (
    root_dir /
    "NaiveBayes_SVM" /
    "saved_images"
)

saved_imgs.mkdir(parents=True, exist_ok=True)

cicd_df = pd.read_csv(cicd_dataset)

print("\nOriginal CICIDS2017 Dataset Shape:")
print(cicd_df.shape)

print("\nOriginal Dataset Preview:")
print(cicd_df.head(10))

cicd_df.columns = cicd_df.columns.str.strip()

target_colmn = "Label"

print("\nOriginal Label Counts:")
print(cicd_df[target_colmn].value_counts())

# Change labels to binary
# BENIGN = 0
# ATTACK = 1
cicd_df[target_colmn] = (
    cicd_df[target_colmn]
    .astype(str)
    .str.strip()
    .str.upper()
)

cicd_df[target_colmn] = np.where(
    cicd_df[target_colmn] == "BENIGN",
    0,
    1
)

print("\nUnique Labels:")
print(cicd_df[target_colmn].unique())

print("\nBinary Label Counts:")
print(cicd_df[target_colmn].value_counts())

# Remove infinite vals
cicd_df.replace([np.inf, -np.inf], np.nan, inplace=True)

# Fill missing vals
cicd_df = cicd_df.fillna(0)

print("\n******Cleaned CICIDS2017 Dataset:******")
print(cicd_df.head(10))

print("\nDataset Shape After Cleaning:")
print(cicd_df.shape)

###########################
# Split fts
###########################

X = cicd_df.drop(columns=[target_colmn])
y = cicd_df[target_colmn]

cat_colmns = X.select_dtypes(include=["object"]).columns.tolist()
num_colmns = X.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical Columns:")
print(cat_colmns)

print("\nNumeric Columns:")
print(num_colmns)

###########################
# Train test split
###########################

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:")
print(X_train.shape)

print("\nTesting Shape:")
print(X_test.shape)

print("\nTraining Label Counts:")
print(y_train.value_counts())

print("\nTesting Label Counts:")
print(y_test.value_counts())

###########################
# Gaussian NB
###########################

gaus_preproc = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            cat_colmns
        ),
        (
            "num",
            "passthrough",
            num_colmns
        )
    ]
)

X_train_gaus = gaus_preproc.fit_transform(X_train)
x_test_gaus = gaus_preproc.transform(X_test)

gaus_ft_names = gaus_preproc.get_feature_names_out()

gaus_form_df = pd.DataFrame(
    X_train_gaus,
    columns=gaus_ft_names
)

print("\n******Dataset for Gaussian Naive Bayes:******")
print(gaus_form_df.head(10))

gaus_nb = GaussianNB()
gaus_nb.fit(X_train_gaus, y_train)

gaus_pred = gaus_nb.predict(x_test_gaus)
gaus_prob = gaus_nb.predict_proba(x_test_gaus)

gaus_accu = accuracy_score(y_test, gaus_pred)

print("\n******Gaussian Naive Bayes Results******")
print("Accuracy:", gaus_accu)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        gaus_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

gaus_prob_df = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": gaus_pred[:10]
})

for i, class_label in enumerate(gaus_nb.classes_):
    if class_label == 0:
        gaus_prob_df["P(BENIGN)"] = gaus_prob[:10, i]
    elif class_label == 1:
        gaus_prob_df["P(ATTACK)"] = gaus_prob[:10, i]

print("\nGaussian Naive Bayes Probability Examples:")
print(gaus_prob_df)

###########################
# Gaussian Conf M
###########################

gaus_confM = confusion_matrix(
    y_test,
    gaus_pred,
    labels=[0, 1]
)

print("\nGaussian Naive Bayes Confusion Matrix:")
print(gaus_confM)

plt.figure(figsize=(7, 6))

sns.heatmap(
    gaus_confM,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["BENIGN", "ATTACK"],
    yticklabels=["BENIGN", "ATTACK"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Gaussian Naive Bayes Confusion Matrix")

plt.tight_layout()

plt.savefig(
    saved_imgs /
    "cicd_gaussian_nb_confusion_matrix.png"
)

plt.show()

###########################
# Gaussian Pred Dist
###########################

plt.figure(figsize=(7, 5))

sns.countplot(
    x=gaus_pred,
    order=[0, 1],
    palette="Blues"
)

plt.xticks([0, 1], ["BENIGN", "ATTACK"])

plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Gaussian Naive Bayes Prediction Distribution")

plt.tight_layout()

plt.savefig(
    saved_imgs /
    "cicd_gaussian_nb_prediction_distribution.png"
)

plt.show()

###########################
# Multinomial NB
###########################

multiN_preproc = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            cat_colmns
        ),
        (
            "num",
            MinMaxScaler(),
            num_colmns
        )
    ]
)

x_train_multiN = multiN_preproc.fit_transform(X_train)
x_test_multiN = multiN_preproc.transform(X_test)

multiN_ft_names = multiN_preproc.get_feature_names_out()

multiN_form_df = pd.DataFrame(
    x_train_multiN,
    columns=multiN_ft_names
)

print("\n******Dataset for Multinomial Naive Bayes:******")
print(multiN_form_df.head(10))

multiN_nb = MultinomialNB()
multiN_nb.fit(x_train_multiN, y_train)

multiN_pred = multiN_nb.predict(x_test_multiN)
multiN_prob = multiN_nb.predict_proba(x_test_multiN)

multiN_accu = accuracy_score(y_test, multiN_pred)

print("\n******Multinomial Naive Bayes Results******")
print("Accuracy:", multiN_accu)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        multiN_pred,
        labels=[0, 1],
        target_names=["BENIGN", "ATTACK"],
        zero_division=0
    )
)

multiN_prob_df = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": multiN_pred[:10]
})

for i, class_label in enumerate(multiN_nb.classes_):
    if class_label == 0:
        multiN_prob_df["P(BENIGN)"] = multiN_prob[:10, i]
    elif class_label == 1:
        multiN_prob_df["P(ATTACK)"] = multiN_prob[:10, i]

print("\nMultinomial Naive Bayes Probability Examples:")
print(multiN_prob_df)

###########################
# MultiNomial Conf M
###########################

multiN_confM = confusion_matrix(
    y_test,
    multiN_pred,
    labels=[0, 1]
)

print("\nMultinomial Naive Bayes Confusion Matrix:")
print(multiN_confM)

plt.figure(figsize=(7, 6))

sns.heatmap(
    multiN_confM,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["BENIGN", "ATTACK"],
    yticklabels=["BENIGN", "ATTACK"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Multinomial Naive Bayes Confusion Matrix")

plt.tight_layout()

plt.savefig(
    saved_imgs /
    "cicd_multinomial_nb_confusion_matrix.png"
)

plt.show()

###########################
# MultiNomial Dist pred
###########################

plt.figure(figsize=(7, 5))

sns.countplot(
    x=multiN_pred,
    order=[0, 1],
    palette="Greens"
)

plt.xticks([0, 1], ["BENIGN", "ATTACK"])

plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Multinomial Naive Bayes Prediction Distribution")

plt.tight_layout()

plt.savefig(
    saved_imgs /
    "cicd_multinomial_nb_prediction_distribution.png"
)

plt.show()