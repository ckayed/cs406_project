from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import ( accuracy_score, classification_report,
                             confusion_matrix )

root_dir = Path(r"C:\Users\Kaye\OneDrive\Documentos\CS406")

train_set = root_dir / "data_sets" / "UNSWNB15" / "unsw_nb15_training-set.csv"
test_set = root_dir / "data_sets" / "UNSWNB15" / "unsw_nb15_testing-set.csv"

saved_imgs = root_dir / "NaiveBayes_SVM" / "saved_images"
saved_imgs.mkdir(parents=True, exist_ok=True)

train_df = pd.read_csv(train_set)
test_df = pd.read_csv(test_set)

print("\nOriginal Training Shape:", train_df.shape)
print("Original Testing Shape:", test_df.shape)

target_colmn = "label"

train_df = train_df.drop(
    columns=[c for c in ["id", "attack_cat"] if c in train_df.columns]
)

test_df = test_df.drop(
    columns=[c for c in ["id", "attack_cat"] if c in test_df.columns]
)

train_df = train_df.fillna(0)
test_df = test_df.fillna(0)

print("\n******Cleaned Training Dataset:******")
print(train_df.head(10))

print("\n******Cleaned Testing Dataset:******")
print(test_df.head(10))

X_train = train_df.drop(columns=[target_colmn])
y_train = train_df[target_colmn]

X_test = test_df.drop(columns=[target_colmn])
y_test = test_df[target_colmn]

cat_colmns = X_train.select_dtypes(include=["object"]).columns.tolist()
num_colmns = X_train.select_dtypes(include=[np.number]).columns.tolist()

print("\nCategorical Columns:")
print(cat_colmns)

print("\nNumeric Columns:")
print(num_colmns)

print("\nAfter Cleaning Training Shape:", train_df.shape)
print("After Cleaning Testing Shape:", test_df.shape)

###############
# Gaussian NB
###############

gaus_preproc = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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

print("\nDataset for Gaussian Naive Bayes:")
print(gaus_form_df.head(10))

gaus_nb = GaussianNB()
gaus_nb.fit(X_train_gaus, y_train)

gaus_pred = gaus_nb.predict(x_test_gaus)
gaus_prob = gaus_nb.predict_proba(x_test_gaus)

gaus_accu = accuracy_score(y_test, gaus_pred)

print("\n******Gaussian Naive Bayes Results******")
print("Accuracy:", gaus_accu)

print("\nClassification Report:")
print(classification_report(y_test, gaus_pred))

gaus_prob_df = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": gaus_pred[:10],
    "P(Normal)": gaus_prob[:10, 0],
    "P(Attack)": gaus_prob[:10, 1]
})

print("\nGaussian Naive Bayes Probability Examples:")
print(gaus_prob_df)

######################
# Gaussian NB: Confusion Matrix
######################

gaus_confM = confusion_matrix(y_test, gaus_pred)

print("\nGaussian Naive Bayes Confusion Matrix:")
print(gaus_confM)

plt.figure(figsize=(7, 6))
sns.heatmap(
    gaus_confM,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Gaussian Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.savefig(saved_imgs / "unsw_gaus_nb_confusion_matrix.png")
plt.show()

######################
# GAUSSIAN NB VISUAL 2: PREDICTION
######################

plt.figure(figsize=(7, 5))

sns.countplot(
    x=gaus_pred,
    palette="Blues"
)

plt.xticks([0, 1], ["Normal", "Attack"])

plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Gaussian Naive Bayes Prediction Distribution")

plt.tight_layout()

plt.savefig(
    saved_imgs / "unsw_gaus_nb_prediction_distribution.png"
)

plt.show()

###############
# Multinomial NB
###############

multiN_preproc = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
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
print(classification_report(y_test, multiN_pred))

multiN_prob_df = pd.DataFrame({
    "Actual": y_test.iloc[:10].values,
    "Predicted": multiN_pred[:10],
    "P(Normal)": multiN_prob[:10, 0],
    "P(Attack)": multiN_prob[:10, 1]
})

print("\n Probability Examples:")
print(multiN_prob_df)

######################
# MULTINOMIAL NB: Confusion Matrix
######################

multiN_confM = confusion_matrix(y_test, multiN_pred)

print("\nMultinomial Naive Bayes Confusion Matrix:")
print(multiN_confM)

plt.figure(figsize=(7, 6))
sns.heatmap(
    multiN_confM,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=["Normal", "Attack"],
    yticklabels=["Normal", "Attack"]
)

plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Multinomial Naive Bayes Confusion Matrix")
plt.tight_layout()
plt.savefig(saved_imgs / "unsw_multiN_nb_confusion_matrix.png")
plt.show()

######################
# MULTINOMIAL NB VISUAL 2: PREDICTION
######################

plt.figure(figsize=(7, 5))

sns.countplot(
    x=multiN_pred,
    palette="Greens"
)

plt.xticks([0, 1], ["Normal", "Attack"])

plt.xlabel("Predicted Class")
plt.ylabel("Count")
plt.title("Multinomial Naive Bayes Prediction Distribution")

plt.tight_layout()

plt.savefig(
    saved_imgs / "unsw_multiN_nb_prediction_distribution.png"
)

plt.show()