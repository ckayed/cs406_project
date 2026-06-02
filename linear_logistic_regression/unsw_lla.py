from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             mean_absolute_error, mean_squared_error,
                             r2_score)
################################

TRAIN_FILE = Path(
    r"C:\Users\Kaye\OneDrive\Documentos\CS406\data_sets\UNSWNB15\UNSW_NB15_training-set.csv"
    )

TEST_FILE = Path(
    r"C:\Users\Kaye\OneDrive\Documentos\CS406\data_sets\UNSWNB15\UNSW_NB15_testing-set.csv"
    )

SAVED_IMGS = Path(
    r"C:\Users\Kaye\OneDrive\Documentos\CS406\linear_logistic_regression"
    )

SAVED_IMGS = SAVED_IMGS / "saved_images"
SAVED_IMGS.mkdir(parents=True, exist_ok=True)
SAVED_IMGS.mkdir(parents=True, exist_ok=True)

print("\nLoading UNSW-NB15 training and testing datasets...")

train_df = pd.read_csv(TRAIN_FILE)
test_df = pd.read_csv(TEST_FILE)

df = pd.concat(
    [train_df, test_df],
    ignore_index=True
)

df.columns = df.columns.str.strip()

print("\nDataset Loaded Successfully")
print("Dataset Shape:", df.shape)
print(df.head())

if "label" not in df.columns:
    raise Exception("The dataset must contain a 'label' column.")

print("\nLabels Found:")
print(df["label"].value_counts())

########################
# Cleaning
######################

df.replace([np.inf, -np.inf], np.nan, inplace=True)

num_df = df.select_dtypes(include=[np.number]).copy()

num_colmns = num_df.select_dtypes(include=[np.number]).columns
num_df[num_colmns] = num_df[num_colmns].fillna(
    num_df[num_colmns].median()
)

df = num_df.copy()

print("\nCleaned Dataset Shape:", df.shape)

############
# Binary label
####################

df["Binary_Label"] = df["label"].astype(int)

print("\nBinary Label Counts:")
print(df["Binary_Label"].value_counts())

if df["Binary_Label"].nunique() < 2:
    raise Exception(
        "Logistic Regression needs both classes. "
        "Your Binary_Label still only has one class."
    )

used_feats = [
    "dur",
    "sbytes",
    "dbytes",
    "sttl",
    "dttl",
    "sload",
    "dload",
    "spkts",
    "dpkts",
    "rate"
    ]

feat_colmns = [
    col for col in used_feats
    if col in df.columns
]

if len(feat_colmns) < 3:
    feat_colmns = [
        col for col in df.select_dtypes(include=[np.number]).columns
        if col not in ["label", "Binary_Label"]
    ][:10]

if len(feat_colmns) < 3:
    raise Exception("Not enough numeric features found for modeling.")

print("\nFeatures Used:")
for feature in feat_colmns:
    print("-", feature)

###################
# Sample seed
########################

samp_seed = 200000

if len(df) > samp_seed:
    print(f"\nSampling {samp_seed} rows for faster training...")
    df = df.sample(
        n=samp_seed,
        random_state=42
    )

print("\nModeling Dataset Shape:", df.shape)

#######################
# Linear Regression
######################

print("**********LINEAR REGRESSION*************")

linear_target = "rate"

if linear_target not in df.columns:
    linear_target = feat_colmns[-1]

linear_features = [
    col for col in feat_colmns
    if col != linear_target
][:5]

X_linear = df[linear_features]
y_linear = df[linear_target]

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(
    X_linear,
    y_linear,
    test_size=0.20,
    random_state=42
)

linear_model = LinearRegression()
linear_model.fit(X_train_lr, y_train_lr)

linear_pred = linear_model.predict(X_test_lr)

mae = mean_absolute_error(y_test_lr, linear_pred)
mse = mean_squared_error(y_test_lr, linear_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test_lr, linear_pred)

equation = f"{linear_target} = {linear_model.intercept_:.4f}"

for feature, coef in zip(linear_features, linear_model.coef_):
    equation += f" + ({coef:.4f} * {feature})"

print("\nLinear Regression Equation:")
print(equation)

print("\nLinear Regression Metrics:")
print(f"MAE: {mae:.4f}")
print(f"MSE: {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R2: {r2:.4f}")

example_row = X_test_lr.iloc[0]
manual_pred = linear_model.intercept_

print("\n(EXAMPLE) Linear Regression Calculation:")
print("Using one test row:")

for feature, coef in zip(linear_features, linear_model.coef_):
    value = example_row[feature]
    manual_pred += coef * value
    print(f"({coef:.4f} * {value:.4f})")

print(f"\nPredicted {linear_target}: {manual_pred:.4f}")
print(f"Actual {linear_target}: {y_test_lr.iloc[0]:.4f}")

plt.figure(figsize=(8, 6))

plt.scatter(
    np.log1p(y_test_lr),
    np.log1p(linear_pred),
    alpha=0.5
)

plt.xlabel("Log(Actual rate)")
plt.ylabel("Log(Predicted rate)")
plt.title("UNSW Linear Regression: Log Actual vs Predicted")
plt.grid(True)

linear_visual_path = (
    SAVED_IMGS / "UNSW_linear_regression_log_actual_vs_predicted.png"
)

plt.savefig(
    linear_visual_path,
    bbox_inches="tight"
)

plt.close()

########################
# Logisitc Regression
########################

print("\n**********LOGISTIC REGRESSION*************")

X_logistic = df[feat_colmns]
y_logistic = df["Binary_Label"]

X_train_log, X_test_log, y_train_log, y_test_log = train_test_split(
    X_logistic,
    y_logistic,
    test_size=0.20,
    random_state=42,
    stratify=y_logistic
)

scaler = StandardScaler()

X_train_log_scaled = scaler.fit_transform(X_train_log)
X_test_log_scaled = scaler.transform(X_test_log)

logistic_model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced"
)

logistic_model.fit(X_train_log_scaled, y_train_log)

logistic_pred = logistic_model.predict(X_test_log_scaled)
logistic_prob = logistic_model.predict_proba(X_test_log_scaled)[:, 1]

accuracy = accuracy_score(y_test_log, logistic_pred)

log_equation = f"z = {logistic_model.intercept_[0]:.4f}"

for feature, coef in zip(feat_colmns, logistic_model.coef_[0]):
    log_equation += f" + ({coef:.4f} * {feature})"

print("\nLogistic Regression Equation:")
print(log_equation)

print("\nSigmoid Formula:")
print("p = 1 / (1 + e^(-z))")

example_scaled = X_test_log_scaled[0]

z_value = logistic_model.intercept_[0] + np.dot(
    logistic_model.coef_[0],
    example_scaled
)

sigmoid_probability = 1 / (1 + np.exp(-z_value))

print("\nExample Logistic Regression Calculation:")
print(f"z value: {z_value:.4f}")
print(f"p = 1 / (1 + e^(-{z_value:.4f}))")
print(f"Probability of attack traffic: {sigmoid_probability:.4f}")

if sigmoid_probability >= 0.5:
    print("Prediction: ATTACK")
else:
    print("Prediction: NORMAL")

actual_label = "ATTACK" if y_test_log.iloc[0] == 1 else "NORMAL"
print(f"Actual Label: {actual_label}")

print("\nLogistic Regression Accuracy:")
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")

report = classification_report(
    y_test_log,
    logistic_pred,
    target_names=["Normal", "Attack"]
)

print(report)

cm = confusion_matrix(y_test_log, logistic_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["Normal", "Attack"]
)

disp.plot(values_format="d")
plt.title("UNSW Logistic Regression Confusion Matrix")

confusion_matrix_path = (
    SAVED_IMGS / "unsw_logistic_regression_confusion_matrix.png"
)

plt.savefig(
    confusion_matrix_path,
    bbox_inches="tight"
)

plt.close()