from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.neighbors import kneighbors_graph
from scipy.cluster.hierarchy import dendrogram, linkage

###############################
# Dataset Loading
###############################
root_folder = Path(r"C:\Users\Kaye\OneDrive\Documentos\CS406\clustering_and_PCA")
cicd_data = Path(r"C:\Users\Kaye\OneDrive\Documentos\CS406\data_sets\CICD2017")
saved_imgs = root_folder / "saved_images"
saved_imgs.mkdir(parents=True, exist_ok=True)

tues_file = cicd_data / "cicids2017_tuesday_cleaned.csv"
wed_file = cicd_data / "cicids2017_wednesday_cleaned.csv"
fri_file = cicd_data / "cicids2017_friday_cleaned.csv"

print("\nChecking dataset paths...")
print("\nMay take time due to file size...")
print("\nTuesday exists:", tues_file.exists())
print("Wednesday exists:", wed_file.exists())
print("Friday exists:", fri_file.exists())

tues_df = pd.read_csv(tues_file)
wed_df = pd.read_csv(wed_file)
fri_data = pd.read_csv(fri_file)

print("\nDatasets loaded:")
print("Tuesday:", tues_df.shape)
print("Wednesday:", wed_df.shape)
print("Friday:", fri_data.shape)


###############################
# combining data
###############################

common_colmn = list(
    set(tues_df.columns)
    .intersection(set(wed_df.columns))
    .intersection(set(fri_data.columns))
)

print("\nCommon columns:", len(common_colmn))

tues_df = tues_df[common_colmn].copy()
wed_df = wed_df[common_colmn].copy()
fri_data = fri_data[common_colmn].copy()

tues_df["Dataset_Day"] = "Tuesday"
wed_df["Dataset_Day"] = "Wednesday"
fri_data["Dataset_Day"] = "Friday"

cicd_DF = pd.concat(
    [tues_df, wed_df, fri_data],
    axis=0,
    ignore_index=True
)

print("\nCombined shape:", cicd_DF.shape)

comb_file = root_folder / "cicids2017_tues_wed_fri_cleaned.csv"
cicd_DF.to_csv(comb_file, index=False)


###############################
# unsupervised Learning
###############################

print("\nPreparing data for clustering and PCA...")
print("\nMay take time due to file size. . . ")

if "Label" not in cicd_DF.columns:
    raise ValueError("Missing Label column.")

y = cicd_DF["Label"]

X = cicd_DF.drop(columns=["Label", "Dataset_Day"], errors="ignore")
X = X.select_dtypes(include=[np.number])

X.replace([np.inf, -np.inf], np.nan, inplace=True)

print("\nRows before cleaning:", X.shape[0])
print("Missing values:", X.isnull().sum().sum())

X = X.fillna(X.mean())

print("Rows after cleaning:", X.shape[0])
print("Columns used:", X.shape[1])

prepared_file = root_folder / "cicids2017_tues_wed_fri_cleaned_fts.csv"
X.to_csv(prepared_file, index=False)


###############################
# Scalign
###############################

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nScaled data shape:", X_scaled.shape)


###############################
# Kmeans
###############################

print("\n\n*******************************************")
print("APPLY KMEANS CLUSTERING")
print("*******************************************")

k_values = [3, 5, 7]
kmeans_sum = []

for k in k_values:
    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=0.0001,
        random_state=42,
        algorithm="lloyd"
    )

    labels = kmeans.fit_predict(X_scaled)

    sil = silhouette_score(
        X_scaled,
        labels,
        sample_size=10000,
        random_state=42
    )

    print("\n***********************************")
    print(f"KMEANS RESULTS WITHOUT PCA: k={k}")
    print("***********************************")
    print("Cluster counts:")
    print(pd.Series(labels).value_counts().sort_index())
    print(f"Silhouette Score: {sil:.4f}")

    kmeans_sum.append([k, sil])

    plt.figure(figsize=(8, 6))
    plt.scatter(
        X_scaled[:, 0],
        X_scaled[:, 1],
        c=labels,
        s=5,
        alpha=0.5
    )
    plt.title(f"KMeans Cluster Visualization Without PCA (k={k})")
    plt.xlabel(X.columns[0])
    plt.ylabel(X.columns[1])
    plt.tight_layout()
    plt.savefig(saved_imgs / f"cicids2017_tues_wed_fri_kmeans_2d_k{k}.png")
    plt.close()

kmeans_sum_df = pd.DataFrame(
    kmeans_sum,
    columns=["k", "silhouette"]
)

kmeans_sum_df.to_csv(
    root_folder / "cicids2017_tues_wed_fri_kmeans_sum.csv",
    index=False
)


###############################
# PCA
###############################

print("\n\n*******************************************")
print("APPLY PCA")
print("*******************************************")

pca = PCA(n_components=3)
Result = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(Result, columns=["PC1", "PC2", "PC3"])

print("\nPCA transformed data:")
print(pca_df.head(15))

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

print("\nTotal variance explained:")
print(pca.explained_variance_ratio_.sum())

print("\nPCA component weights:")
print(pd.DataFrame(
    pca.components_,
    columns=X.columns,
    index=["PC1", "PC2", "PC3"]
))

pca_df.to_csv(
    root_folder / "cicids2017_tues_wed_fri_pca_transformed.csv",
    index=False
)


###############################
# PCA VISUALIZATION 2D
###############################

plt.figure(figsize=(8, 6))
plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    s=5,
    alpha=0.5
)
plt.title("PCA 2D Visualization: PC1 vs PC2")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(saved_imgs / "cicids2017_tues_wed_fri_pca_2d.png")
plt.close()


###############################
# PCA VISUALIZATION 3D
###############################

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection="3d")

ax.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    pca_df["PC3"],
    s=5,
    alpha=0.4
)

ax.set_title("PCA 3D Visualization")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.tight_layout()
plt.savefig(saved_imgs / "cicids2017_tues_wed_fri_pca_3d.png")
plt.close()


###############################
# KMEANS AFTER PCA
###############################

print("\n\n*******************************************")
print("KMEANS AFTER PCA")
print("*******************************************")

pca_kmeans_sum = []

for k in k_values:
    kmeans_pca = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init="auto",
        max_iter=300,
        tol=0.0001,
        random_state=42,
        algorithm="lloyd"
    )

    labels = kmeans_pca.fit_predict(Result)

    sil = silhouette_score(
        Result,
        labels,
        sample_size=10000,
        random_state=42
    )

    print("\n***********************************")
    print(f"KMEANS RESULTS AFTER PCA: k={k}")
    print("***********************************")
    print("Cluster counts:")
    print(pd.Series(labels).value_counts().sort_index())
    print(f"Silhouette Score: {sil:.4f}")

    pca_kmeans_sum.append([k, sil])

pca_kmeans_sum_df = pd.DataFrame(
    pca_kmeans_sum,
    columns=["k", "silhouette"]
)

pca_kmeans_sum_df = pca_kmeans_sum_df.sort_values(
    by="silhouette",
    ascending=False
)

best_k = int(pca_kmeans_sum_df.iloc[0]["k"])

print("\nBest k based on PCA silhouette score:", best_k)

pca_kmeans_sum_df.to_csv(
    root_folder / "cicids2017_tues_wed_fri_pca_kmeans_sum.csv",
    index=False
)


###############################
# Best PCA
###############################

best_model = KMeans(
    n_clusters=best_k,
    init="k-means++",
    n_init="auto",
    max_iter=300,
    tol=0.0001,
    random_state=42,
    algorithm="lloyd"
)

best_labels = best_model.fit_predict(Result)

pca_df["Cluster"] = best_labels

plt.figure(figsize=(8, 6))
plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=pca_df["Cluster"],
    s=5,
    alpha=0.6
)
plt.title(f"KMeans PCA 2D Cluster Visualization (k={best_k})")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(saved_imgs / f"cicids2017_tues_wed_fri_kmeans_pca_2d_k{best_k}.png")
plt.close()

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection="3d")

ax.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    pca_df["PC3"],
    c=pca_df["Cluster"],
    s=5,
    alpha=0.6
)

ax.set_title(f"KMeans PCA 3D Cluster Visualization (k={best_k})")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.tight_layout()
plt.savefig(saved_imgs / f"cicids2017_tues_wed_fri_kmeans_pca_3d_k{best_k}.png")
plt.close()


###############################
# Identifying anomalies
###############################

print("\n\n*******************************************")
print("IDENTIFY ANOMALIES")
print("*******************************************")

distances = best_model.transform(Result)
closest_distance = np.min(distances, axis=1)

threshold = np.percentile(closest_distance, 95)

pca_df["Distance_To_Closest_Cluster"] = closest_distance
pca_df["Possible_Anomaly"] = pca_df["Distance_To_Closest_Cluster"] > threshold

print("\nAnomaly threshold:")
print(threshold)

print("\nPossible anomaly counts:")
print(pca_df["Possible_Anomaly"].value_counts())

anomaly_file = root_folder / "cicids2017_tues_wed_fri_possible_anomalies.csv"
pca_df.to_csv(anomaly_file, index=False)

plt.figure(figsize=(8, 6))
plt.scatter(
    pca_df["PC1"],
    pca_df["PC2"],
    c=pca_df["Possible_Anomaly"],
    s=5,
    alpha=0.6
)
plt.title("Possible Anomalies Based on Cluster Distance")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.savefig(saved_imgs / "cicids2017_tues_wed_fri_possible_anomalies.png")
plt.close()


###############################
# Hierarchial Agglomerative Clustering
##############################

print("\n\n*******************************************")
print("HIERARCHICAL AGGLOMERATIVE CLUSTERING")
print("*******************************************")

np.random.seed(42)

sub_sample = min(50000, Result.shape[0])

idx = np.random.choice(
    Result.shape[0],
    size=sub_sample,
    replace=False
)

X_sub = Result[idx]

print("\nUsing rows for Agglomerative:")
print(X_sub.shape[0])

connectivity = kneighbors_graph(
    X_sub,
    n_neighbors=30,
    include_self=False
)

connectivity = 0.5 * (connectivity + connectivity.T)

for k in k_values:
    agg = AgglomerativeClustering(
        n_clusters=k,
        linkage="ward",
        connectivity=connectivity,
        compute_full_tree=False
    )

    agg_labels = agg.fit_predict(X_sub)

    print(f"\nAgglomerative counts (k={k})")
    print(pd.Series(agg_labels).value_counts().sort_index())


###############################
# Dendogram
###############################

np.random.seed(42)

dend_sample = min(2000, Result.shape[0])

idx = np.random.choice(
    Result.shape[0],
    size=dend_sample,
    replace=False
)

rand_samp = Result[idx, :]

print("\nUsing dendrogram sample size:")
print(dend_sample)

Z = linkage(rand_samp, method="ward")

plt.figure(figsize=(14, 7))
dendrogram(
    Z,
    truncate_mode="lastp",
    p=30,
    leaf_rotation=90,
    leaf_font_size=10,
    show_contracted=True
)

plt.title("Hierarchical Clustering Dendrogram: CICIDS2017 Tues/Wed/Fri")
plt.xlabel("Cluster")
plt.ylabel("Distance")
plt.tight_layout()
plt.savefig(saved_imgs / "cicids2017_tues_wed_fri_hierarchical_dendrogram.png")
plt.close()


###############################
# Summary
###############################

print("\n\n*******************************************")
print("PHASE 1 FINAL SUMMARY")
print("*******************************************")
print("\nBest PCA KMeans k:", best_k)

print("Combined dataset : ", comb_file)
print("Prepared feature file :", prepared_file)
print("PCA transformed file :", root_folder / "cicids2017_tues_wed_fri_pca_transformed.csv")
print("Anomaly file :", anomaly_file)
print("Images :", saved_imgs)
