

# STEP 1 & 2: Load and Combine Data

import pandas as pd

trail1 = pd.read_csv("Trail1_extracted_features_acceleration_m1ai1-1.csv")
trail2 = pd.read_csv("Trail2_extracted_features_acceleration_m1ai1.csv")
trail3 = pd.read_csv("Trail3_extracted_features_acceleration_m2ai0.csv")

df = pd.concat([trail1, trail2, trail3], ignore_index=True)

print(f"Combined dataset shape: {df.shape}")
print(f"Event distribution:\n{df['event'].value_counts()}")

# STEP 3 & 4: Remove Columns & Encode Labels

columns_to_remove = ["start_time", "axle", "cluster", "tsne_1", "tsne_2"]
df = df.drop(columns=columns_to_remove)

df["event"] = df["event"].apply(lambda x: 0 if str(x).strip().lower() == "normal" else 1)

print("Class distribution after encoding:")
print(f"Normal (0): {(df['event'] == 0).sum()}")
print(f"Event  (1): {(df['event'] == 1).sum()}")

# STEP 5: Separate Features and Normalize

from sklearn.preprocessing import StandardScaler

X = df.drop(columns=["event"])
y = df["event"]

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

print(f"Before scaling - Mean: {X.mean().mean():.4f}, Std: {X.std().mean():.4f}")
print(f"After scaling  - Mean: {X_scaled.mean().mean():.4f}, Std: {X_scaled.std().mean():.4f}")

# STEP 6: 80/20 Train-Test Split

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set:     {X_test.shape[0]} samples")
print()

# STEP 7: Train SVM + Evaluate (80/20)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

svm_model = SVC(kernel="rbf", random_state=42)
svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)
accuracy_8020 = accuracy_score(y_test, y_pred)

print(f"SVM Accuracy (80/20 split): {accuracy_8020:.4f}")
print(classification_report(y_test, y_pred, target_names=["Normal", "Event"]))
print()

# STEP 8: 5-Fold Cross-Validation

from sklearn.model_selection import StratifiedKFold, cross_val_score

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_scores = cross_val_score(
    SVC(kernel="rbf", random_state=42),
    X_scaled, y,
    cv=cv,
    scoring="accuracy"
)
print("K-Fold Cross-Validation Results (5-Fold):")
for i, score in enumerate(cv_scores, 1):
    print(f"Fold {i}: {score:.4f}")

cv_mean = cv_scores.mean()
cv_std = cv_scores.std()

print(f"\nMean Accuracy: {cv_mean:.4f} +/- {cv_std:.4f}")

# Accuracy Comparison

import matplotlib.pyplot as plt

methods = ["80/20 Test Accuracy", "5-Fold CV Mean"]
values = [accuracy_8020, cv_mean]
plt.figure()
plt.bar(methods, values)
plt.ylim(0, 1)
plt.ylabel("Accuracy")
plt.title("Comparison of SVM Performance")
plt.show()