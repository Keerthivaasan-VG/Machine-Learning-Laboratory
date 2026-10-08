import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Load Dataset
df = pd.read_csv('health.csv')

# 2. Display Dataset Information
print("Dataset shape:")
print(df.shape)
print()
print("First 5 rows:")
print(df.head())
print()
print("Target distribution:")
print(df['HeartDisease'].value_counts())
print()

# Warning message as shown in lab output
if len(df) < 50:
    print("WARNING:")
    print(f"Your health.csv contains only {len(df)} rows.")
    print("For meaningful machine-learning results, use a much larger dataset.")
    print()

# 3. Features and Target Separation
X = df.drop(columns=['HeartDisease'])
y = df['HeartDisease']

# 4. Train-Test Split (6 training, 3 testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=3, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print()

# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ========================================
# 5. SVM CLASSIFIER
# ========================================
svm_model = SVC(kernel='linear', C=1.0, random_state=42)
svm_model.fit(X_train_scaled, y_train)

y_pred_svm = svm_model.predict(X_test_scaled)
svm_acc = accuracy_score(y_test, y_pred_svm) * 100

print("=" * 40)
print("SVM RESULTS")
print("=" * 40)
print(f"SVM Accuracy: {svm_acc:.1f} %")
print()

# ========================================
# 6. HYBRID SVM + MLP CLASSIFIER
# ========================================
mlp_model = MLPClassifier(hidden_layer_sizes=(8,), max_iter=300, random_state=21)
mlp_model.fit(X_train_scaled, y_train)

y_pred_mlp = mlp_model.predict(X_test_scaled)

hybrid_preds = np.where(y_pred_svm == y_pred_mlp, y_pred_svm, y_pred_mlp)
hybrid_acc = accuracy_score(y_test, hybrid_preds) * 100

print("=" * 40)
print("HYBRID SVM + MLP RESULTS")
print("=" * 40)
print(f"Hybrid Accuracy: {hybrid_acc:.2f} %")

# ========================================
# 7. CONFUSION MATRIX PLOT
# ========================================
cm = confusion_matrix(y_test, y_pred_svm)
display_labels = ['No Heart Disease', 'Heart Disease']

fig, ax = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
disp.plot(cmap=plt.cm.Blues, ax=ax, colorbar=True)

ax.set_title("SVM Confusion Matrix")
plt.tight_layout()
plt.show()
