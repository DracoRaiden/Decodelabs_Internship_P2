import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def main():

    # Top-level description
    print("--- DecodeLabs Project 2: Data Classification Pipeline ---\n")

    # 1) Load dataset from CSV
    # Note: this expects a file named 'iris.csv' in the current working directory.
    print("[1/5] Loading Iris Benchmark Dataset...")
    df = pd.read_csv('iris.csv')

    # 2) Detect which column contains the class/target labels
    # We try common label names (case-sensitive). If none match, fall back
    # to the last column in the DataFrame.
    label_candidates = ['species', 'Species', 'label', 'Label', 'target', 'class', 'Class']
    label_col = None
    for cand in label_candidates:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[-1]

    # 3) Split into features (X) and target (y)
    # `X` is a DataFrame of feature columns; `y` is a Series with labels.
    X = df.drop(label_col, axis=1)
    y = df[label_col]

    # 4) Shuffle and split into training and test sets (80/20)
    print("[2/5] Shuffling and splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

    # 5) Scale features to zero mean and unit variance for KNN
    print("[3/5] Applying StandardScaler (Mean=0, Variance=1)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 6) Train a K-Nearest Neighbors classifier (k=5)
    print("[4/5] Instantiating and Fitting K-Nearest Neighbors (K=5)...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    # 7) Predict labels for the held-out test set
    print("[*] Predicting unseen data...")
    predictions = model.predict(X_test_scaled)

    # 8) Evaluation: confusion matrix and classification report
    print("[5/5] Output Validation Pipeline Complete. Generating Metrics...\n")
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix (rows=true classes, columns=predicted classes):")
    print(conf_matrix)
    print("\n")

    # `classification_report` accepts `target_names` as a list of class labels
    # in the same order as the integer-labelled classes passed to the function.
    # We use `model.classes_` to get the class ordering learned by the model
    # (this preserves consistency between predictions and labels).
    print("Strategic Trade-offs Report (Precision, Recall, F1 Score):")
    labels = model.classes_
    report = classification_report(y_test, predictions, target_names=[str(c) for c in labels])
    print(report)

if __name__ == "__main__":
    main()