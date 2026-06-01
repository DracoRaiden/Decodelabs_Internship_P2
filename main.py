import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

def main():

    print("--- DecodeLabs Project 2: Data Classification Pipeline ---\n")


    print("[1/5] Loading Iris Benchmark Dataset...")
    df = pd.read_csv('iris.csv')

    label_candidates = ['species', 'Species', 'label', 'Label', 'target', 'class', 'Class']
    label_col = None
    for cand in label_candidates:
        if cand in df.columns:
            label_col = cand
            break
    if label_col is None:
        label_col = df.columns[-1]

    X = df.drop(label_col, axis=1)
    y = df[label_col]
    
    print("[2/5] Shuffling and splitting data (80% Train / 20% Test)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    print("[3/5] Applying StandardScaler (Mean=0, Variance=1)...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("[4/5] Instantiating and Fitting K-Nearest Neighbors (K=5)...")
    model = KNeighborsClassifier(n_neighbors=5)
    model.fit(X_train_scaled, y_train)

    print("[*] Predicting unseen data...")
    predictions = model.predict(X_test_scaled)
    
    print("[5/5] Output Validation Pipeline Complete. Generating Metrics...\n")
    conf_matrix = confusion_matrix(y_test, predictions)
    print("Confusion Matrix (TP, FP, FN, TN distribution):")
    print(conf_matrix)
    print("\n")

    print("Strategic Trade-offs Report (Precision, Recall, F1 Score):")
    labels = model.classes_
    report = classification_report(y_test, predictions, target_names=[str(c) for c in labels])
    print(report)

if __name__ == "__main__":
    main()