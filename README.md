# 46
import argparse
import logging
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
def compute_ci95(scores):
    """Calculates 95% Confidence Interval for a list/array of scores."""
    mean = np.mean(scores)
    std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
    ci = 1.96 * (std / math.sqrt(len(scores)))
    return mean, ci

def get_classical_models():
    """Returns dictionary of baseline classical ML models."""
    return {
        "Multinomial_NB": MultinomialNB(),
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SGD_Classifier": SGDClassifier(loss="log_loss", random_state=42),
        "Decision_Tree": DecisionTreeClassifier(random_state=42),
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "MLP_Classifier": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=200, random_state=42)
    }

def run_cross_validation(X_text, y, cv_folds=5):
    """Executes 5-Fold CV on text data using TF-IDF vectorization and classical classifiers."""
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    models = get_classical_models()
    results = []

    for name, model in models.items():
        logging.info(f"Running 5-Fold CV for model: {name}")
        accs, precs, recs, f1s, aucs = [], [], [], [], []

        for fold, (train_idx, val_idx) in enumerate(skf.split(X_text, y)):
            X_train, X_val = X_text[train_idx], X_text[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Vectorization per fold to prevent data leakage
            vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
            X_train_vec = vectorizer.fit_transform(X_train)
            X_val_vec = vectorizer.transform(X_val)

            model.fit(X_train_vec, y_train)
            preds = model.predict(X_val_vec)

            # Probabilities for AUC
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val_vec)[:, 1]
            elif hasattr(model, "decision_function"):
                probs = model.decision_function(X_val_vec)
            else:
                probs = preds

            acc = accuracy_score(y_val, preds)
            p, r, f, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
            auc = roc_auc_score(y_val, probs)

            accs.append(acc)
            precs.append(p)
            recs.append(r)
            f1s.append(f)
            aucs.append(auc)

        acc_m, acc_ci = compute_ci95(accs)
        p_m, p_ci = compute_ci95(precs)
        r_m, r_ci = compute_ci95(recs)
        f1_m, f1_ci = compute_ci95(f1s)
        auc_m, auc_ci = compute_ci95(aucs)

        results.append({
            "Model": name,
            "Accuracy": f"{acc_m:.4f} ± {acc_ci:.4f}",
            "Precision": f"{p_m:.4f} ± {p_ci:.4f}",
            "Recall": f"{r_m:.4f} ± {r_ci:.4f}",
            "F1_Score": f"{f1_m:.4f} ± {f1_ci:.4f}",
            "ROC_AUC": f"{auc_m:.4f} ± {auc_ci:.4f}",
            "F1_Raw_Mean": f1_m
        })

    return pd.DataFrame(results)

def plot_and_save_results(df_results, output_plot_path="experiment_summary.png"):
    """Generates and saves performance comparison chart."""
    df_results_sorted = df_results.sort_values(by="F1_Raw_Mean", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_results_sorted, x="F1_Raw_Mean", y="Model", palette="viridis")
    plt.title("Phishing Detection Model Performance (Mean F1-Score)")
    plt.xlabel("Mean F1-Score (5-Fold CV)")
    plt.ylabel("Model Architecture")
    plt.xlim(0, 1.0)
    plt.tight_layout()
    plt.savefig(output_plot_path, dpi=300)
    logging.info(f"Summary plot saved to {output_plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Full Phishing Detection Experiment Pipeline")
    parser.add_argument("--data_path", type=str, required=True, help="Path to raw CSV dataset containing 'text' and 'label' columns")
    parser.add_argument("--out_csv", type=str, default="experiment_results.csv", help="Output path for CSV results")
    args = parser.parse_args()

    logging.info(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)

    # Basic data validation
    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("Dataset must contain 'text' and 'label' columns.")

    X_text = df["text"].astype(str).values
    y = df["label"].values

    # Run CV Evaluation
    df_results = run_cross_validation(X_text, y, cv_folds=5)
    
    # Export results and visualization
    df_results.to_csv(args.out_csv, index=False)
    logging.info(f"Experiment results saved to {args.out_csv}")
    
    plot_and_save_results(df_results)

if __name__ == "__main__":
    main()
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Classical Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Sequence Model baseline (HMM)
from hmmlearn import hmm

def run_cpu_batch_experiments(data_filepath, output_csv="batch_cpu_results.csv"):
    """Runs batch cross-validation for CPU-friendly classical and sequence models."""
    df = pd.read_csv(data_filepath)
    X = df["text"].astype(str).values
    y = df["label"].values

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    classical_models = {
        "MNB": MultinomialNB(),
        "LogisticRegression": LogisticRegression(max_iter=1000, n_jobs=-1, random_state=42),
        "SGD": SGDClassifier(loss="log_loss", random_state=42),
        "DecisionTree": DecisionTreeClassifier(random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "MLP": MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
    }

    summary_rows = []

    # 1. Classical ML Batch
    for name, model in classical_models.items():
        print(f"Evaluating {name}...")
        f1_list, acc_list, auc_list = [], [], []

        for train_idx, val_idx in skf.split(X, y):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            vec = TfidfVectorizer(max_features=3000, stop_words="english")
            X_train_vec = vec.fit_transform(X_train)
            X_val_vec = vec.transform(X_val)

            model.fit(X_train_vec, y_train)
            preds = model.predict(X_val_vec)
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val_vec)[:, 1]
            else:
                probs = preds

            acc = accuracy_score(y_val, preds)
            _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
            auc = roc_auc_score(y_val, probs)

            acc_list.append(acc)
            f1_list.append(f1)
            auc_list.append(auc)

        summary_rows.append({
            "Model_Type": "Classical",
            "Model": name,
            "Accuracy_Mean": np.mean(acc_list),
            "F1_Mean": np.mean(f1_list),
            "ROC_AUC_Mean": np.mean(auc_list)
        })

    # 2. Sequence Model (HMM Baseline)
    print("Evaluating Sequence Model (HMM)...")
    hmm_f1s, hmm_accs = [], []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Use integer-encoded token counts for HMM sequence modeling
        count_vec = CountVectorizer(max_features=100, stop_words="english", binary=True)
        X_train_counts = count_vec.fit_transform(X_train).toarray()
        X_val_counts = count_vec.transform(X_val).toarray()

        # Fit separate Multinomial HMMs for Class 0 (Legitimate) and Class 1 (Phishing)
        hmm_0 = hmm.MultinomialHMM(n_components=3, random_state=42, n_iter=20)
        hmm_1 = hmm.MultinomialHMM(n_components=3, random_state=42, n_iter=20)

        hmm_0.fit(X_train_counts[y_train == 0])
        hmm_1.fit(X_train_counts[y_train == 1])

        # Predict based on log-likelihood ratio
        preds = []
        for sample in X_val_counts:
            sample_seq = sample.reshape(1, -1)
            score_0 = hmm_0.score(sample_seq)
            score_1 = hmm_1.score(sample_seq)
            preds.append(1 if score_1 > score_0 else 0)

        acc = accuracy_score(y_val, preds)
        _, _, f1, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
        
        hmm_accs.append(acc)
        hmm_f1s.append(f1)

    summary_rows.append({
        "Model_Type": "Sequence",
        "Model": "Multinomial_HMM",
        "Accuracy_Mean": np.mean(hmm_accs),
        "F1_Mean": np.mean(hmm_f1s),
        "ROC_AUC_Mean": N/A
    })

    # Export CPU Results Spreadsheet
    results_df = pd.DataFrame(summary_rows)
    results_df.to_csv(output_csv, index=False)
    print(f"\nBatch CPU Execution Complete. Exported to '{output_csv}'.")
    return results_df

# Example execution call:
# run_cpu_batch_experiments("phishing_dataset.csv")
python run_phishing_experiments.py --data_path path/to/your_dataset.csv --out_csv experiment_results.csv
# 1. Install required dependency for HMM
!pip install hmmlearn scikit-learn pandas numpy matplotlib seaborn -q

import os
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Classical Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

# Sequence Model
from hmmlearn import hmm

# ---------------------------------------------------------
# Step 1: Create Dummy Dataset (Skip if you have a CSV)
# ---------------------------------------------------------
DATA_FILE = "phishing_dataset.csv"

if not os.path.exists(DATA_FILE):
    print(f"Creating sample dataset '{DATA_FILE}' for demonstration...")
    sample_data = {
        "text": [
            "Urgent: Your account is suspended, click here to verify immediately",
            "Hey, are we still meeting for lunch today?",
            "Dear customer, update your bank details now to avoid termination",
            "Project update meeting scheduled for tomorrow at 10 AM",
            "Security alert: Unusual login detected on your paypal account",
            "Attached is the weekly status report for the team",
            "Congratulations! You won a gift card, claim your prize here",
            "Please review the attached document and send feedback by EOD"
        ] * 25,  # Repeat to create a larger sample dataset
        "label": [1, 0, 1, 0, 1, 0, 1, 0] * 25
    }
    pd.DataFrame(sample_data).to_csv(DATA_FILE, index=False)

# ---------------------------------------------------------
# Step 2: Run Full Model Suite & Generate Results
# ---------------------------------------------------------
def compute_ci95(scores):
    mean = np.mean(scores)
    std = np.std(scores, ddof=1) if len(scores) > 1 else 0.0
    ci = 1.96 * (std / math.sqrt(len(scores)))
    return mean, ci

print("Loading dataset...")
df = pd.read_csv(DATA_FILE)
X = df["text"].astype(str).values
y = df["label"].values

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

classical_models = {
    "Multinomial_NB": MultinomialNB(),
    "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42),
    "SGD_Classifier": SGDClassifier(loss="log_loss", random_state=42),
    "Decision_Tree": DecisionTreeClassifier(random_state=42),
    "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "MLP_Classifier": MLPClassifier(hidden_layer_sizes=(64,), max_iter=200, random_state=42)
}

results = []

# --- Evaluate Classical Models ---
for name, model in classical_models.items():
    print(f"Evaluating {name}...")
    accs, precs, recs, f1s, aucs = [], [], [], [], []

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        vec = TfidfVectorizer(max_features=3000, stop_words="english")
        X_train_vec = vec.fit_transform(X_train)
        X_val_vec = vec.transform(X_val)

        model.fit(X_train_vec, y_train)
        preds = model.predict(X_val_vec)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X_val_vec)[:, 1]
        elif hasattr(model, "decision_function"):
            probs = model.decision_function(X_val_vec)
        else:
            probs = preds

        accs.append(accuracy_score(y_val, preds))
        p, r, f, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
        precs.append(p)
        recs.append(r)
        f1s.append(f)
        aucs.append(roc_auc_score(y_val, probs))

    acc_m, acc_ci = compute_ci95(accs)
    p_m, p_ci = compute_ci95(precs)
    r_m, r_ci = compute_ci95(recs)
    f1_m, f1_ci = compute_ci95(f1s)
    auc_m, auc_ci = compute_ci95(aucs)

    results.append({
        "Model_Type": "Classical",
        "Model": name,
        "Accuracy": f"{acc_m:.4f} ± {acc_ci:.4f}",
        "Precision": f"{p_m:.4f} ± {p_ci:.4f}",
        "Recall": f"{r_m:.4f} ± {r_ci:.4f}",
        "F1_Score": f"{f1_m:.4f} ± {f1_ci:.4f}",
        "ROC_AUC": f"{auc_m:.4f} ± {auc_ci:.4f}",
        "F1_Mean": f1_m
    })

# --- Evaluate Sequence Model (Multinomial HMM) ---
print("Evaluating Sequence Model (Multinomial HMM)...")
hmm_accs, hmm_precs, hmm_recs, hmm_f1s = [], [], [], []

for train_idx, val_idx in skf.split(X, y):
    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]

    count_vec = CountVectorizer(max_features=50, stop_words="english", binary=True)
    X_train_counts = count_vec.fit_transform(X_train).toarray()
    X_val_counts = count_vec.transform(X_val).toarray()

    hmm_0 = hmm.MultinomialHMM(n_components=2, random_state=42, n_iter=15)
    hmm_1 = hmm.MultinomialHMM(n_components=2, random_state=42, n_iter=15)

    hmm_0.fit(X_train_counts[y_train == 0])
    hmm_1.fit(X_train_counts[y_train == 1])

    preds = []
    for sample in X_val_counts:
        sample_seq = sample.reshape(1, -1)
        score_0 = hmm_0.score(sample_seq)
        score_1 = hmm_1.score(sample_seq)
        preds.append(1 if score_1 > score_0 else 0)

    accs.append(accuracy_score(y_val, preds))
    p, r, f, _ = precision_recall_fscore_support(y_val, preds, average="binary", zero_division=0)
    hmm_precs.append(p)
    hmm_recs.append(r)
    hmm_f1s.append(f)

acc_m, acc_ci = compute_ci95(hmm_accs) if len(hmm_accs) > 0 else (0, 0)
p_m, p_ci = compute_ci95(hmm_precs)
r_m, r_ci = compute_ci95(hmm_recs)
f1_m, f1_ci = compute_ci95(hmm_f1s)

results.append({
    "Model_Type": "Sequence",
    "Model": "Multinomial_HMM",
    "Accuracy": f"{acc_m:.4f} ± {acc_ci:.4f}",
    "Precision": f"{p_m:.4f} ± {p_ci:.4f}",
    "Recall": f"{r_m:.4f} ± {r_ci:.4f}",
    "F1_Score": f"{f1_m:.4f} ± {f1_ci:.4f}",
    "ROC_AUC": "N/A",
    "F1_Mean": f1_m
})

# ---------------------------------------------------------
# Step 3: Save CSV and Plot Summary
# ---------------------------------------------------------
results_df = pd.DataFrame(results)
output_csv = "experiment_results.csv"
results_df.to_csv(output_csv, index=False)
print(f"\nResults successfully saved to '{output_csv}'!")

# Display DataFrame in Colab
display(results_df[["Model_Type", "Model", "Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]])

# Generate Summary Plot
plt.figure(figsize=(9, 5))
sns.barplot(data=results_df.sort_values(by="F1_Mean", ascending=False), x="F1_Mean", y="Model", palette="magma")
plt.title("Phishing Detection Baseline Comparison (F1-Score)")
plt.xlabel("Mean F1-Score")
plt.ylabel("Model")
plt.xlim(0, 1.0)
plt.tight_layout()
plt.savefig("experiment_summary.png", dpi=300)
plt.show()

# ---------------------------------------------------------
# Step 4: Download Results CSV to Local Machine
# ---------------------------------------------------------
files.download(output_csv)
files.download("experiment_summary.png")
DATA_FILE = "your_actual_file_name.csv"  # Replace with your file name or copied path
