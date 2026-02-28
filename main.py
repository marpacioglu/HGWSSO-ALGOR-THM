# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:00:13 2024

@author: mustafa
"""
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from FS.hibrit_gwossa_yeni import jfs   # change this to switch algorithm 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay,precision_score, recall_score, f1_score


# Define parameters
num_runs = 10  # Number of times to repeat the process
k = 5          # k-value in KNN
N = 30         # Number of particles
T = 1000    # Maximum number of iterations

# Prepare lists to store results
all_curves = np.zeros((T, num_runs))
feature_sizes = []
accuracies = []
precisions = []   # Precision sonuçları için liste
recalls = []      # Recall sonuçları için liste
f1_scores = []  
conf_matrices = []

for run in range(num_runs):
    print(f"Run {run + 1}/{num_runs}")

    # Load data
    data = pd.read_csv('normalize.csv').values
    feat = np.asarray(data[:, :-1])
    label = np.asarray(data[:, -1])

    # Split data into train & validation (70% - 30%)
    xtrain, xtest, ytrain, ytest = train_test_split(feat, label, test_size=0.30, stratify=label)
    fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

    # Parameters for feature selection
    opts = {'k': k, 'fold': fold, 'N': N, 'T': T}

    # Perform feature selection
    fmdl = jfs(feat, label, opts)  # Ensure this function is correctly defined
    sf = fmdl['sf']

    # Model with selected features
    x_train = xtrain[:, sf]
    y_train = ytrain
    x_valid = xtest[:, sf]
    y_valid = ytest

    # Train KNN model
    mdl = KNeighborsClassifier(n_neighbors=k)
    mdl.fit(x_train, y_train)

    # Accuracy
    y_pred = mdl.predict(x_valid)
    accuracy = np.sum(y_valid == y_pred) / len(y_valid)
    accuracies.append(accuracy)
    
    # Precision, Recall, F1 Score hesaplama
    precision = precision_score(y_valid, y_pred, average='weighted')
    recall = recall_score(y_valid, y_pred, average='weighted')
    f1 = f1_score(y_valid, y_pred, average='weighted')

    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    
    # Number of selected features
    num_feat = fmdl['nf']
    feature_sizes.append(num_feat)

    # Save curve values for this run
    curve = fmdl['c'].reshape(-1)
    all_curves[:, run] = curve

    # Save the feature sizes and accuracy for this run
    run_summary_df = pd.DataFrame({
        'Run': [run + 1],
        'Feature Size': [num_feat],
        'Accuracy': [100 * accuracy],
        'Precision': [100 * precision],
        'Recall': [100 * recall],
        'F1 Score': [100 * f1]
    })
    run_summary_df.to_csv(f'run_{run + 1}_summary.csv', index=False)
    
    cm = confusion_matrix(y_valid, y_pred)
    conf_matrices.append(cm)  # Her çalışmanın sonucunu sakla

    # Karışıklık matrisini çizdir
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Confusion Matrix - Run {run + 1}-gwo_emodb_knn')
    plt.show()
    

# Create DataFrame with iterations as rows and runs as columns
iteration_range = np.arange(1, T + 1)
all_curves_df = pd.DataFrame(all_curves, index=iteration_range)
all_curves_df.columns = [f'Run_{i + 1}' for i in range(num_runs)]

# Save all curves to CSV
all_curves_df.to_csv('all_curves_values_gwo_emodb_knn.csv', index=True)

# Save feature sizes and accuracies to CSV
feature_results_df = pd.DataFrame({
    'Run': np.arange(1, num_runs + 1),
    'Feature Size': feature_sizes,
    'Accuracy': [100 * acc for acc in accuracies],
    'Precision': [100 * p for p in precisions],
    'Recall': [100 * r for r in recalls],
    'F1 Score': [100 * f for f in f1_scores]
})
feature_results_df.to_csv('feature_sizes_and_accuracies_gwo_emodb_knn.csv', index=False)

# Example of how to plot a convergence curve for one of the runs, e.g., the first run
plt.figure()
for run in range(num_runs):
    plt.plot(iteration_range, all_curves[:, run], label=f'Run {run + 1}')

plt.xlabel('Number of Iterations')
plt.ylabel('Fitness')
plt.title('Feature Selection Convergence Across Runs-gwo_emodb_knn')
plt.legend()
plt.grid()
plt.show()

plt.figure()
for run in range(num_runs):
    plt.plot(iteration_range, 1 - all_curves[:, run], label=f'Run {run + 1}')

plt.xlabel('Number of Iterations')
plt.ylabel('Accuracy')
plt.title('Feature Selection Accuracy Across Runs-gwo_emodb_knn')
plt.legend()
plt.grid(True)
plt.show()



