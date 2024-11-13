import os
import sys
import pandas as pd
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve, matthews_corrcoef

import argparse
import numpy as np
import matplotlib.pyplot as plt
from models import Create_Train_Test

parser = argparse.ArgumentParser()
parser.add_argument('filename')
args = parser.parse_args()

df = pd.read_csv(args.filename)
X_train, X_test, y_train, y_test = Create_Train_Test(df)
file = args.filename.split('/')[-1].split('.')[0]

#### MIN_SAMPLES_SPLIT #####

train_scores = []
test_scores = []
min_splits = np.arange(2, 90)

for split in min_splits:
    model = tree.DecisionTreeClassifier(min_samples_split=split).fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure()
plt.plot(min_splits, train_scores, c='green', label='Train Score')
plt.plot(min_splits, test_scores, c='red', label='Test Score')
plt.grid()
plt.ylabel('Accuracy')
plt.xlabel('MIN_SAMPLES_SPLIT')
plt.legend()
plt.title('Train Test Scores en fonction de MIN_SAMPLES_SPLIT')
plt.savefig(f'images/hyperparameters/Train_Test_MIN_SAMPLES_SPLIT_{file}.png')
plt.show()


#### MIN_SAMPLES_LEAF #####

train_scores = []
test_scores = []
min_leaf = np.arange(1, 90)

for leaf in min_leaf:
    model = tree.DecisionTreeClassifier(min_samples_leaf=leaf).fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure()
plt.plot(min_leaf, train_scores, c='green', label='Train Score')
plt.plot(min_leaf, test_scores, c='red', label='Test Score')
plt.grid()
plt.ylabel('Accuracy')
plt.xlabel('MIN_SAMPLES_LEAF')
plt.legend()
plt.title('Train Test Scores en fonction de MIN_SAMPLES_LEAF')
plt.savefig(f'images/hyperparameters/Train_Test_MIN_SAMPLES_LEAF_{file}.png')
plt.show()

#### MAX_DEPTHS #####

train_scores = []
test_scores = []
max_depths = np.arange(1, 20)

for depth in max_depths:
    model = tree.DecisionTreeClassifier(max_depth=depth).fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure()
plt.plot(max_depths, train_scores, c='green', label='Train Score')
plt.plot(max_depths, test_scores, c='red', label='Test Score')
plt.grid()
plt.ylabel('Accuracy')
plt.xlabel('MAX_DEPTH')
plt.legend()
plt.title("Train Test Scores en fonction de MAX_DEPTH")
plt.savefig(f'images/hyperparameters/Train_Test_MAX_DEPTH_{file}.png')
plt.show()

#### CCP_ALPHA #####

train_scores = []
test_scores = []
ccp_alphas = np.arange(.00001, .01, .0001)

for ccp in ccp_alphas:
    model = tree.DecisionTreeClassifier(ccp_alpha=ccp).fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure()
plt.plot(ccp_alphas, train_scores, c='green', label='Train Score')
plt.plot(ccp_alphas, test_scores, c='red', label='Test Score')
plt.grid()
plt.ylabel('Accuracy')
plt.xlabel('CCP_ALPHA')
plt.legend()
plt.title("Train Test Scores en fonction de CCP_ALPHA")
plt.savefig(f'images/hyperparameters/Train_Test_CCP_ALPHA_{file}.png')
plt.show()



#### MAX_LEAF_NODES #####

train_scores = []
test_scores = []
max_leafs = np.arange(2, 100, 1)

for max_leaf in max_leafs:
    model = tree.DecisionTreeClassifier(max_leaf_nodes=max_leaf).fit(X_train, y_train)
    train_scores.append(model.score(X_train, y_train))
    test_scores.append(model.score(X_test, y_test))

plt.figure()
plt.plot(max_leafs, train_scores, c='green', label='Train Score')
plt.plot(max_leafs, test_scores, c='red', label='Test Score')
plt.grid()
plt.ylabel('Accuracy')
plt.xlabel('MAX_LEAF_NODES')
plt.legend()
plt.title("Train Test Scores en fonction de MAX_LEAF_NODES")
plt.savefig(f'images/hyperparameters/Train_Test_MAX_LEAF_NODES_{file}.png')
plt.show()
