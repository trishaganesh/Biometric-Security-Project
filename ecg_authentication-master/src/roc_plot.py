import glob
import os
import csv
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
import warnings
warnings.filterwarnings("ignore")

# load libraries

which_db = 'mitdb'


def plot_all_roc(all_file_name):
    plt.figure()
    temp = []
    # Create feature matrix and target vector
    for file_name in all_file_name:
        X, y = make_classification(
            n_samples=10000, n_features=100, n_classes=2)

        X = []
        y = []
        y.append(1)
        X.append(1)
        with open(file_name, newline='') as csvfile:
            reader = csv.reader(csvfile)
            for row in reader:
                y.append(float(row[1]) / 100)
                X.append(float(row[2]) / 100)
        y.append(0)
        X.append(0)
        # X = X/100
        # y = y/100
        lw = 2
        legend_list = file_name.split('_')
        if "non-fiducial" in legend_list:
            # legend = legend_list[4] + ' ' + legend_list[5] + ' ' + legend_list[6] + ' ' + legend_list[7]
            legend = legend_list[3] + ' ' + legend_list[4] + \
                ' ' + legend_list[5] + ' ' + legend_list[6]

        else:
            # legend = legend_list[4] + ' ' + legend_list[5] + ' ' + legend_list[6]
            legend = legend_list[3] + ' ' + \
                legend_list[4] + ' ' + legend_list[5]

        # legend = legend_list[3] + ' ' + legend_list[4] + ' ' + legend_list[5] + ' ' + legend_list[6]
        if legend not in temp:
            temp.append(legend)
        else:
            continue
        plt.plot(X, y,
                 lw=lw, label=legend)
        plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic')

        plt.legend(loc="lower right")
    save_file = file_name.replace('.csv', '.eps')
    plt.savefig(
        '/Users/mohitingale/Workspace/RA/ecg_authentication/src/Result/ROC_Plot/new/' +
        which_db +
        '.eps',
        format='eps')


def plot_roc(file_name):
    print()
    print(format('How to plot a ROC Curve in Python', '*^82'))

    # Create feature matrix and target vector
    X, y = make_classification(n_samples=10000, n_features=100, n_classes=2)

    X = []
    y = []

    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            y.append(float(row[1]) / 100)
            X.append(float(row[2]) / 100)

    # X = X/100
    # y = y/100

    plt.figure()
    lw = 2
    plt.plot(X, y, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)')
    plt.plot([0, 1], [1, 0], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    save_file = file_name.replace('.csv', '.eps')
    plt.savefig('ROC_Plot/' + save_file, format='eps')
    # plt.show()


def plot_err(file_name):
    X = []
    y = []

    with open(file_name, newline='') as csvfile:
        reader = csv.reader(csvfile)
        for row in reader:
            y.append(1 - float(row[1]) / 100)
            X.append((float(row[2]) / 100))
    plt.figure()
    import numpy as np
    np_X = np.array(X)
    np_y = np.array(y)
    plt.plot(np_X, label='FRR')
    plt.plot(np_y, label='FAR')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    save_file = file_name.replace('.csv', '.eps')
    plt.savefig('ERR_Plot/' + save_file, format='eps')


path = '/Users/mohitingale/Workspace/RA/ecg_authentication/src/Result/' + which_db
extension = 'csv'
os.chdir(path)
result = glob.glob('*.{}'.format(extension))
plot_all_roc(result)
# for each_result in result:
#     plot_roc(each_result)
#     plot_err(each_result)
