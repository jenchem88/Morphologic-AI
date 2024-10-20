#uses the morphological features extracted by the 
# previous algorithm to train a model that can distinguish between infected and 
#uninfected cells.

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

def load_data(features_file, labels_file):
    features = pd.read_csv(features_file)
    labels = pd.read_csv(labels_file)
    return features, labels['infected']

def train_classifier(features, labels):
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    
    return clf

def save_model(clf, filename):
    joblib.dump(clf, filename)

if __name__ == "__main__":
    features, labels = load_data('morphological_features.csv', 'infection_labels.csv')
    clf = train_classifier(features, labels)
    save_model(clf, 'infection_classifier.joblib')
