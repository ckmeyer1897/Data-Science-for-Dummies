import streamlit as st
import numpy as np
import pandas as pd
import pages
#from pages import utils
#from utils import load_df
#from utils import summary_target
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.figure import Figure
import random
import sklearn
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
#from pages.utils import summary_target
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, recall_score ,classification_report, roc_curve, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from scikitplot.metrics import plot_lift_curve
from scikitplot.metrics import plot_cumulative_gain
from sklearn import metrics
import time

def app():

    df = pd.read_csv('data/df.csv')
# Split into train test 
    st.markdown('# Train & Testing Sets')
    df = df.copy()
    X = df.drop('y', axis = 1)
    y = df['y']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    test_df = x_test.copy()
    train_df = x_train.copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('## Training Data')
        st.write('Features Dimensions: ', x_train.shape)
        st.write('Target Dimensions: ', y_train.shape)
    with col2:
        st.markdown('## Testing Data')
        st.write('Feature Dimensions: ', x_test.shape)
        st.write('Target Dimensions: ', y_test.shape)      

# Logistic Regression
## Functions
    def get_model_metrics(x_train, y_train, x_test, y_test, preds, mdl):
        train_acc = mdl.score(x_train,y_train)
        test_acc = mdl.score(x_test, y_test)
        rmse = (np.sqrt(mean_squared_error(y_test, preds)))
        
        results = {'Train_acc': train_acc, 'Test_acc': test_acc, 'rmse': rmse}
        #model = f'{mdl=}'.split('=')[1:]
        model = 'Value'
        model_metrics = pd.DataFrame(results.items(), columns = ['Metric', str(model)]).set_index('Metric')
        #model_metrics['Model'] = str(model)
        return model_metrics

    def classification_metrics(x_train, y_train, x_test, y_test, preds ,probs ,mdl):
        #Logistic Evaluation Metrics
        Accuracy = accuracy_score(y_test, preds)
        Precision  = precision_score(y_test, preds)
        Recall = recall_score(y_test, preds)

        #Confusion Matrix
        cm = pd.DataFrame(confusion_matrix(y_test, preds, labels=[0,1]))
        TN = cm[0][0]                                                                                          # True Positives
        FN = cm[0][1]                                                                                          # False Positives
        FP = cm[1][0]                                                                                        # True Negatives
        TP = cm[1][1]
        TPR = TN/(FP+TN)   
        FPR = FP/(FP+TN)

        # ROC Curve
        fpr, tpr, thresholds = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr )

        #Summary df
        logit_summary = {'Accuracy': Accuracy, 
                        'Precision': Precision, 
                        'Recall': Recall, 
                        'True Positive Rate': TPR, 
                        'False Positive Rate': FPR,
                        }
        
        #model  = f'{mdl=}'.split('=')[1:]
        model = 'Value'
        class_metrics = pd.DataFrame(logit_summary.items(), columns = ['Metric', str(model)]).set_index('Metric')
        #class_metrics['Model'] = str(model)
        # ROC Curve
        with col1:
            plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % roc_auc)
            plt.plot([0, 1], [0, 1], 'k--')  # random predictions curve
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.0])
            plt.xlabel('False Positive Rate or (1 - Specifity)')
            plt.ylabel('True Positive Rate or (Sensitivity)')
            plt.title('Receiver Operating Characteristic (ROC)')
            plt.legend(loc="lower right")
            st.pyplot()

        #Confusion Matrix
        with col2:
            fig, ax = plt.subplots(figsize=(4, 4))
            sklearn.metrics.plot_confusion_matrix(mdl, x_test, y_test, cmap=plt.cm.Blues, ax=ax)
            plt.tight_layout()
            plt.title('Confusion Matrix', y = 1.1)
            st.pyplot()

        
        return class_metrics
    st.empty()
## Code
    st.markdown('# Models')
    st.markdown('#### Select Model')

    st.empty()
    if st.button('Logistic Regression'):
        st.header('Logistic Regression')    
        logit = LogisticRegression(solver='liblinear', random_state=0)
        #Training 
        st.text('Training model. . .')
        training = st.progress(0)
        for perc_complete in range(100):
            time.sleep(0.01)
            training.progress(perc_complete + 1)
            logit.fit(x_train, y_train)
        st.success('Training Complete')

 ### #Predictions
        st.text('Making predictions on the test set. . .')
        predicting = st.progress(0)
        for perc_complete in range(100):
            time.sleep(0.01)
            predicting.progress(perc_complete + 1)
        logit_preds = logit.predict(x_test)
        logit_probs = logit.predict_proba(x_test)[:,1]
        st.success('Predictions Completed')
        col1, col2, col3 = st.columns(3)

        model_metrics = get_model_metrics(x_train, y_train, x_test, y_test, logit_preds, logit)
        class_metrics = classification_metrics(x_train, y_train, x_test, y_test, logit_preds,logit_probs, logit)
        
        metrics_list = [model_metrics, class_metrics]
        lgrmetrics = pd.concat(metrics_list)
        lgrmetrics = lgrmetrics.reset_index()
        lgrmetrics['Model'] = 'Logisitc'
        with col3:
            st.write('Metrics', lgrmetrics)
        
