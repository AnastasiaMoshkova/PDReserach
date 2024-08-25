from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix
import sklearn.metrics
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import numpy as np


def confusion_matrix_plot(df, datasets, output_dir, name):
    if not os.path.isdir(os.path.join(output_dir, 'confusion_matrix')):
        os.mkdir(os.path.join(output_dir, 'confusion_matrix'))
    df = df[df['dataset'].isin(datasets)]
    _confusion_matrix_plot_by_df(df,  os.path.join(output_dir, 'confusion_matrix'), name + '_by_records')
    df = round(df.groupby('id')[['class', 'pred']].mean())
    _confusion_matrix_plot_by_df(df, os.path.join(output_dir, 'confusion_matrix'), name + '_by_patients')

def _confusion_matrix_plot_by_df(df, output_dir, name):
    y_true, y_pred = df['class'], round(df['pred'])
    labels = sorted(np.array(df['class'].unique()))
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.savefig(os.path.join(output_dir, name + '.jpg'))


class Metrics():
    def __init__(self, **config):
        self.config = config

    def get_metrics(self, test):
        result = {}
        result.update({'accuracy': round(accuracy_score(test['class'], test['pred']) * 100, 2)})
        #for average in self.config['f1_type']:
            #result.update({'f1_' + average: round(f1_score(test['class'], test['pred'], average=average), 2)})
        clf_report = classification_report(test['class'], test['pred'], output_dict=True) #labels=self.config['target_names'])
        print(clf_report.keys())
        for key in clf_report.keys():
            print(clf_report[key])
        test.loc[test['pred'] > 0, 'pred'] = 1
        test.loc[test['class'] > 0, 'class'] = 1
        clf_report = classification_report(test['class'], test['pred'], output_dict=True) #labels=self.config['target_names'])
        print(clf_report.keys())
        for key in clf_report.keys():
            print(clf_report[key])
        return result
