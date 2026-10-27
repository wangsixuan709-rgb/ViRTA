from sklearn.metrics import accuracy_score, precision_recall_fscore_support


def classification_metrics(labels, predictions):
    p, r, f, support = precision_recall_fscore_support(labels, predictions, labels=[0, 1], zero_division=0)
    result = {'accuracy': float(accuracy_score(labels, predictions)), 'macro_f1': float(f.mean())}
    for i, name in enumerate(('fake', 'real')):
        result.update({f'{name}_precision': float(p[i]), f'{name}_recall': float(r[i]), f'{name}_f1': float(f[i])})
    return result


def metric_table(metrics):
    keys = ['accuracy','fake_precision','fake_recall','fake_f1','real_precision','real_recall','real_f1']
    return 'Accuracy | Fake Precision | Fake Recall | Fake F1-score | Real Precision | Real Recall | Real F1-score\n' + ' | '.join(f'{100*metrics[k]:.2f}' for k in keys)
