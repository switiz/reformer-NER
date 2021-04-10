from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import os
def compute_metrics(labels, preds):
    assert len(preds) == len(labels)
    return f1_pre_rec(labels, preds)


def f1_pre_rec(labels, preds):
    return {
        "precision": precision_score(labels, preds, suffix=True),
        "recall": recall_score(labels, preds, suffix=True),
        "f1": f1_score(labels, preds, suffix=True)
    }

def show_report(labels, preds):
    return classification_report(labels, preds, suffix=True)

def get_test_texts(data_dir, test_file):
    texts = []
    with open(os.path.join(data_dir, test_file), 'r', encoding='utf-8') as f:
        for line in f:
            text, _ = line.split('\t')
            text = text.split()
            texts.append(text)

    return texts