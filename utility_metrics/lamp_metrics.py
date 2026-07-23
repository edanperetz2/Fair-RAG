# Adapted from https://github.com/LaMP-Benchmark/LaMP/blob/main/eval/evaluation.py
from rouge import Rouge
from typing import Dict, List


LAMP_CLASSIFICATION_LABELS: Dict[int, List[str]] = {
    1: ["[1]", "[2]"],
    2: [
        "sci-fi", "based on a book", "comedy", "action", "twist ending",
        "dystopia", "dark comedy", "classic", "psychology", "fantasy",
        "romance", "thought-provoking", "social commentary", "violence", "true story",
    ],
    3: ["1", "2", "3", "4", "5"],
}


def _postprocess_text_classification(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]
    return preds, labels


def _postprocess_text_generation(preds, labels):
    preds = [str(pred).strip() for pred in preds]
    labels = [str(label).strip() for label in labels]

    return preds, labels


def get_metric_fn_accuracy(all_labels):
    def create_mapping(x):
        try:
            return all_labels.index(x)
        except:
            return -1

    def metric_fn(decoded_preds, decoded_labels):
        decoded_preds, decoded_labels = _postprocess_text_classification(
            decoded_preds, decoded_labels
        )
        decoded_preds = [create_mapping(x) for x in decoded_preds]
        decoded_labels = [create_mapping(x) for x in decoded_labels]
        accuracy_scores = [
            (1 if pred == ref else 0)
            for pred, ref in zip(decoded_preds, decoded_labels)
        ]
        return accuracy_scores

    return metric_fn


def get_metric_fn_mae():
    def create_mapping(x, y):
        try:
            return float(x)
        except:
            print(x)
            y = float(y)
            if abs(1 - y) > abs(5 - y):
                return 1.0
            else:
                return 5.0

    def metric_fn(decoded_preds, decoded_labels) -> list:
        decoded_preds, decoded_labels = _postprocess_text_classification(
            decoded_preds, decoded_labels
        )
        decoded_preds = [
            create_mapping(x, y) for x, y in zip(decoded_preds, decoded_labels)
        ]
        decoded_labels = [create_mapping(x, x) for x in decoded_labels]
        mae_scores = [
            abs(pred - ref) for pred, ref in zip(decoded_preds, decoded_labels)
        ]
        return mae_scores

    return metric_fn


def get_metric_fn_rouge_L():
    def metric_fn(predictions, references) -> list:
        rouge = Rouge()
        predictions, references = _postprocess_text_generation(predictions, references)
        scores = rouge.get_scores(predictions, references, avg=False)
        # Extracting ROUGE-L F1 scores for each prediction-reference pair
        rouge_l_scores = [score["rouge-l"]["f"] for score in scores]
        return rouge_l_scores

    return metric_fn
