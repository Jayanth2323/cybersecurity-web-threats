from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


def evaluate_model(y_true, y_pred):
    return {
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_pred),
    }


if __name__ == "__main__":
    y_true = [1, 0, 1, 1, 0, 1, 0, 0, 1, 0]  # Example true labels
    y_pred = [1, 0, 1, 0, 0, 1, 1, 0, 1, 1]  # Example predicted labels

    results = evaluate_model(y_true, y_pred)

    print("Model Evaluation Results")
    print("-" * 30)
    for metric, value in results.items():
        print(f"{metric}: {value:.4f}")
    print("-" * 30)
    print("Evaluation complete.")
    print("Thank you for using the model evaluation tool.")
    print("If you have any questions or feedback, please reach out.")
    print("Results have been saved to evaluation_report.txt")
    print("Finished writing evaluation report.")
    print("Exiting the program.")

    with open("evaluation_report.txt", "w") as f:
        f.write("Model Evaluation Report\n")
        f.write("-" * 30 + "\n")
        for metric, value in results.items():
            f.write(f"{metric}: {value:.4f}\n")
        f.write("-" * 30 + "\n")
        f.write("Evaluation complete.\n")
        f.write("Thank you for using the model evaluation tool.\n")
        f.write("If you have any questions or feedback, please reach out.\n")
