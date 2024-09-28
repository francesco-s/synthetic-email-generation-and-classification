from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


# Model evaluation function after training
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy}")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["spam", "promotion", "work", "personal"]))

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))