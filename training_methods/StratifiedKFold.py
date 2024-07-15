def train_and_evaluate(model, X, y, X_test, folds=5, random_state=None):
    """
       basic modeling
    """
    print(f'Training {model.__class__.__name__}\n')

    scores = []
    feature_importances = np.zeros(X.shape[1])
    evaluation_history = []

    oof_pred_probs = np.zeros(X.shape[0])
    test_pred_probs = np.zeros(X_test.shape[0])

    skf = StratifiedKFold(n_splits=FOLDS, random_state=94, shuffle=True)

    for fold_index, (train_index, val_index) in enumerate(skf.split(X, y)):
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y.iloc[train_index], y.iloc[val_index]

        model_clone = copy.deepcopy(model)
        model_clone.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=LOG_STEPS)

        feature_importances += model_clone.feature_importances_ / folds
        evaluation_history.append(model_clone.evals_result())

        y_pred_probs = model_clone.predict_proba(X_val)[:, 1]
        oof_pred_probs[val_index] = y_pred_probs

        temp_test_pred_probs = model_clone.predict_proba(X_test)[:, 1]
        test_pred_probs += temp_test_pred_probs / folds

        auc_score = roc_auc_score(y_val, y_pred_probs)
        scores.append(auc_score)

        print(f'\n--- Fold {fold_index + 1} - AUC: {auc_score:.5f}\n\n')

        del model_clone
        gc.collect()

    print(f'------ Average AUC: {np.mean(scores):.5f} ± {np.std(scores):.5f}\n\n')

    return oof_pred_probs, test_pred_probs
