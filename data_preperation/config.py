from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV


def generate_params(lgbm=True, lgbm_learning_rate=[0.075, 0.1, 0.125],
                    lgbm_max_depth=[10, 11, 12], lgbm_n_estimators=[2500, 3000, 3750],
                    lgbm_min_child_samples=[21, 42, 63], lgbm_num_leaves=[12, 14, 16]):
    if lgbm:
        lgbm_params = {"learning_rate": lgbm_learning_rate,
                       "max_depth": lgbm_max_depth,
                       "n_estimators": lgbm_n_estimators,
                       "min_child_samples": lgbm_min_child_samples,
                       "num_leaves": lgbm_num_leaves}

    regressors = [("LightGBM", LGBMClassifier(random_state=17), lgbm_params)]

    return regressors


def hyperparameter_optimization(X, y, cv=3, lgbm=True, lgbm_learning_rate=[0.075, 0.1, 0.125],
                                lgbm_max_depth=[10, 11, 12], lgbm_n_estimators=[2500, 3000, 3750],
                                lgbm_min_child_samples=[21, 42, 63], lgbm_num_leaves=[12, 14, 16]):

    regressors = generate_params(lgbm, lgbm_learning_rate, lgbm_max_depth, lgbm_n_estimators,
                                 lgbm_min_child_samples, lgbm_num_leaves)

    print("Hyperparameter Optimization...")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
    for name, classifier, params in regressors:
        print(f"########## {name} ##########")

        final_model_before = classifier.fit(X_train, y_train)
        cv_results = cross_validate(final_model_before, X_train, y_train, cv=cv, scoring=["accuracy"])
        score_train_b = cv_results['test_accuracy'].mean()

        y_pred = classifier.predict(X_test)
        score_test_b = accuracy_score(y_pred, y_test)
        print(f"Before Optimization Accuracy:\n"
              f"Train Score -->  {round(score_train_b, 4)} ({name}) \n"
              f"Test Score  -->  {round(score_test_b, 4)} ({name}) ")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_).fit(X_train, y_train)

        cv_results = cross_validate(final_model, X_train, y_train, cv=cv, scoring=["accuracy"])
        score_train_a = cv_results['test_accuracy'].mean()

        y_pred = final_model.predict(X_test)
        score_test_a = accuracy_score(y_pred, y_test)
        print(f"After Optimization Accuracy:\n"
              f"Train Score -->  {round(score_train_a, 4)} ({name}) \n"
              f"Test Score  -->  {round(score_test_a, 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    return final_model
