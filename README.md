# DOAO

    DOAO.py is an implementation of :
    Kang, Seokho, Sungzoon Cho, and Pilsung Kang. "Constructing a multi-class classifier
    using one-against-one approach with different binary classifiers." Neurocomputing 149 (2015): 677-682.

    -The one-against-one binary classifiers and their hyperparameters are mostly identical
    the ones mentioned in the paper.

    - The cross-validation error is the same validation error mentioned in the paper ( see binary_mean_error_score ).


    Using the model:

    Using this class is the same as using any sklearn-classification algorithm. First the 'fit' method
    needs to be ran on the training-data, and only then the predict method can be called with some test-data
    which yields the predicted labels. A few notes:

        * Labels in the training-set are assumed to be label-encoded. Meaning, if there are K classes,
        then the set(y) should be {0,1,2....K-1}.

        * With small and imbalanced datasets, choosing some cv_folds parameters will cause sklearn to raise
        an exception as there aren't enough training samples for the given amount of folds (or classes in each fold).
        This isn't handled as to prevent unexpected behavior within the algorithm. In such a case, choose different
        cv_folds arg or add more training data.



    Experiments:
    The experiments folder contains a dataset file that includes metrics on 150 classification datasets. The experiments were conducted using a 10-fold cross validation in
    which parameters were tuned on the 90% training data using an internal 3-fold cross validation. 
    DOAO Parameters are already tuned inside the model's fit method. The parameters grid used for AdaBoost is (with a decision tree base classifier):

    param_grid = {"base_estimator__criterion": ["gini", "entropy"],
                  "base_estimator__max_depth": [None, 1, 2, 3],
                  "base_estimator__min_samples_split": [2, 4, 6, 0.05],
                  "n_estimators": [i for i in range(45, 51)] + [2, 3, 4, 5],
                  "learning_rate": [2 ** i for i in range(-4, 1)]
              }
