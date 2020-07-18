# ML_COURSE_DOAO

    This class is an implementation of :
    Kang, Seokho, Sungzoon Cho, and Pilsung Kang. "Constructing a multi-class classifier
    using one-against-one approach with different binary classifiers." Neurocomputing 149 (2015): 677-682.

    -The one-against-one binary classifiers and their hyperparameters are identical
    the ones mentioned in the paper.

    - The cross-validation error is the same validation error mentioned in the paper ( see binary_mean_error_score ).


    Using the model:

    Using this class is the same as using any sklearn-classification algorithm. First the 'fit' method
    needs to be ran on the training-data, and only then the predict method can be called with some test-data
    and yields the predicted labels. A few notes:

        * Labels in the training-set are assumed to be label-encoded. Meaning, if there are K classes,
        then the set(y) should be {0,1,2....K-1}.

        * With small and imbalanced datasets, choosing some cv_folds parameters will cause sklearn to raise
        an exception as there aren't enough training samples for the given amount of folds (or classes in each fold).
        This isn't handled as to prevent unexpected behavior within the algorithm. In such a case, choose different
        cv_folds arg or add more training data.

