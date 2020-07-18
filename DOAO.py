import numpy as np
from itertools import combinations
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.utils import shuffle
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.utils.testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


def binary_mean_error_score(y_true, y_predicted):
    assert len(y_true) != 0
    return sum([i != j for i, j in zip(y_true, y_predicted)]) / len(y_true)


def setup_class_dict(x_train, labels):
    """
    :param x_train: x samples.
    :param labels: corresponding y labels.
    :return: A dictionary that contains a mapping for each class k in labels. The dictionary maps
    each class k to all x-samples whose label is k.
    """
    class_dict = {}
    for label in set(labels):
        class_dict[label] = []

    for index, label in enumerate(labels):
        class_dict[label] += [x_train[index]]
    return class_dict


def join_two_classes(i, j, class_dict):
    """
    :param i: class i
    :param j: class j
    :param class_dict: A dictionary that contains a mapping for every class k, to a matrix of all
    training samples that are labeled as k (see setup_class_dict above).
    :return: A unified matrix containing all training samples that are labeled either i or j, and
    their matching labels.
    """
    x_i = class_dict[i]
    x_j = class_dict[j]
    x_i_j = np.concatenate((x_i, x_j))
    # first in pair is marked as 0, second is marked as 1
    y_i_j = np.array([0] * len(x_i) + [1] * len(x_j))
    return x_i_j, y_i_j


class DiverseOAO:
    """"
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

    """
    param_grids = {
        "knn": {"n_neighbors": [1, 3, 5, 7, 9, 21, 29]},
        "lr": None,
        "lda": None,
        "svc": {"C": [2 ** i for i in range(-3, 11)], "kernel": ['rbf'], "gamma": [2 ** i for i in range(-5, 6)],
                "probability": [True]},
        "dt": {"min_samples_leaf": [1, 2, 3, 5], "min_samples_split": [5, 10]},
        "mlp": {'hidden_layer_sizes': [tuple([i]) for i in range(3, 21)], 'max_iter': [300]}
    }
    classifiers_dict = {
        "knn": KNeighborsClassifier,
        "lr": LogisticRegression,
        "lda": LinearDiscriminantAnalysis,
        "svc": SVC,
        "dt": DecisionTreeClassifier,
        "mlp": MLPClassifier
    }
    error_function = make_scorer(binary_mean_error_score, greater_is_better=False)

    def __init__(self, num_classes, cv_folds=3, n_jobs=1, n_iterations=50):
        """
        hard = majority voting rule
        soft = argmax of the sum of predicted probabilities
        """
        self.cv_folds = cv_folds
        self.n_jobs = n_jobs
        self.n_iterations = n_iterations
        self.fraction_for_validation = (cv_folds - 1) / cv_folds
        self.classifier_per_pair = {}
        self.hyper_parameters_per_pair = {}
        self.num_classes = num_classes
        self.ran_fit = False

    @ignore_warnings(category=ConvergenceWarning)
    def get_score_and_params_for_classifier(self, x_train, y_train, name):
        """
        Performs 3 fold cross validation with a randomized grid-search, to find the best classifier
        and parameters for a given pair of labels.

        :param x_train: Training data
        :param y_train: Training labels (Assumed to contain only two classes)
        :param name: classifier name
        :return: model kfold score and best params found
        """

        classifier_class = self.classifiers_dict[name]
        param_grid = self.param_grids[name]
        param_grid_copy = param_grid

        # making sure that n_samples <= k while using knn
        if name == "knn":
            param_grid_copy = param_grid.copy()
            param_grid['n_neighbors'] = [i for i in param_grid['n_neighbors'] if
                                         i <= int(self.fraction_for_validation * len(x_train))]

        if param_grid is None:
            model_score = cross_val_score(classifier_class(), x_train, y_train, cv=self.cv_folds,
                                          scoring=self.error_function,
                                          n_jobs=self.n_jobs, error_score=-1)
            model_score = np.mean(model_score)
            model_params = {}
        else:
            cv_search = RandomizedSearchCV(classifier_class(), self.param_grids[name], cv=self.cv_folds,
                                           scoring=self.error_function, n_iter=self.n_iterations, n_jobs=self.n_jobs,
                                           error_score=-1)

            # Error within the dataset --> return an error score
            cv_search.fit(x_train, y_train)
            model_score = cv_search.best_score_
            model_params = cv_search.best_params_

        # score is negative because sklearn always tries to maximize the score function. So when the best score
        # needs to be minimized (greater_is_better = False) score values are
        # multiplied by -1 (like neg-mean-squared-error).

        assert model_score <= 0, f'model score : {model_score}, model name : {name}'
        if name == "knn":
            self.param_grids[name] = param_grid_copy

        return -1 * model_score, model_params

    def get_best_classifier_by_cv_score(self, x_train, y_train):
        """
        :param x_train: reduced feature arrays corresponding to the given labels
        :param y_train: assumed to contain ONLY two labels
        :return: Best OAO binary classifier found.
        """
        assert len(set(y_train)) == 2, "Only two labels are expected"
        assert len(y_train) == len(x_train), "Labels and features length should match"

        best_cv_score = float("inf")
        best_classifier = None
        best_params = None

        # iterating trough all classifiers
        for name in self.classifiers_dict.keys():
            model_score, model_params = self.get_score_and_params_for_classifier(x_train, y_train, name)

            # is the best score better than the current one ?
            if model_score < best_cv_score:
                best_cv_score = model_score
                best_classifier = name
                best_params = model_params

            # optimization: 0 is the best score therefore we can stop.
            if model_score == 0:
                break

        chosen_class = self.classifiers_dict[best_classifier]
        classifier = chosen_class(**best_params)

        # training chosen model on whole train data
        classifier.fit(x_train, y_train)
        return classifier, best_params

    def fit(self, x_train, y_train):
        """
        :param x_train: training data
        :param y_train: training labels
        :return: None
        """
        self.classifier_per_pair.clear()
        self.hyper_parameters_per_pair.clear()

        assert len(x_train) == len(y_train), "labels and features length should match"

        labels = set(y_train)
        possible_pairs = combinations(labels, 2)
        class_dict = setup_class_dict(x_train, y_train)

        for i, j in possible_pairs:
            x_i_j, y_i_j = join_two_classes(i, j, class_dict)

            # shuffling to remove bias
            x_i_j, y_i_j = shuffle(x_i_j, y_i_j, random_state=42)

            assert (i, j) not in self.classifier_per_pair

            chosen_classifier, chosen_params = self.get_best_classifier_by_cv_score(x_i_j, y_i_j)
            self.classifier_per_pair[(i, j)] = chosen_classifier
            self.hyper_parameters_per_pair[(i, j)] = chosen_params

        self.ran_fit = True

    def predict(self, x_test, verbose=False, output_predictions=True):
        """
        :param output_predictions: if true predictions are returned, otherwise the voting matrix is returned.
        :param x_test: samples to predict on
        :param verbose: if true - outputs the classifiers chosen for each pair of labels and the chosen params.
        :return: an array of predicted labels or a voting-matrix if output_predictions is false.
        """
        assert len(x_test) > 0, "Cannot predict on empty sample list"
        assert self.ran_fit, "Run Fit method before running Predict"

        num_samples = len(x_test)
        voting_matrix = np.zeros((num_samples, self.num_classes))

        for i, j in self.classifier_per_pair.keys():
            classifier = self.classifier_per_pair[(i, j)]

            if verbose:
                print(f"Chosen classifier for the pair ({i},{j}) is - {type(classifier).__name__}")

            preds = classifier.predict(x_test)

            for index, y_pred in enumerate(preds):
                if y_pred == 0:
                    voting_matrix[index][i] += 1
                else:
                    voting_matrix[index][j] += 1

        if output_predictions:
            return np.argmax(voting_matrix, axis=1)
        else:
            return voting_matrix

    def predict_proba_per_pair(self, class_dict):
        """
        :param class_dict: A class-dict mapping as mentioned before (see setup_class_dict)
        :return: A dictionary that contains a mapping from each-pair of classes to
        the probabilities predicted by their selected binary classifier.
        This function is mainly used to calculate metrics and for sanity-checking.
        """

        assert self.ran_fit, "Run the Fit method before running Predict"
        labels = set(class_dict.keys())
        possible_pairs = combinations(labels, 2)
        predictions_dict = {}

        for i, j in possible_pairs:
            if (i, j) not in self.classifier_per_pair:
                continue

            x_i_j, y_i_j = join_two_classes(i, j, class_dict)
            model = self.classifier_per_pair[(i, j)]
            probas = model.predict_proba(x_i_j)[:, 1]
            predictions_dict[str((i, j))] = [float(i) for i in probas]

        return predictions_dict

    def get_chosen_classifiers_and_params(self):
        """
        Used for debugging and to get the hyperparams information.
        :return: Outputs a dictionary that contains a mapping for each pair of classes, that contains the
        chosen binary classifier for that pair and the chosen hyperparameters.
        """
        info_dict = {}
        for i, j in self.classifier_per_pair.keys():
            info_dict[str((i, j))] = {}
            info_dict[str((i, j))]["chosen_classifier"] = type(self.classifier_per_pair[(i, j)]).__name__
            info_dict[str((i, j))]["chosen_hyperparams"] = self.hyper_parameters_per_pair[(i, j)].copy()

        return info_dict
