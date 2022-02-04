import time
import numpy as np
import pandas as pd
from itertools import combinations, product
from sklearn.metrics import accuracy_score
from collections import defaultdict


class RULES:
    def __init__(self, contains_header=True, discretize_mode='equal', number_bins=7, discretize_ints=False):
        self.contains_header = contains_header
        self.discretize_mode = discretize_mode
        self.number_bins = number_bins
        self.discretize_ints = discretize_ints
        self.bins = []

        self.attribute_names = None
        self.preproc_dict = None
        self.labels_dict = None

        self.rules = []
        self.most_probable_class = None
        self.n_attributes = 0

    # ######################### F I T #########################
    def fit(self, x, y, method='RRULES', show_rules=True, show_metrics=True, show_time=True):
        since = time.time()
        x, y = self.__preproc_train_data(x, y)
        preproc_time = time.time() - since

        since = time.time()
        if method == 'RRULES':
            print('Training with RRULES...')
            self.__fit_RRULES(x, y)
        elif method == 'Original':
            print('Training with Original RULES...')
            self.__fit_original_RULES(x, y)
        fit_time = time.time() - since

        since = time.time()
        if show_rules:
            metrics = None
            if show_metrics:
                metrics = self.__compute_metrics(x, y)
                metrics_time = time.time() - since
                since = time.time()
            print('\nInduced rules:')
            print(self.__print_rules(metrics))
            print()
        print_time = time.time() - since

        if show_time:
            print(f"Time to preprocess data = {preproc_time:.2f}s")
            print(f"Time to fit data = {fit_time:.2f}s")
            if show_rules:
                if show_metrics:
                    print(f"Time to comput metrics = {metrics_time:.2f}s")
                print(f"Time to print rules = {print_time:.2f}s")
            print()

    def __fit_RRULES(self, x, y):
        # We calculate the most probable class to create a default rule for unseen combinations of attributes
        classes, counts = np.unique(y, return_counts=True)
        self.most_probable_class = classes[np.argmax(counts)]

        # ##### RRULES #####
        n_examples, n_attributes = x.shape
        self.n_attributes = n_attributes
        # Track non-classified by index
        indices_not_classified = np.arange(n_examples)

        # For each n_conditions = 1, ..., n_attributes
        for n_conditions in range(1, n_attributes + 1):
            # Generate all possible combinations of attributes (without repetition and without order)
            # of length n_conditions
            attribute_combinations_n = combinations(range(n_attributes), n_conditions)
            # For each combination of attributes (columns)
            for attribute_group in attribute_combinations_n:
                lists_of_values = []
                # Calculate the unique values of the chosen attributes from the non-classified instances,
                # and generate all combinations of selectors <attribute, value> given the chosen attributes
                # These combination of selectors form conditions
                for attribute in attribute_group:
                    lists_of_values.append(np.unique(x[indices_not_classified, attribute]))
                value_combinations = product(*lists_of_values)
                # For each condition <att1, val1>, <att2, val2>, ...
                for value_group in value_combinations:
                    # Find indices of ALL INSTANCES that match the condition
                    indices_match = np.where((x[:, list(attribute_group)] == value_group).all(axis=1))[0]
                    # Find indices of NON-CLASSIFIED INSTANCES that match the condition
                    indices_match_not_classified = \
                        np.where((x[np.ix_(indices_not_classified, attribute_group)] == value_group).all(axis=1))[0]
                    if len(indices_match) == 0:
                        # This condition is not present in the training set of examples
                        continue
                    if len(indices_match_not_classified) == 0:
                        # Although this condition is present in the set of examples,
                        # it does not match any non-classified instance
                        # It is the case of a condition that could end generating an IRRELEVANT RULE
                        continue
                    # Take the ground truth of the matched instances and look if they belong to a single class
                    classes = y[indices_match]
                    unique_classes = np.unique(classes)
                    if len(unique_classes) == 1:
                        # Generate the rule and add it to the set of rules
                        # The rule is encoded as the set of attributes to match, their values and the class
                        self.rules.append((attribute_group, value_group, unique_classes[0]))
                        # Remove the classified instances from the set of non-classified ones
                        indices_not_classified = np.setdiff1d(indices_not_classified, indices_match, assume_unique=True)
                        # If there aren't any more instances to classify, return
                        if len(indices_not_classified) == 0:
                            return
                    # If we are in the last iteration, we are checking for the full antecedent
                    # If there is more than a single class, we have a contradiction in the data
                    # Let's choose the most probable class (or random if tie)
                    elif n_conditions == n_attributes:
                        print("WARNING: There are contradictions in the training set")
                        unique_classes, counts = np.unique(classes, return_counts=True)
                        self.rules.append((attribute_group, value_group, unique_classes[np.argmax(counts)]))
                        # Remove the classified instances from the set of non-classified ones
                        indices_not_classified = np.setdiff1d(indices_not_classified, indices_match, assume_unique=True)
                        # If there aren't any more instances to classify, return
                        if len(indices_not_classified) == 0:
                            return

    def __fit_original_RULES(self, x, y):
        # We calculate the most probable class to create a default rule for unseen combinations of attributes
        classes, counts = np.unique(y, return_counts=True)
        self.most_probable_class = classes[np.argmax(counts)]

        # ##### RULES #####
        n_examples, n_attributes = x.shape
        self.n_attributes = n_attributes
        # Track non-classified by index
        indices_not_classified = np.arange(n_examples)

        # For each n_conditions = 1, ..., n_attributes
        for n_conditions in range(1, n_attributes + 1):
            # Check stopping condition only at the beginning of every outer iteration
            if len(indices_not_classified) == 0:
                return
            # Generate all combinations of selectors of length n_conditions
            conditions = []
            # Generate all possible combinations of attributes (without repetition and without order)
            # of length n_conditions
            attribute_combinations_n = combinations(range(n_attributes), n_conditions)
            # For each combination of attributes (columns)
            for attribute_group in attribute_combinations_n:
                lists_of_values = []
                # Calculate the unique values of the chosen attributes, and generate
                # all combinations of selectors <attribute, value> given the chosen attributes
                # These combination of selectors form conditions
                for attribute in attribute_group:
                    lists_of_values.append(np.unique(x[indices_not_classified, attribute]))
                value_combinations = product(*lists_of_values)
                for value_group in value_combinations:
                    conditions.append((attribute_group, value_group))

            # For each condition <att1, val1>, <att2, val2>, ...
            for attribute_group, value_group in conditions:
                # Find indices of ALL INSTANCES that match the condition
                indices_match = np.where((x[:, list(attribute_group)] == value_group).all(axis=1))[0]
                if len(indices_match) == 0:
                    # This condition is not present in the training set of examples
                    continue
                # Take the ground truth of the matched instances and look if they belong to a single class
                classes = y[indices_match]
                unique_classes = np.unique(classes)
                if len(unique_classes) == 1:
                    # Check for irrelevant conditions
                    is_irrelevant = False
                    if n_conditions > 1:
                        for rule in self.rules:
                            # If there is a previous rule (with less conditions) that includes all the selectors for
                            # the new rule, this becomes irrelevant as it does not classify any new instance
                            if all(selector in zip(attribute_group, value_group) for selector in zip(rule[0], rule[1])):
                                is_irrelevant = True
                                break
                    if not is_irrelevant:
                        # Generate the rule and add it to the set of rules
                        # The rule is encoded as the set of attributes to match, their values and the class
                        self.rules.append((attribute_group, value_group, unique_classes[0]))
                        # Remove the classified instances from the set of non-classified ones
                        indices_not_classified = np.setdiff1d(indices_not_classified, indices_match, assume_unique=True)

                # If we are in the last iteration, we are checking for the full antecedent
                # If there is more than a single class, we have a contradiction in the data
                # Let's choose the most probable class (or random if tie)
                # There won't be irrelevant conditions here because it is the last iteration and we have
                # non-classified instances!
                elif n_conditions == n_attributes:
                    print("WARNING: There are contradictions in the training set")
                    unique_classes, counts = np.unique(classes, return_counts=True)
                    self.rules.append((attribute_group, value_group, unique_classes[np.argmax(counts)]))
                    # Remove the classified instances from the set of non-classified ones
                    indices_not_classified = np.setdiff1d(indices_not_classified, indices_match, assume_unique=True)

    # ######################### P R E D I C T #########################
    def predict(self, x):
        # Predict
        y_pred = self.__predict(x)
        # Predictions are integers, convert back to original values
        y_pred = np.vectorize(self.labels_dict[-1].get)(y_pred)
        return y_pred

    def score(self, x, y):
        y_pred = self.__predict(x)
        # Convert true class to integers (predictions already are)
        y = np.vectorize(self.preproc_dict[-1].get)(y)
        return accuracy_score(y, y_pred)

    def __predict(self, x):
        print('Predicting...')
        x = self.__preproc_test_data(x)
        y_pred = []
        # For each instance
        for instance in x:
            classified = False
            # For each rule
            for attributes, values, tag in self.rules:
                # Check if antecedent matches
                if np.array_equal(instance[list(attributes)], values):
                    y_pred.append(tag)
                    classified = True
                    break
            # No rule matched, we apply the default rule --> Most probable class
            if not classified:
                print("WARNING: There are unseen combinations")
                y_pred.append(self.most_probable_class)
        return np.array(y_pred)

    # ################ M E T R I C S   A N D   R U L E S ################
    def compute_metrics(self, x, y):
        # Use the already computed (training) bins and integer conversions
        x = self.__preproc_test_data(x)
        y = np.vectorize(self.preproc_dict[-1].get)(y)
        return self.__compute_metrics(x, y)

    def __compute_metrics(self, x, y):
        n_examples = x.shape[0]

        # Store (Coverage, Precision) for each rule
        metrics = []
        overall_coverage = []
        overall_precision = []
        for attributes, values, tag in self.rules:
            indices_match_condition = np.where((x[:, list(attributes)] == values).all(axis=1))[0]
            coverage = len(indices_match_condition) / n_examples
            indices_match_rule = np.where(y[indices_match_condition] == tag)[0]
            precision = len(indices_match_rule) / len(indices_match_condition)
            overall_precision.append(precision)
            overall_coverage.append(coverage)
            metrics.append((coverage, precision))
        # Add overall metrics
        metrics.append((sum(overall_coverage), sum(overall_precision) / len(overall_precision)))

        return metrics

    def __print_rules(self, metrics=None):
        # Set default attribute names if not present in dataset
        if self.attribute_names is None:
            attribute_names = [f"Attribute_{i + 1}" for i in range(self.n_attributes)]
            attribute_names.append("Class")
            self.attribute_names = attribute_names

        rule_strings = []
        for i in range(len(self.rules)):
            attributes, values, tag = self.rules[i]
            rule = f"Rule {i + 1:3}.  IF " \
                   f"{' AND '.join([f'{self.attribute_names[attributes[j]]} IS {self.labels_dict[attributes[j]][values[j]]}' for j in range(len(attributes))])}" \
                   f" THEN {self.attribute_names[-1]} IS {self.labels_dict[-1][tag]}"

            if metrics:
                rule = f"{rule:100}    Coverage = {100 * metrics[i][0]:5.2f}%    Precision = " \
                       f"{100 * metrics[i][1]:5.2f}%"
            rule_strings.append(rule)
        if metrics:
            overall = f"Overall Coverage = {100 * metrics[-1][0]:5.2f}%     Overall Precision = {100 * metrics[-1][1]:5.2f}%"
            rule_strings.append(overall)
        return '\n'.join(rule_strings)

    # ############### P R E P R O C E S S I N G ###############
    def __preproc_train_data(self, x, y):
        # Set attribute names for pretty printing
        if self.contains_header:
            names_x = x.columns.values.tolist()
            name_y = y.columns.values.tolist()
            self.attribute_names = names_x + name_y

        column_types = x.dtypes
        # Missing Values
        for attribute, dtype in zip(x, column_types):
            # We take the mean for floats
            if np.issubdtype(dtype, np.floating):
                x.loc[:, attribute].fillna(x[attribute].mean(), inplace=True)
            # We take the mode for categoricals (including integers)
            else:
                # Intermediate conversion into '?' (to join different representations for mv)
                x.loc[:, attribute].fillna('?', inplace=True)
                uniques, counts = np.unique(x[attribute], return_counts=True)
                mode = uniques[np.argmax(counts)]
                if mode == '?':
                    mode = uniques[np.argsort(counts)[-2]]
                x.loc[x[attribute] == '?', attribute] = mode

        # Discretize Numeric attributes
        # Store exact discretization bins for test data
        if self.number_bins != 0:
            to_discretize = np.number if self.discretize_ints else np.floating
            for attribute, dtype in zip(x, column_types):
                if np.issubdtype(dtype, to_discretize):
                    if self.discretize_mode == 'equal':
                        x[attribute], bins = pd.cut(x[attribute], bins=self.number_bins, retbins=True)
                        self.bins.append((attribute, bins, np.unique(x[attribute])))
                    elif self.discretize_mode == 'freq':
                        x[attribute], bins = pd.qcut(x[attribute], q=self.number_bins, retbins=True)
                        self.bins.append((attribute, bins, np.unique(x[attribute])))
                    else:
                        raise ValueError("Wrong discretize_mode")

        # Move everything to integer, so numpy works faster
        x = x.to_numpy()
        y = y.to_numpy()
        data = np.concatenate((x, y), axis=1)
        _, n_cols = data.shape

        # Store conversion from original values to integers for pretty printing
        inv_conversions = []
        conversions = []
        for i in range(n_cols):
            col = data[:, i]
            uniques = np.unique(col).tolist()
            d = defaultdict(lambda: -1, zip(uniques, range(len(uniques))))
            d_inv = dict(zip(range(len(uniques)), uniques))
            data[:, i] = np.vectorize(d.get)(col)
            conversions.append(d)
            inv_conversions.append(d_inv)

        # Preprocessing Dictionary
        # Contains all the conversions from original values to integers --> To be used when preprocessing test data
        self.preproc_dict = conversions
        # Labels Dictionary
        # Contains all the conversions from integers to original values --> To be used when pretty printing
        self.labels_dict = inv_conversions

        return data[:, :-1].astype(np.uint8), data[:, -1].astype(np.uint8)

    def __preproc_test_data(self, x):
        # Preprocess only attributes (not class)
        # Use the same exact steps than in training
        #   MV
        #   Discretization using same bins
        #   Conversion to integers using same mapping

        column_types = x.dtypes
        # Missing Values
        for attribute, dtype in zip(x, column_types):
            # We take the mean for floats
            if np.issubdtype(dtype, np.floating):
                x.loc[:, attribute].fillna(x[attribute].mean(), inplace=True)
            # We take the mode for categoricals (including integers)
            else:
                # Intermediate conversion into '?' (to join different representations for mv)
                x.loc[:, attribute].fillna('?', inplace=True)
                uniques, counts = np.unique(x[attribute], return_counts=True)
                mode = uniques[np.argmax(counts)]
                if mode == '?':
                    mode = uniques[np.argsort(counts)[-2]]
                x.loc[x[attribute] == '?', attribute] = mode

        # Discretize Numeric attributes using training bins
        for attribute, bins, labels in self.bins:
            if len(labels) + 1 == len(bins):
                x[attribute] = pd.cut(x[attribute], bins=bins, labels=labels)
            else:
                x[attribute] = pd.cut(x[attribute], bins=bins)

        # Move everything to integer, so numpy works faster
        data = x.to_numpy()
        _, n_cols = data.shape

        # Use original-integer training mapping
        for i in range(n_cols):
            col = data[:, i]
            data[:, i] = np.vectorize(self.__my_vectorized_mapping)(i, col)

        return data.astype(np.uint8)

    def __my_vectorized_mapping(self, i, x):
        return self.preproc_dict[i][x]
    
print('Hello World')
