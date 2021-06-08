import argparse
from rule_based_classifiers import RULES

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd
pd.options.mode.chained_assignment = None


parser = argparse.ArgumentParser()  # TODO add help everywhere

# Dataset Arguments
parser.add_argument('--dataset', type=str, help='Name of the dataset to use', default='mushroom')
parser.add_argument('--no_header', action='store_true', help='Flag to indicate the dataset file does not contain a '
                                                             'header row')
parser.add_argument('--has_index', action='store_true', help='Flag to indicate the dataset file contains an index '
                                                             'column')
parser.add_argument('--class_first', action='store_true', help='Flag to indicate the class column is the first one')

# Preprocessing Arguments
parser.add_argument('--discretize_ints', action='store_true', help='If set, integers are treated as a continuous '
                                                                   'class')
parser.add_argument('--bins', type=int, default=7, help='Number of bins in which to discretize non-categorical '
                                                        'attributes')
parser.add_argument('--discretize_mode', type=str, default='equal', choices=['equal', 'freq'])

# Training Arguments
parser.add_argument('--train_only', action='store_true', help='If set, all data is used to induce rules')
parser.add_argument('--method', type=str, default='RRULES', choices=['Original', 'RRULES'])
parser.add_argument('--print_time', action='store_true', help='Print induction')
parser.add_argument('--print_rules', action='store_true', help='Print inducted rules')
parser.add_argument('--print_metrics', action='store_true', help='Print metrics for inducted rules')

args = parser.parse_args()

#############################################

# Read CSV
dataset_path = f"data/{args.dataset}.csv"
dataframe = pd.read_csv(dataset_path,
                        header=0 if not args.no_header else None,
                        index_col=0 if args.has_index else None)

# Features - Class split
if args.class_first:
    x_df = dataframe.iloc[:, 1:]
    y_df = dataframe.iloc[:, :1]
else:
    x_df = dataframe.iloc[:, :-1]
    y_df = dataframe.iloc[:, -1:]


# Train - Test split
if not args.train_only:
    try:
        x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=9, stratify=y_df)
    except ValueError:
        print('Not enough samples to stratify. Generating random train-test splits...')
        x_train, x_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.2, random_state=9)
else:
    x_train = x_df.copy()
    y_train = y_df.copy()
    x_test = x_df.copy()
    y_test = y_df.copy()

# Create RULES object and train
rules = RULES(contains_header=not args.no_header,
              number_bins=args.bins,
              discretize_mode=args.discretize_mode,
              discretize_ints=args.discretize_ints)
rules.fit(x_train,
          y_train,
          method=args.method,
          show_rules=args.print_rules,
          show_time=args.print_time,
          show_metrics=args.print_metrics)


# Predict and get accuracy report
if not args.train_only:
    y_pred = rules.predict(x_test)
    print(f"Test Accuracy = {100*accuracy_score(y_test.to_numpy(), y_pred):.2f}%")
    print(classification_report(y_test.to_numpy(), y_pred))
