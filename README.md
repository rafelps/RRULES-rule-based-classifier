# RRULES-rule-based-classifier
Official code for our paper [RRULES: An improvement of the RULES rule-based classifier][paper].

RRULES is presented as an improvement and optimization over RULES, a simple inductive learning algorithm for 
extracting IF-THEN rules from a set of training examples. RRULES optimizes the algorithm by implementing a more 
effective mechanism to detect irrelevant rules, at the same time that checks the stopping conditions more often. 
This results in a more compact rule set containing more general rules which prevent overfitting the training set and 
obtain a higher test accuracy. Moreover, the results show that RRULES outperforms the original algorithm by reducing 
the coverage rate up to a factor of 7 while running twice or three times faster consistently over several datasets. 

This repository contains both the code for our algorithm [RRULES][paper], and an implementation of the original paper 
[RULES][rules].

## Requirements
This project has been build using:
- [Python][python] 3.7
- [NumPy][numpy] 1.19.2
- [Pandas][pandas] 1.2.0
- [Scikit-learn][sklearn] 0.24.2

## Usage
The complete usage can be seen typing:
```bash
$ python main.py -h
```

The most important arguments are:
- `--dataset dataset`: Name of the dataset to use. There are several datasets inside the `data` folder, but the user 
  can add 
  any other in .csv format. By default, the script expects a header row, no index column and the class in the last 
  column.
- `--no_header`: Flag to indicate that there is no header in the specified dataset. The first row will be used as 
  the first example.
- `--has_index`: Flag to indicate that the first column of the .csv file should be treated as index and not as 
  attribute.
- `--class_first`: Flag to indicate that the first column contains the class and not an attribute.

Rule-based algorithms are usually defined for discrete attributes and class only. This implementation, nonetheless, 
contains a preprocessing step that discretizes any continuous attribute into different bins, so the algorithm can be 
applied to any dataset. Regarding the preprocessing step, there are the following parameters:
- `--discretize_ints`: Flag to indicate that integers have to be discretized. By default, the script treats integers 
  as categories (as many categorical datasets are encoded this way).
- `--bins bins` : Number of bins in which numerical attributes should be discretized in.
- `--discretize_mode 'equal'|'freq'`: If 'equal' is selected, the range of any numerical attribute will be divided 
  into `bins` equal bins. If 'freq' is selected, bin boundaries will be decided such that each bin contain the same 
  number of examples.
  
Finally, there are some parameters that let the user control the training and output:
- `--train_only`: If set, the script will not split the data into train and test, and only train stage will be run. 
  This flag should be set to reproduce the results of the original [RULES paper][rules].
- `--method 'Original'|'RRULES'`: Induction algorithm to use. Defaults to 'RRULES'.
- `--print_time`: If set, the script outputs the induction time.
- `--print_rules`: If set, the script outputs the inducted rule set.
- `--print_metrics`: If set, the script outputs the Precision and Coverage metrics for each rule, as well as overall 
  metrics.
  
Usage example:
```bash
$ python main.py --dataset mushroom --print_time --print_rules --print_metrics
```

## Citation
You can cite our work using:
```bibtex
@misc{pallisersans2021rrules,
      title={RRULES: An improvement of the RULES rule-based classifier}, 
      author={Rafel Palliser-Sans},
      year={2021},
      eprint={2106.07296},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

[paper]: https://arxiv.org/abs/2106.07296
[rules]: https://www.sciencedirect.com/science/article/abs/pii/S0957417499800086
[python]: https://www.python.org/
[numpy]: https://numpy.org/
[pandas]: https://pandas.pydata.org/
[sklearn]: https://scikit-learn.org/
