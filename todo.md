# Tasks
## Task 1: Tokeniser
- [x] implement a tokeniser to split the input texts into meaningful tokens:
  - [x] use regex
  - [ ] one output line per input line
  - [ ] spaces between tokens
- [x] use the training dataset
  - [x] split it into train and test - do not use validation dataset
- [x] Compare the output of out tokeniser with the NLTK tweet tokeniser

## Task 2: Corpus stats
- [x] differences between the two datasets (comparisons made between irony and atheism)
- [x] most frequent tokens
- [x] zipfs law plot for both datasets
  - [x] irony
  - [x] stance (atheism only)

## Task 3: Manual annotation
Chosen to annotate the irony dataset
- [x] Annotate the dataset individually
- [ ] Report on what characteristics of the data caused the biggest problems for agreement.

## Task 4: Training and prediction
Use scikit-learn to train a classifier for the automatic prediction of the labels in the two datasets you have chosen.

- [ ] start with the sklearn.linearmodel.SGDClassifier in a logistic regression configuration (loss=’log’) using bag of words features
- [ ] run at least additional experiments trying to improve your initial scores by any means you can think of. Try out at least 4 different methods.
- [ ] Run all classification experiments on both of the tasks you’ve chosen (one binary and one multi-class task). 
- [ ] Evaluate your different classifiers on the validation set and 
- [ ] report relevant evaluation metrics:
  - [ ] accuracy
  - [ ] precision
  - [ ] recall
  - [ ] F-score

# Report
## Introduction
- [ ] Providing context and motivation for the problem. 
- [ ] What are the main ideas you pursued, and 
- [ ] why does your research provide value?

## Data and Preprocessing
- [ ] Describe the datasets and explain the tasks you selected for your project. 
- [ ] Briefly describe your preprocessing procedure and 
- [ ] the main difficulties you encountered. 
- [ ] Present data statistics and compare between your two datasets if it makes sense to do so.

## Annotation
- [ ] Present the results of your annotation quality check, including:
- [ ] inter-annotator agreement figures, and
- [ ] discuss the most important sources of disagreement, if there was any.

## Classification
- [ ] Briefly describe your classification experiments and 
- [ ] report and 
- [ ] discuss their results.

## Conclusion and Future Work
- [ ] Summarise the main lessons you’ve learnt in this project and 
- [ ] discuss how your work could be improved and extended.

# Hand in
- [ ] report.pdf
- [ ] gitlog.txt
- [ ] code.zip
  - [ ] main notebook
  - [ ] include datasets used
    - [ ] irony
    - [ ] stance
  - [ ] dataset of annotations