This is Luminoso's entry to SemEval-2018 task 10, "Capturing Discriminative Attributes".

The current code was submitted as run 2, which achieved an accuracy of 72.81% on the
validation set.

Run 1 got a validation accuracy of 72.30%. The differences in run 2 are:

- It uses the task's updated training data
- It uses a Semantic Matching Energy model that trained for 3 more days
- It uses 11 features for SME results
- It discards features that end up with a negative weight

As all the features were intended to be positive, removing negative-weighted
features may be a way to isolate and remove components that are doing nothing
but overfitting.
