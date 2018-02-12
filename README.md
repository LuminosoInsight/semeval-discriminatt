This is Luminoso's entry to SemEval-2018 task 10, "Capturing Discriminative Attributes".

This code corresponds to run 3, a late entry to fix a show-stopping bug in producing the
test results. Due to a caching bug (see #15), the test results we submitted for runs 1
and 2 were actually a copy of our validation results, and were therefore useless for
evaluation. Run 3 was identical to run 2 except that the code to produce the test results
was fixed.

Run 1 got a validation accuracy of 72.30%, and runs 2 and 3 got a validation accuracy
of 72.85%. Run 3 achieved a test accuracy of 73.68%, and can be found as our entry on the
[post-evaluation leaderboard](https://competitions.codalab.org/competitions/17326#results)
on CodaLab.
