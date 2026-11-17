# Data science conventions
- Logic lives in `src/<package>/` modules (loading, features, models, metrics) and is unit-tested. Notebooks and `reports/` only call those modules and present results.
- Reproducible: fixed random seeds passed explicitly, data snapshots identified by path and hash, dependencies locked.
- Start with exploratory analysis: shape, types, missingness, distributions, leakage checks. Write down what you found before modelling.
- Statistics: state the hypothesis, check test assumptions, report effect sizes with confidence intervals, not only p-values. Correct for multiple comparisons.
- Modelling: split before any fitting (no leakage), use pipelines for preprocessing, cross-validate, compare against a simple baseline, and report the metric that matches the question.
- Every number in a report comes from code that's tested; charts have titles, labeled axes with units, and a sentence saying what they show.
- Reports end with findings, caveats and limitations in plain language.
