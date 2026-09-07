# Cheat sheet 05 — Reviewing someone else's model

A working checklist for reviewing ML work you did not build — a vendor
demonstration, an internal team's model, a validation exercise. Ordered by how
often each question finds something.

The framing that makes this efficient: **you are not checking whether they used
a good algorithm. You are checking whether the number they are quoting means
what they think it means.**

---

## 1. Framing (before any technical question)

- [ ] What decision does this model change? Who acts on it, and how?
- [ ] What does a false positive cost? A false negative? **Who owns those
      numbers?** If the modelling team invented them, that is the finding.
- [ ] What is the deployment scenario — new entities or known ones? Forward in
      time or contemporaneous?
- [ ] What is the *incumbent*, and how does the model compare to it? "Better
      than nothing" is not the relevant comparison if a manual process exists.

## 2. Leakage — the highest-yield section

- [ ] **"Which decisions were made by looking at the whole dataset?"** Feature
      selection, encoding, outlier removal, resampling, threshold choice,
      choice of model family.
- [ ] Is the preprocessing inside the pipeline, or applied before the split?
- [ ] For every feature: **when is this value written, relative to the outcome?**
      Ask for data lineage on the top five features by importance.
- [ ] Does any single feature score nearly as well as the whole model?
- [ ] Are identifiers, row numbers or timestamps in the feature set?
- [ ] If target encoding is used — is it cross-fitted?

## 3. Validation

- [ ] Does the splitter match the deployment scenario? (Entities → grouped.
      Forward in time → temporal.)
- [ ] `KFold` without `shuffle=True` on data that arrived sorted?
- [ ] Is there a holdout, and how many times has it been scored?
- [ ] **What is the noise floor?** Ask for `mean ± std` over repeated CV. Then
      check whether any claimed improvement exceeds it.
- [ ] How large was the hyperparameter search, and is the reported score
      `best_score_`? (It is biased upward, and the bias grows with search size.)
- [ ] Were fairness and calibration checked before or after the holdout was
      opened?

## 4. Metrics

- [ ] Is accuracy being quoted on an imbalanced problem?
- [ ] On an imbalanced problem, is there an **average precision** figure, not
      just ROC AUC?
- [ ] What is precision **at the operating point**, and what is the resulting
      alert/referral volume per day? Can the team handle it?
- [ ] Is the model calibrated? Compare mean predicted probability to the
      observed base rate. If they differ, every derived number is wrong.
- [ ] Was the threshold chosen from costs, or left at 0.5?
- [ ] Are AP figures being compared across periods with different base rates?

## 5. Interpretability

- [ ] Is `feature_importances_` (impurity) being presented as evidence? It is
      biased towards high-cardinality features — a row counter will rank highly.
- [ ] Is permutation importance computed on **held-out** data?
- [ ] Are any important features strongly correlated with each other? If so,
      single-feature importances are uninterpretable — ask for grouped
      permutation.
- [ ] If a PDP is shown, is it being read outside the 5th–95th percentile?
- [ ] If a surrogate tree is shown, **what is its fidelity (R²)?** Absent that
      number, the tree invites people to believe the model is simpler than it is.
- [ ] Do the SHAP values use interventional or conditional perturbation, and
      what background set?

## 6. Fairness (regulated contexts)

- [ ] Which fairness criterion has been chosen, and why? They are mutually
      incompatible when base rates differ — the choice must be explicit.
- [ ] Four-fifths selection-rate ratio, per protected attribute, **at the
      deployed threshold**. (It changes with the threshold.)
- [ ] Calibration *within* each group, not just overall.
- [ ] **Proxy test:** can the protected attribute be predicted from the
      remaining features? If AUC ≫ 0.5, "we don't use it" is a statement about
      the schema, not the model.
- [ ] If group-specific thresholds are proposed — has legal signed off? In many
      lending jurisdictions this is disparate treatment.
- [ ] Has legal been involved, or is this a modelling team's interpretation?

## 7. Robustness and production

- [ ] What happens to an unseen category at scoring time? (`handle_unknown`)
- [ ] What happens to a missing value that was never missing in training?
- [ ] Is there an input schema contract, and what does it do on violation?
- [ ] What is the persistence format, and is the loading environment pinned to
      the saving one?
- [ ] Is there a reference batch with expected outputs, to verify a deployment?
- [ ] **Are scores logged at serving time?** Without that log there is no
      monitoring, only a monitoring plan.
- [ ] What are the alert thresholds, and what is their expected false-alarm rate?
- [ ] Covariate drift is detectable without labels. Concept drift is not. How
      long is the outcome lag, and what is the early proxy?
- [ ] What triggers a retrain, and who decides?

## 8. Documentation

- [ ] Is there a model card: intended use, data, performance, limitations,
      fairness, monitoring?
- [ ] Are the limitations honest, or a formality?
- [ ] Can someone else reproduce the result from what is written down?

---

## Five questions that find the most, fastest

1. **"Which decisions were made by looking at all the data?"**
2. **"What is the fold-to-fold standard deviation, and is your improvement
   bigger than it?"**
3. **"At the deployed threshold, what is precision and what is the daily
   volume?"**
4. **"Does the mean predicted probability match the observed base rate?"**
5. **"At inference time, will you have seen this entity before?"**

## Three answers that should worry you

- *"We used XGBoost so it's state of the art."* — the algorithm is the least
  consequential choice in the pipeline.
- *"AUC is 0.94."* — on what split, against what noise floor, at what operating
  point, and is it calibrated?
- *"We removed the protected attributes."* — did you test whether they are
  recoverable? If so, you removed your ability to measure the disparity, not the
  disparity.

## And one finding that is always worth reporting

**"We ran a 500-point search and gained 0.004, which is inside the measurement
noise."** Redirecting a workstream away from something that cannot move the
outcome is worth more than a marginally better model.
