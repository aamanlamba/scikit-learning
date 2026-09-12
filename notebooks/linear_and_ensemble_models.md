# Linear Models in sklearn.linear_model

Found 37 classes in sklearn.linear_model:

- **ARDRegression**
  - Description: Bayesian ARD regression.
  - Key Hyperparameters: ['max_iter', 'tol', 'alpha_1', 'alpha_2', 'lambda_1', 'lambda_2', 'compute_score', 'threshold_lambda', 'fit_intercept', 'copy_X', 'verbose']

- **BayesianRidge**
  - Description: Bayesian ridge regression.
  - Key Hyperparameters: ['max_iter', 'tol', 'alpha_1', 'alpha_2', 'lambda_1', 'lambda_2', 'alpha_init', 'lambda_init', 'compute_score', 'fit_intercept', 'copy_X', 'verbose']

- **ElasticNet**
  - Description: Linear regression with combined L1 and L2 priors as regularizer.
  - Key Hyperparameters: ['alpha', 'l1_ratio', 'fit_intercept', 'precompute', 'max_iter', 'copy_X', 'tol', 'warm_start', 'positive', 'random_state', 'selection']

- **ElasticNetCV**
  - Description: Elastic Net model with iterative fitting along a regularization path.
  - Key Hyperparameters: ['l1_ratio', 'eps', 'alphas', 'fit_intercept', 'precompute', 'max_iter', 'tol', 'cv', 'copy_X', 'verbose', 'n_jobs', 'positive', 'random_state', 'selection']

- **GammaRegressor**
  - Description: Generalized Linear Model with a Gamma distribution.
  - Key Hyperparameters: ['alpha', 'fit_intercept', 'solver', 'max_iter', 'tol', 'warm_start', 'verbose']

- **HuberRegressor**
  - Description: L2-regularized linear regression model that is robust to outliers.
  - Key Hyperparameters: ['epsilon', 'max_iter', 'alpha', 'warm_start', 'fit_intercept', 'tol']

- **Lars**
  - Description: Least Angle Regression model aka LAR.
  - Key Hyperparameters: ['fit_intercept', 'verbose', 'precompute', 'n_nonzero_coefs', 'eps', 'copy_X', 'fit_path', 'jitter', 'random_state']

- **LarsCV**
  - Description: Cross-validated Least Angle Regression model.
  - Key Hyperparameters: ['fit_intercept', 'verbose', 'max_iter', 'precompute', 'cv', 'max_n_alphas', 'n_jobs', 'eps', 'copy_X']

- **Lasso**
  - Description: Linear Model trained with L1 prior as regularizer (aka the Lasso).
  - Key Hyperparameters: ['alpha', 'fit_intercept', 'precompute', 'copy_X', 'max_iter', 'tol', 'warm_start', 'positive', 'random_state', 'selection']

- **LassoCV**
  - Description: Lasso linear model with iterative fitting along a regularization path.
  - Key Hyperparameters: ['eps', 'alphas', 'fit_intercept', 'precompute', 'max_iter', 'tol', 'copy_X', 'cv', 'verbose', 'n_jobs', 'positive', 'random_state', 'selection']

- **LassoLars**
  - Description: Lasso model fit with Least Angle Regression aka Lars.
  - Key Hyperparameters: ['alpha', 'fit_intercept', 'verbose', 'precompute', 'max_iter', 'eps', 'copy_X', 'fit_path', 'positive', 'jitter', 'random_state']

- **LassoLarsCV**
  - Description: Cross-validated Lasso, using the LARS algorithm.
  - Key Hyperparameters: ['fit_intercept', 'verbose', 'max_iter', 'precompute', 'cv', 'max_n_alphas', 'n_jobs', 'eps', 'copy_X', 'positive']

- **LassoLarsIC**
  - Description: Lasso model fit with Lars using BIC or AIC for model selection.
  - Key Hyperparameters: ['criterion', 'fit_intercept', 'verbose', 'precompute', 'max_iter', 'eps', 'copy_X', 'positive', 'noise_variance']

- **LinearRegression**
  - Description: 
  - Key Hyperparameters: ['fit_intercept', 'copy_X', 'tol', 'n_jobs', 'positive']

- **LogisticRegression**
  - Description: 
  - Key Hyperparameters: ['penalty', 'C', 'l1_ratio', 'dual', 'tol', 'fit_intercept', 'intercept_scaling', 'class_weight', 'random_state', 'solver', 'max_iter', 'verbose', 'warm_start', 'n_jobs']

- **LogisticRegressionCV**
  - Description: Logistic Regression CV (aka logit, MaxEnt) classifier.
  - Key Hyperparameters: ['Cs', 'l1_ratios', 'fit_intercept', 'cv', 'dual', 'penalty', 'scoring', 'solver', 'tol', 'max_iter', 'class_weight', 'n_jobs', 'verbose', 'refit', 'intercept_scaling', 'random_state', 'use_legacy_attributes']

- **MultiTaskElasticNet**
  - Description: Multi-task ElasticNet model trained with L1/L2 mixed-norm as regularizer.
  - Key Hyperparameters: ['alpha', 'l1_ratio', 'fit_intercept', 'copy_X', 'max_iter', 'tol', 'warm_start', 'random_state', 'selection']

- **MultiTaskElasticNetCV**
  - Description: Multi-task L1/L2 ElasticNet with built-in cross-validation.
  - Key Hyperparameters: ['l1_ratio', 'eps', 'alphas', 'fit_intercept', 'max_iter', 'tol', 'cv', 'copy_X', 'verbose', 'n_jobs', 'random_state', 'selection']

- **MultiTaskLasso**
  - Description: Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
  - Key Hyperparameters: ['alpha', 'fit_intercept', 'copy_X', 'max_iter', 'tol', 'warm_start', 'random_state', 'selection']

- **MultiTaskLassoCV**
  - Description: Multi-task Lasso model trained with L1/L2 mixed-norm as regularizer.
  - Key Hyperparameters: ['eps', 'alphas', 'fit_intercept', 'max_iter', 'tol', 'copy_X', 'cv', 'verbose', 'n_jobs', 'random_state', 'selection']

- **OrthogonalMatchingPursuit**
  - Description: Orthogonal Matching Pursuit model (OMP).
  - Key Hyperparameters: ['n_nonzero_coefs', 'tol', 'fit_intercept', 'precompute']

- **OrthogonalMatchingPursuitCV**
  - Description: Cross-validated Orthogonal Matching Pursuit model (OMP).
  - Key Hyperparameters: ['copy', 'fit_intercept', 'max_iter', 'cv', 'n_jobs', 'verbose']

- **PassiveAggressiveClassifier**
  - Description: Passive Aggressive Classifier.
  - Key Hyperparameters: ['C', 'fit_intercept', 'max_iter', 'tol', 'early_stopping', 'validation_fraction', 'n_iter_no_change', 'shuffle', 'verbose', 'loss', 'n_jobs', 'random_state', 'warm_start', 'class_weight', 'average']

- **PassiveAggressiveRegressor**
  - Description: Passive Aggressive Regressor.
  - Key Hyperparameters: ['C', 'fit_intercept', 'max_iter', 'tol', 'early_stopping', 'validation_fraction', 'n_iter_no_change', 'shuffle', 'verbose', 'loss', 'epsilon', 'random_state', 'warm_start', 'average']

- **Perceptron**
  - Description: Linear perceptron classifier.
  - Key Hyperparameters: ['penalty', 'alpha', 'l1_ratio', 'fit_intercept', 'max_iter', 'tol', 'shuffle', 'verbose', 'eta0', 'n_jobs', 'random_state', 'early_stopping', 'validation_fraction', 'n_iter_no_change', 'class_weight', 'warm_start']

- **PoissonRegressor**
  - Description: Generalized Linear Model with a Poisson distribution.
  - Key Hyperparameters: ['alpha', 'fit_intercept', 'solver', 'max_iter', 'tol', 'warm_start', 'verbose']

- **QuantileRegressor**
  - Description: Linear regression model that predicts conditional quantiles.
  - Key Hyperparameters: ['quantile', 'alpha', 'fit_intercept', 'solver', 'solver_options']

- **RANSACRegressor**
  - Description: RANSAC (RANdom SAmple Consensus) algorithm.
  - Key Hyperparameters: ['estimator', 'min_samples', 'residual_threshold', 'is_data_valid', 'is_model_valid', 'max_trials', 'max_skips', 'stop_n_inliers', 'stop_score', 'stop_probability', 'loss', 'random_state']

- **Ridge**
  - Description: Linear least squares with l2 regularization.
  - Key Hyperparameters: ['alpha', 'fit_intercept', 'copy_X', 'max_iter', 'tol', 'solver', 'positive', 'random_state']

- **RidgeCV**
  - Description: Ridge regression with built-in cross-validation.
  - Key Hyperparameters: ['alphas', 'fit_intercept', 'scoring', 'cv', 'gcv_mode', 'store_cv_results', 'alpha_per_target']

- **RidgeClassifier**
  - Description: Classifier using Ridge regression.
  - Key Hyperparameters: ['alpha', 'fit_intercept', 'copy_X', 'max_iter', 'tol', 'class_weight', 'solver', 'positive', 'random_state']

- **RidgeClassifierCV**
  - Description: Ridge classifier with built-in cross-validation.
  - Key Hyperparameters: ['alphas', 'fit_intercept', 'scoring', 'cv', 'class_weight', 'store_cv_results']

- **SGDClassifier**
  - Description: Linear classifiers (SVM, logistic regression, etc.) with SGD training.
  - Key Hyperparameters: ['loss', 'penalty', 'alpha', 'l1_ratio', 'fit_intercept', 'max_iter', 'tol', 'shuffle', 'verbose', 'epsilon', 'n_jobs', 'random_state', 'learning_rate', 'eta0', 'power_t', 'early_stopping', 'validation_fraction', 'n_iter_no_change', 'class_weight', 'warm_start', 'average']

- **SGDOneClassSVM**
  - Description: Solves linear One-Class SVM using Stochastic Gradient Descent.
  - Key Hyperparameters: ['nu', 'fit_intercept', 'max_iter', 'tol', 'shuffle', 'verbose', 'random_state', 'learning_rate', 'eta0', 'power_t', 'warm_start', 'average']

- **SGDRegressor**
  - Description: Linear model fitted by minimizing a regularized empirical loss with SGD.
  - Key Hyperparameters: ['loss', 'penalty', 'alpha', 'l1_ratio', 'fit_intercept', 'max_iter', 'tol', 'shuffle', 'verbose', 'epsilon', 'random_state', 'learning_rate', 'eta0', 'power_t', 'early_stopping', 'validation_fraction', 'n_iter_no_change', 'warm_start', 'average']

- **TheilSenRegressor**
  - Description: Theil-Sen Estimator: robust multivariate regression model.
  - Key Hyperparameters: ['fit_intercept', 'max_subpopulation', 'n_subsamples', 'max_iter', 'tol', 'random_state', 'n_jobs', 'verbose']

- **TweedieRegressor**
  - Description: Generalized Linear Model with a Tweedie distribution.
  - Key Hyperparameters: ['power', 'alpha', 'fit_intercept', 'link', 'solver', 'max_iter', 'tol', 'warm_start', 'verbose']

# Ensemble Models in sklearn.ensemble

Found 19 classes in sklearn.ensemble:

- **AdaBoostClassifier**
  - Description: An AdaBoost classifier.
  - Key Hyperparameters: ['estimator', 'n_estimators', 'learning_rate', 'random_state']

- **AdaBoostRegressor**
  - Description: An AdaBoost regressor.
  - Key Hyperparameters: ['estimator', 'n_estimators', 'learning_rate', 'loss', 'random_state']

- **BaggingClassifier**
  - Description: A Bagging classifier.
  - Key Hyperparameters: ['estimator', 'n_estimators', 'max_samples', 'max_features', 'bootstrap', 'bootstrap_features', 'oob_score', 'warm_start', 'n_jobs', 'random_state', 'verbose']

- **BaggingRegressor**
  - Description: A Bagging regressor.
  - Key Hyperparameters: ['estimator', 'n_estimators', 'max_samples', 'max_features', 'bootstrap', 'bootstrap_features', 'oob_score', 'warm_start', 'n_jobs', 'random_state', 'verbose']

- **BaseEnsemble**
  - Description: Base class for all ensemble classes.
  - Key Hyperparameters: ['estimator', 'n_estimators', 'estimator_params']

- **ExtraTreesClassifier**
  - Description: 
  - Key Hyperparameters: ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples', 'monotonic_cst']

- **ExtraTreesRegressor**
  - Description: 
  - Key Hyperparameters: ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start', 'ccp_alpha', 'max_samples', 'monotonic_cst']

- **GradientBoostingClassifier**
  - Description: Gradient Boosting for classification.
  - Key Hyperparameters: ['loss', 'learning_rate', 'n_estimators', 'subsample', 'criterion', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_depth', 'min_impurity_decrease', 'init', 'random_state', 'max_features', 'verbose', 'max_leaf_nodes', 'warm_start', 'validation_fraction', 'n_iter_no_change', 'tol', 'ccp_alpha']

- **GradientBoostingRegressor**
  - Description: Gradient Boosting for regression.
  - Key Hyperparameters: ['loss', 'learning_rate', 'n_estimators', 'subsample', 'criterion', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_depth', 'min_impurity_decrease', 'init', 'random_state', 'max_features', 'alpha', 'verbose', 'max_leaf_nodes', 'warm_start', 'validation_fraction', 'n_iter_no_change', 'tol', 'ccp_alpha']

- **HistGradientBoostingClassifier**
  - Description: Histogram-based Gradient Boosting Classification Tree.
  - Key Hyperparameters: ['loss', 'learning_rate', 'max_iter', 'max_leaf_nodes', 'max_depth', 'min_samples_leaf', 'l2_regularization', 'max_features', 'max_bins', 'categorical_features', 'monotonic_cst', 'interaction_cst', 'warm_start', 'early_stopping', 'scoring', 'validation_fraction', 'n_iter_no_change', 'tol', 'verbose', 'random_state', 'class_weight']

- **HistGradientBoostingRegressor**
  - Description: Histogram-based Gradient Boosting Regression Tree.
  - Key Hyperparameters: ['loss', 'quantile', 'learning_rate', 'max_iter', 'max_leaf_nodes', 'max_depth', 'min_samples_leaf', 'l2_regularization', 'max_features', 'max_bins', 'categorical_features', 'monotonic_cst', 'interaction_cst', 'warm_start', 'early_stopping', 'scoring', 'validation_fraction', 'n_iter_no_change', 'tol', 'verbose', 'random_state']

- **IsolationForest**
  - Description: 
  - Key Hyperparameters: ['n_estimators', 'max_samples', 'contamination', 'max_features', 'bootstrap', 'n_jobs', 'random_state', 'verbose', 'warm_start']

- **RandomForestClassifier**
  - Description: 
  - Key Hyperparameters: ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start', 'class_weight', 'ccp_alpha', 'max_samples', 'monotonic_cst']

- **RandomForestRegressor**
  - Description: 
  - Key Hyperparameters: ['n_estimators', 'criterion', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_features', 'max_leaf_nodes', 'min_impurity_decrease', 'bootstrap', 'oob_score', 'n_jobs', 'random_state', 'verbose', 'warm_start', 'ccp_alpha', 'max_samples', 'monotonic_cst']

- **RandomTreesEmbedding**
  - Description: 
  - Key Hyperparameters: ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'min_weight_fraction_leaf', 'max_leaf_nodes', 'min_impurity_decrease', 'sparse_output', 'n_jobs', 'random_state', 'verbose', 'warm_start']

- **StackingClassifier**
  - Description: Stack of estimators with a final classifier.
  - Key Hyperparameters: ['estimators', 'final_estimator', 'cv', 'stack_method', 'n_jobs', 'passthrough', 'verbose']

- **StackingRegressor**
  - Description: Stack of estimators with a final regressor.
  - Key Hyperparameters: ['estimators', 'final_estimator', 'cv', 'n_jobs', 'passthrough', 'verbose']

- **VotingClassifier**
  - Description: Soft Voting/Majority Rule classifier for unfitted estimators.
  - Key Hyperparameters: ['estimators', 'voting', 'weights', 'n_jobs', 'flatten_transform', 'verbose']

- **VotingRegressor**
  - Description: Prediction voting regressor for unfitted estimators.
  - Key Hyperparameters: ['estimators', 'weights', 'n_jobs', 'verbose']

