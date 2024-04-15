# Databricks notebook source
# MAGIC %md 
# MAGIC
# MAGIC <h1>Ongoing Risk - Model Definition</h1>
# MAGIC
# MAGIC This notebook holds the definitions of the model classes. This is imported in the training notebook.

# COMMAND ----------

# MAGIC %md ***Ongoing Risk Model Definitions***

# COMMAND ----------

class CalibratedRandomSurvivalForest(RandomSurvivalForest, BaseEstimator):
    def __init__(self, **rsf_params):
        super().__init__(**rsf_params)
        self.classes_ = [0, 1]

    def fit(self, X, y):
        X_rsf, y_rsf = get_x_y(pd.concat([X, y], axis=1), attr_labels=[EVENT_COL, DURATION_COL], pos_label=1,
                               survival=True)
        return super().fit(X_rsf, y_rsf)

    def decision_function(self, X):
        return self.predict_survival_function(X, return_array=True)[:, -1]

    def score(self, X, y):
        X_rsf, y_rsf = get_x_y(pd.concat([X, y], axis=1), attr_labels=[EVENT_COL, DURATION_COL], pos_label=1,
                               survival=True)
        return super().score(X_rsf, y_rsf)


# COMMAND ----------

class OngoingRisk:

    def __init__(
            self,
            uk_areas_list,
            hyper_param_grid=None,
            cv_splits=5,
            cv_repeats=3,
            test_size=0.2,
            random_seed=123
    ):
        """Init method for the ongoing risk class

        Args:
            transformer_pipeline_preprocess (sklearn.pipeline.Pipeline): Optional, Sklearn Pipeline used for data transformation,
            param_grid (dict): Optional, Parameter grid to be used for hyper parameter tuning
            feature_list (list): the feature columns obtained after pre-processing

        Returns: None
        """
        self.random_seed = random_seed
        self.cv_splits = cv_splits
        self.cv_repeats = cv_repeats
        self.test_size = test_size
        self.feature_transformer = FeatureTransformer(uk_areas_list)
        self.feature_selection = FeatureSelection()
        self.hyper_param_grid = hyper_param_grid
        self.uncalibrated_model = None
        self.calibrated_model = None
        self.best_params = {
            'n_estimators': 250,
            'max_depth': 25,
            'min_samples_split': 20,
            'min_samples_leaf': 15,
            'bootstrap': True,
            'n_jobs': -1,
            'random_state': self.random_seed,
            'max_samples': 10000
        }
        self.metrics = {}

    def _discretize_target_variable(self, y):
        y = y.reset_index(drop=True)
        y[DURATION_COL] = (y[DURATION_COL] / 7).round()
        return y

    def _cross_val_scores(self, X, y, best_params):
        import statistics
        rskf = RepeatedStratifiedKFold(n_splits=self.cv_splits, n_repeats=self.cv_repeats,
                                       random_state=self.random_seed)
        train_cindex, test_cindex = [], []
        for train_index, test_index in rskf.split(X, y.loc[:, EVENT_COL]):
            X_train_cv, X_test_cv = X.iloc[train_index, :], X.iloc[test_index, :]
            y_train_cv, y_test_cv = y.iloc[train_index, :], y.iloc[test_index, :]
            X_train_cv = self.feature_selection.fit_transform(self.feature_transformer.fit_transform(X_train_cv))
            X_test_cv = self.feature_selection.transform(self.feature_transformer.transform(X_test_cv))
            y_train_cv, y_test_cv = self._discretize_target_variable(y_train_cv), self._discretize_target_variable(
                y_test_cv)
            model = CalibratedRandomSurvivalForest(**best_params)
            model.fit(X_train_cv, y_train_cv)
            train_cindex.append(model.score(X_train_cv, y_train_cv))
            test_cindex.append(model.score(X_test_cv, y_test_cv))
        return statistics.mean(train_cindex), statistics.stdev(train_cindex), statistics.mean(
            test_cindex), statistics.stdev(test_cindex)

    @staticmethod
    def risk_threshold_func(x: float, low_risk_threshold=0.005, high_risk_threshold=0.01) -> str:
        return 'Low' if 1 - x <= low_risk_threshold else ('High' if 1 - x >= high_risk_threshold else 'Medium')

    @staticmethod
    def get_risk_stats(risk_ratings, events, low_risk_threshold=0.005, high_risk_threshold=0.01):
        temp_df = pd.DataFrame()
        temp_df['risk_level'] = pd.Series(risk_ratings).apply(OngoingRisk.risk_threshold_func,
                                                              args=(low_risk_threshold, high_risk_threshold))
        temp_df['event'] = events
        risk_level_pct = (temp_df['risk_level'].value_counts(normalize=True) * 100.0).round(2).to_dict()
        risk_level_fincrime = (temp_df.groupby('risk_level')['event'].mean().sort_values(ascending=True) * 100).round(
            3).to_dict()
        return risk_level_pct, risk_level_fincrime

    def fit(self, input_df: pd.DataFrame):
        """Fit method to train the Ongoing Risk Model and fit the transformer pipeline

        Args:
        input_df (pd.DataFrame): Input data used to fit the model
        test_size (float): Optional, Ratio of input data used for testing - default value: 0.25
        random_state (int): Optional, Random seed used for shuffling data - default value: 123

        Returns: None
        """
        train_df = input_df.copy().loc[:, FEATURES + EVENT_COLS]

        # Split dataset into features and target labels
        print('Preparing Dataset')
        X, y = train_df.loc[:, list(FEATURES)], train_df.loc[:, list(EVENT_COLS)]
        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size,
        #                                                     random_state=self.random_seed, stratify=y[EVENT_COL])

        # # Cross Validation
        # cross_val_results = self._cross_val_scores(X_train, y_train, self.best_params)
        # self.metrics['crossval_train_cindex_avg'], self.metrics['crossval_train_cindex_std'] = round(
        #     cross_val_results[0], 3), round(cross_val_results[1], 3)
        # self.metrics['crossval_test_cindex_avg'], self.metrics['crossval_test_cindex_std'] = round(cross_val_results[2],
        #                                                                                            3), round(
        #     cross_val_results[3], 3)

        # # Fit and transform of pipelines on training data
        # X_train = self.feature_selection.fit_transform(self.feature_transformer.fit_transform(X_train))
        # X_test = self.feature_selection.transform(self.feature_transformer.transform(X_test))
        # y_train, y_test = self._discretize_target_variable(y_train), self._discretize_target_variable(y_test)

        # # Test Dataset Metrics
        # model = CalibratedRandomSurvivalForest(**self.best_params)
        # model.fit(X_train, y_train)
        # self.metrics['val_cindex'] = round(model.score(X_test, y_test), 3)
        # ## Add other metrics like brier score ##

        # Final fitting using the entire training data
        print('Train Test Split')
        X_train, X_calibration, y_train, y_calibration = train_test_split(X, y, test_size=self.test_size,
                                                                          random_state=self.random_seed,
                                                                          stratify=y[EVENT_COL])
        X_train = self.feature_selection.fit_transform(self.feature_transformer.fit_transform(X_train))
        print('Feature Selected and transformed')
        X_calibration = self.feature_selection.transform(self.feature_transformer.transform(X_calibration))
        print('Calibrated and data transformed')
        y_train, y_calibration = self._discretize_target_variable(y_train), self._discretize_target_variable(
            y_calibration)
        print('Labels calibrated')
        self.uncalibrated_model = CalibratedRandomSurvivalForest(**self.best_params)
        self.uncalibrated_model.fit(X_train, y_train)
        print('Model Fitted')
        self.calibrated_model = CalibratedClassifierCV(self.uncalibrated_model, method='sigmoid', cv='prefit')
        self.calibrated_model.fit(X_calibration, y_calibration[EVENT_COL])
        print('Model Calibrated')

        return self

    def predict(self, input_df: pd.DataFrame, low_risk_threshold=0.005, high_risk_threshold=0.01) -> pd.DataFrame:
        """Method to make predictions using the fitted ongoing risk model

        Args:
            input_df (pd.DataFrame): Input data for predictions

        Returns:
            pd.DataFrame: Predicted risk ratings of the ongoing risk model
        """
        X = input_df.loc[:, FEATURES + ('company_id',)].copy()
        risk_rating_df = pd.DataFrame()
        risk_rating_df['company_id'] = X.company_id
        risk_rating_df['survival_probability'] = self.calibrated_model.predict_proba(
            self.feature_selection.transform(self.feature_transformer.transform(X.loc[:, FEATURES])))[:, 0]
        risk_rating_df['risk_rating'] = risk_rating_df.survival_probability.apply(OngoingRisk.risk_threshold_func,
                                                                                  args=(
                                                                                      low_risk_threshold,
                                                                                      high_risk_threshold))
        return risk_rating_df


# COMMAND ----------

# MAGIC %md MLFlow Wrapper

# COMMAND ----------

class MLFlowWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, wrapped_class):
        self.model = wrapped_class

    def fit(self, model_input):
        return self.model.fit(model_input)

    def predict(self, context, model_input, low_risk_threshold=0.005, high_risk_threshold=0.01):
        return self.model.predict(model_input, low_risk_threshold, high_risk_threshold)