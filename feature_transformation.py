# Databricks notebook source
# MAGIC %md Import libraries

# COMMAND ----------

from feature_engine.encoding import RareLabelEncoder
from sklearn.pipeline import (make_pipeline, Pipeline)
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import (StandardScaler, OneHotEncoder)
from sklearn.feature_selection import VarianceThreshold
from sklearn.base import BaseEstimator, TransformerMixin


# COMMAND ----------

# MAGIC %md Feature Transformer

# COMMAND ----------

class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, uk_areas_list):
        self.uk_areas_list = uk_areas_list
        self.encoded_feature_list = None
        pipeline_steps = make_pipeline(
            make_column_transformer(
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0.0), StandardScaler()),
                 ['days_since_incorp']),
                (make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'),
                               RareLabelEncoder(tol=0.0005, n_categories=10), OneHotEncoder(handle_unknown='ignore')),
                 ['company_type']),
                (make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'),
                               RareLabelEncoder(tol=0.1, n_categories=5), OneHotEncoder(handle_unknown='ignore')),
                 ['applicant_id_type']),
                (make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'),
                               RareLabelEncoder(tol=0.01, n_categories=5), OneHotEncoder(handle_unknown='ignore')),
                 ['applicant_id_country_issue']),
                (make_pipeline(SimpleImputer(missing_values=None, strategy='constant', fill_value='missing'),
                               RareLabelEncoder(tol=0.02, n_categories=5), OneHotEncoder(handle_unknown='ignore')),
                 ['applicant_postcode']),
                (make_pipeline(SimpleImputer(strategy='constant', fill_value='missing'),
                               RareLabelEncoder(tol=0.03, n_categories=5), OneHotEncoder(handle_unknown='ignore')),
                 ['section_description']),
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0), StandardScaler()), TRANSACTION_COLS),
                (make_pipeline(StandardScaler()), [x for x in NUMERICAL_COLS if
                                                   x != 'days_since_incorp' and x not in TRANSACTION_COLS and x not in IFRE_COLS]),
                remainder='passthrough'
            )
        )
        self.pipeline = Pipeline([("preprocessing", pipeline_steps)])

    def get_feature_list(self, df):
        """Method to get the list of features post input feature transformation

        Args:
          df (pd.DataFrame): input dataframe

        Returns:
          list: List of features post input feature transformation
        """

        pipelines = self.pipeline['preprocessing']['columntransformer'].transformers_
        feature_list = []
        for pipeline in pipelines:
            pipeline_steps = pipeline[1]

            if pipeline_steps != 'passthrough':
                pipeline_name = pipeline[2]
                steps_dict = pipeline_steps.named_steps
                steps_list = list(steps_dict.keys())

                if "onehotencoder" in steps_list:
                    orginial_features = list(steps_dict.get("onehotencoder").get_feature_names())
                    pipeline_name = pipeline_name[0]
                    new_features = [pipeline_name + "_" + s for s in orginial_features]
                    feature_list = feature_list + new_features
                else:
                    feature_list = feature_list + list(pipeline_name)
            else:
                left_over_indices = pipeline[2]
                feature_list = feature_list + list(df.columns[left_over_indices].values)
        return feature_list

    def extract_area_code(self, x):
        """Method to extract area code from a given postal code

        Args:
          x (string): string

        Returns:
          string: area code
        """

        if x:
            matching_areas = [y for y in self.uk_areas_list if x.startswith(y)]
            return max(matching_areas, key=len) if matching_areas else None
        return None

    def transform_applicant_postcode(self, applicant_postcodes):
        """Method to return series of area codes from given input series of postal codes

        Args:
          applicant_postcodes (pd.Series): series of postal codes

        Returns:
          pd.Series: series of area codes
        """

        return applicant_postcodes.astype(str).apply(self.extract_area_code)

    def transform_applicant_id_type(self, applicant_id_type):
        """Method to pre-process input series of applicant_id_type

        Args:
          applicant_id_type (pd.Series): series of applicant_id_type

        Returns:
          pd.Series: series of pre-processed applicant_id_type
        """

        transformed_data = pd.Series(
            np.where(pd.isnull(applicant_id_type), applicant_id_type, applicant_id_type.astype(str))).str.lower()
        return transformed_data.replace({"driving_license": "driving_licence"})

    def transform_company_type(self, company_type):
        """Method to pre-process input series of company_type

        Args:
          applicant_id_type (pd.Series): series of company_type

        Returns:
          pd.Series: series of pre-processed company_type
        """

        transformed_data = pd.Series(
            np.where(pd.isnull(company_type), company_type, company_type.astype(str))).str.lower()
        return transformed_data.replace({"private-limited-guarant-nsc": "private_limited_guarant_nsc"})

    def transform_applicant_id_country_issue(self, applicant_id_country_issue):
        """Method to pre-process input series of applicant_id_country_issue

        Args:
          applicant_id_type (pd.Series): series of applicant_id_country_issue

        Returns:
          pd.Series: series of pre-processed applicant_id_country_issue
        """

        return applicant_id_country_issue.replace({"GB": "GBR"})

    def preprocess_features(self, X):
        """Method to pre-process input features

        Args:
          applicant_id_type (pd.DataFrame): input dataframe

        Returns:
          pd.DataFrame: pre-processed dataframe of input features
        """

        X.applicant_postcode = self.transform_applicant_postcode(X.applicant_postcode)
        X.applicant_id_type = self.transform_applicant_id_type(X.applicant_id_type)
        X.company_type = self.transform_company_type(X.company_type)
        X.applicant_id_country_issue = self.transform_applicant_id_country_issue(X.applicant_id_country_issue)
        # preprocess_df['section_description'].replace({"": "missing"}, inplace=True)
        return X

    def fit(self, X, y=None):
        """Method to fit the feature transformer with input features

        Args:
          X (pd.DataFrame): input dataframe

        Returns:
          FeatureTransformer class object: it is fit on input data
        """

        X_fit = X.loc[:, FEATURES].copy()
        self.pipeline.fit(self.preprocess_features(X_fit))
        self.encoded_feature_list = self.get_feature_list(X_fit)
        return self

    def transform(self, X):
        """Method to pre-process any given input data using the trasnformer pipeline

        Args:
          X (pd.DataFrame): input dataframe

        Returns:
          pd.DataFrame: pre-processed dataframe of input features
        """

        X_transform = X.loc[:, FEATURES].copy()
        return pd.DataFrame(self.pipeline.transform(self.preprocess_features(X_transform)),
                            columns=self.encoded_feature_list)


# COMMAND ----------

# MAGIC %md Feature Selection

# COMMAND ----------

class CollinearFeatures(BaseEstimator, TransformerMixin):

    def __init__(self, thresh=10):
        """Init method for the collinear features to remove collinear features

		Args:
			thresh (list): Variance threshold to be considered in feature transformation pipeline

		Returns: None
		"""

        self.thresh = thresh
        self.vif_columns = None

    def fit(self, X, y=None):
        """"Method to fit the feature transformer with input features to remove collinear features

        Args:
          X (pd.DataFrame): input dataframe

        Returns:
          CollinearFeatures class object: it is fit on input data
        """

        self.vif_columns = []
        columns_list = list(X.columns)
        max_vif = 0
        while True:
            vifs = pd.Series(np.linalg.inv(X.loc[:, columns_list].corr().to_numpy()).diagonal(), index=columns_list,
                             name='VIF')
            vif_column, max_vif = vifs.sort_values(ascending=False).index[0], vifs.sort_values(ascending=False).values[
                0]
            if max_vif >= self.thresh:
                columns_list.remove(vif_column)
                self.vif_columns.append(vif_column)
            else:
                break
        return self

    def transform(self, X, y=None):
        """Method to pre-process any given input data using the trasnformer pipeline

        Args:
          X (pd.DataFrame): input dataframe

        Returns:
          pd.DataFrame: pre-processed dataframe of input features
        """

        return X.loc[:, ~X.columns.isin(self.vif_columns)]


class MinVariance(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Init method for the Min Variance class to filter input features based on variance

        Args:
            None

        Returns: None
        """

        self.pipeline = Pipeline([("feature_selection", make_pipeline(VarianceThreshold(threshold=0.1)))])

    def fit(self, X, y=None):
        """"Method to fit the feature transformer with input features  to removen features which fall below certain variance threshold

        Args:
          X (pd.DataFrame): input dataframe

        Returns:
          MinVariance class object: it is fit on input data
        """

        self.pipeline.fit(X)
        return self

    def transform(self, X):
        """Method to pre-process any given input data using the trasnformer pipeline

        Args:
          X (pd.DataFrame): input dataframe

        Returns:
          pd.DataFrame: pre-processed dataframe of input features
        """

        return X.loc[:, self.pipeline['feature_selection']['variancethreshold'].get_support()]


class FeatureSelection(BaseEstimator, TransformerMixin):

    def __init__(self):
        """Init method for the Feature Selection class to transform input features

		Args:
			None

		Returns: None
		"""

        self.collinear_features = CollinearFeatures()
        self.variance_threshold = MinVariance()

    def fit(self, X, y=None):
        """Method to fit the feature transformer with input features using above two feature transformation pipeline classes - CollinearFeatures & MinVariance

        Args:
          X (pd.DataFrame): input dataframe

        Returns:
          CollinearFeatures class object: it is fit on input data
        """

        X_var = self.variance_threshold.fit_transform(X)
        self.collinear_features.fit(X_var)
        return self

    def transform(self, X, y=None):
        """Method to pre-process any given input data using the trasnformer pipeline

        Args:
          X (pd.DataFrame): input dataframe

        Returns:
          pd.DataFrame: pre-processed dataframe of input features
        """

        return self.collinear_features.transform(self.variance_threshold.transform(X))