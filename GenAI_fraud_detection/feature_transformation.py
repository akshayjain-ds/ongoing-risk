# Databricks notebook source
class FeatureTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        areas = [
            'AB', 'AL', 'B', 'BA', 'BB', 'BD', 'BH', 'BL', 'BN', 'BR', 'BS', 'BT', 'CA', 'CB', 'CF', 'CH', 'CM', 'CO',
            'CR', 'CT', 'CV', 'CW', 'DA', 'DD', 'DE', 'DG',
            'DH', 'DL', 'DN', 'DT', 'DY', 'E', 'EC', 'EH', 'EN', 'EX', 'FK', 'FY', 'G', 'GL', 'GU', 'GY', 'HA', 'HD',
            'HG', 'HP', 'HR', 'HS', 'HU', 'HX', 'IG', 'IM',
            'IP', 'IV', 'JE', 'KA', 'KT', 'KW', 'KY', 'L', 'LA', 'LD', 'LE', 'LL', 'LN', 'LS', 'LU', 'M', 'ME', 'MK',
            'ML', 'N', 'NE', 'NG', 'NN', 'NP', 'NR', 'NW',
            'OL', 'OX', 'PA', 'PE', 'PH', 'PL', 'PO', 'PR', 'RG', 'RH', 'RM', 'S', 'SA', 'SE', 'SG', 'SK', 'SL', 'SM',
            'SN', 'SO', 'SP', 'SR', 'SS', 'ST', 'SW', 'SY',
            'TA', 'TD', 'TF', 'TN', 'TQ', 'TR', 'TS', 'TW', 'UB', 'W', 'WA', 'WC', 'WD', 'WF', 'WN', 'WR', 'WS', 'WV',
            'YO', 'ZE'
        ]

        def _construct_tree(ls):
            if not ls:
                return {'': None}
            else:
                return {
                    k: _construct_tree([v[1:] for v in values]) if k else None
                    for k, values in itertools.groupby(sorted(ls), key=lambda x: x[:1])
                }

        self.areas_tree = _construct_tree(areas)
        self.input_features = INPUT_FEATURES_PRE_TRANSFORMATION  # Base feature(renamed) list returned by tecton
        self.num_feats = NUMERICAL_COLS
        self.cat_feats = CATEGORICAL_COLS
        self.epsilon = 1e-8
        self.model_features = MODEL_INPUT_FEATURES  # Final feature list returned to the model (after filtering, aggregation etc.)
        self._exempted_accounts_sort_codes = (
            '12001039083210', '12001020083210', '11963155083200', '41320424406425', '00200476300000', '00331708300000',
            '12000962083200'
        )
        self._credit_line_event_expiry_seconds = 30
        self._whitelisting_expiry = {'LOW': 90, 'MEDIUM': 60, 'HIGH': 30}

        imputing_pipeline_steps = make_pipeline(
            make_column_transformer(
                # Default registration status assumed to be sole trader to assume higher risk, less than 1% of missing values
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)), ['is_registered_company']),
                # Median age at onboarding assumed in case of missing values, less than 1% of missing values
                (make_pipeline(SimpleImputer(strategy='median')), ['member_age_at_onboarding']),
                # Treating missing values as a special value, less than 1% of missing values
                (make_pipeline(SimpleImputer(strategy='constant', fill_value='missing')),
                 ['member_area_code', 'member_id_country_issue', 'member_id_type']),
                # Static/special value assumed in case of missing values, 10-20% missing values due to unregistered companies
                (make_pipeline(SimpleImputer(strategy='constant', fill_value='missing')),
                 ['registered_company_type', 'registered_company_industry_classification']),
                # Net balance indicator - imputing with -1 to group no activity along with high deposit activity (low risk)
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=-1)), ['net_balance_indicator']),

                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)),
                 ['ratio_of_rolling_sum_of_wtd_2h_to_max_wtd_60d',
                  'ratio_of_pmts_value_to_payee_of_overall_pmts_60d',
                  'ratio_of_pmts_value_to_payee_of_overall_pmts_48h',
                  'ratio_of_pmts_value_to_payee_of_overall_pmts_2h'
                ]),

                # Cashflow Value Indicator, Cashflow Volume Indicator, Deposit Withdrawal Ratio - applying Imputation
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)),
                 ['cashflow_value_indicator', 'cashflow_volume_indicator', 'deposit_withdrawal_ratio']),
                # Most frequent value assumed in case of missing values, missing value only possible if there are no transactions on account previously
                (make_pipeline(SimpleImputer(strategy='constant', fill_value='fast_payment')),
                 ['payment_channel_most_used_for_withdrawals', 'payment_channel_most_used_for_deposits']),
                # Number of high value transactions - imputing with zero to indicate no high risk activity
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)),
                 ['number_of_high_value_transactions_1y', 'number_of_withdrawals_over_threshold_60d']),
                # If no withdrawals in last 60 days, set 60 as days since last withdrawal
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=60)),
                 ['number_of_days_since_last_withdrawal_60d']),
                # If no withdrawals in last 15 mins, imputing with 0
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)),
                 ['is_previous_payment_in_last_15mins']),
                # Null value only possible if there no beneficiaries and benefactors on account, imputing with zero to go with low risk i.e. no beneficiaries, only benefactors
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)),
                 ['fastpmt_beneficiaries_benefactors_ratio_1y']),
                # Zero imputation and scaling for all count fields
                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)),
                 ['number_of_tester_payments_1y', 'number_of_latehours_cash_transactions_1y',
                  'number_of_card_acceptors_1y',
                  'number_of_direct_debit_beneficiaries_1y', 'number_of_hmrc_transactions_1y',
                  'number_of_xero_transactions_1y',
                  'number_of_atms_used_1y', 'number_of_cards_used_1y', 'number_of_tester_payments_24h',
                  'percentage_of_round_transactions_1y',
                  'outgoing_payments_sum_account_60d', 'incoming_payments_sum_account_60d',
                  "incoming_payment_density_over_the_past_2h_vs_60days_ratio"]),
                # Imputing with median to group no activity or no withdrawals along with high deposit activity (low risk)                
                (make_pipeline(SimpleImputer(strategy='median')), ['deposit_withdrawal_frequency_ratio_1y']),

                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)),
                 ['ratio_pmts_outside_business_60d', 'incoming_payment_requested_avg_pmt_value_with_payee_ratio_60d',
                  'outgoing_payment_requested_avg_pmt_value_with_payee_ratio_60d',
                  'mean_of_payments_to_new_payees_60d', 'sum_of_payments_to_new_payees_48h']),


                (make_pipeline(SimpleImputer(strategy='constant', fill_value=0)),
                 ['rolling_sum_of_pmts_over_threshold_to_new_payee_48h',
                  'rolling_sum_of_pmts_over_threshold_to_new_payee_60d',
                  'incoming_payment_density_over_the_past_2h_vs_60days',
                  'ratio_of_rolling_sum_of_deposits_2h_to_max_deposit_60d',
                  'count_above_thr_deposite_from_counter_party_1000_60d', 'payment_transfer_rate_on_cop',
                  'requested_payment_to_avg_deposit_60d_ratio', 'is_requested_payment_gt_60day_max_wtd']),

                # Below columns go through with no imputation, standardization, encoding => RULE_FEATURES
                remainder='passthrough'
            )
        )

        """Pipeline for Categorical feature transformation """
        cat_encode_pipeline_steps = make_pipeline(
            make_column_transformer(
                (make_pipeline(TargetEncoder()), self.cat_feats),
                remainder='passthrough'
            )
        )

        self.imputing_pipeline = Pipeline([("imputation", imputing_pipeline_steps)])
        self.cat_encode_pipeline = Pipeline([("cat-encode", cat_encode_pipeline_steps)])

    def get_feature_list(self, df, pipelines):
        """Method to get the list of features post input feature transformation"""
        # pipelines = self.pipeline['preprocessing']['columntransformer'].transformers_
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

    def extract_area_code(self, code):
        """
        Method to extract area code from a given postal code
          if x:
              matching_areas = [y for y in self.uk_areas_list if x.startswith(y)]
              return max(matching_areas, key=len) if matching_areas else None
          return None
        """
        if not code:
            return None
        current_root = self.areas_tree
        for i, c in enumerate(code):
            next_root = current_root.get(c, None)
            if next_root is None:
                return code[:i] if '' in current_root else None
            current_root = next_root
        return code if '' in current_root else None

    @timer
    def transform_member_area_code(self, applicant_postcodes):
        """Method to return series of area codes from given input series of postal codes"""
        return applicant_postcodes.astype(str).apply(self.extract_area_code)

    @timer
    def transform_member_id_type(self, applicant_id_type):
        """Method to pre-process input series of applicant_id_type"""
        # Cast applicant_id_type to string when not null and then replace values with typo
        transformed_data = pd.Series(
            np.where(pd.isnull(applicant_id_type), applicant_id_type, applicant_id_type.astype(str))).str.lower()
        return transformed_data.replace({"driving_license": "driving_licence"})

    @timer
    def transform_registered_company_type(self, company_type):
        """Method to pre-process input series of company_type"""
        # Cast company_type to string when not null and then replace values with other variant
        transformed_data = pd.Series(
            np.where(pd.isnull(company_type), company_type, company_type.astype(str))).str.lower()
        return transformed_data.replace({"private-limited-guarant-nsc": "private_limited_guarant_nsc"})

    @timer
    def transform_member_id_country_issue(self, applicant_id_country_issue):
        """Method to pre-process input series of applicant_id_country_issue"""
        return np.where(applicant_id_country_issue == 'GB', 'GBR', applicant_id_country_issue)

    @timer
    def transform_net_balance_indicator(self, X):
        """
        Method to calculate netbalance indicator which provides the level of balance remaining in account
        - Formula: (withdrawals + requested_payment - deposits) / (withdrawals + requested_payment + deposits)
        - Calculate net_balance_indicator for rolling windows of 2, 4, 8, 12, 24, 48 hours and the maximum value among all windows is assumed as the final net_balance_indicator
        - Provides simplified balance indicator, but doesn’t capture volume or value information
        - Value ranges between -1 and 1
        """

        def calculate_net_balance_indicator(deposits, withdrawals, requested_payment):
            return (np.absolute(withdrawals) + requested_payment - np.absolute(deposits)) / (
                        np.absolute(withdrawals) + requested_payment + np.absolute(deposits))

        net_balance_values = np.empty(X.shape[0])
        net_balance_values.fill(np.NINF)
        for x in [2, 4, 8, 12, 24, 48]:
            net_balance_values = np.maximum(
                net_balance_values,
                calculate_net_balance_indicator(
                    X[f"rolling_sum_of_deposits_on_account_{x}h"].values,
                    X[f"rolling_sum_of_withdrawals_on_account_{x}h"].values,
                    X.requested_payment_value
                )
            )
        # assert np.NINF not in net_balance_values
        assert np.isinf(net_balance_values).any() == False
        return net_balance_values

    @timer
    def transform_cashflow_value_indicator(self, X):
        """
        Method to calculate total value of cashflow on account
        - Formula: total_value_of_deposits + requested_payment  + total_value_of_withdrawals
        - Calculate cashflow_value_indicator for rolling windows of 2, 4, 8, 12, 24, 48 hours and the maximum value among all windows is assumed as the final cashflow_value_indicator
        - Provides the total value of deposits and withdrawals on account
        - Value unbounded with no fixed range, with a minimum value of requested_payment
        """
        cashflow_values = np.empty(X.shape[0])
        cashflow_values.fill(np.NINF)
        for x in [2, 4, 8, 12, 24, 48]:
            cashflow_values = np.maximum(
                cashflow_values,
                np.add(
                    X[f"rolling_sum_of_deposits_on_account_{x}h"].values,
                    np.absolute(X[f"rolling_sum_of_withdrawals_on_account_{x}h"].values)
                ) + X.requested_payment_value
            )
        # assert np.NINF not in cashflow_values
        assert np.isinf(cashflow_values).any() == False
        return cashflow_values

    @timer
    def transform_deposit_withdrawal_ratio(self, X):
        """
        Method to calculate ratio of rolling sum of withdrawals to deposit transactions on account
        - Formula: (#withdrawals + 1 - #deposits) / (#withdrawals + 1 + #deposits)
        - Calculate deposit_withdrawal_ratio for rolling windows of 2, 4, 8, 12, 24, 48 hours and the maximum value among all windows is assumed as the final deposit_withdrawal_ratio
        - Provides a comparison of rolling sum of deposits to withdrawals
        - Value ranges between -1 and 1
        """

        def calculate_deposit_withdrawal_ratio(sum_of_withdrawals_on_account, sum_of_deposits_on_account):
            return (sum_of_withdrawals_on_account + 1 - sum_of_deposits_on_account) / (
                        sum_of_withdrawals_on_account + 1 + sum_of_deposits_on_account)

        deposit_withdrawal_ratio_values = np.empty(X.shape[0])
        deposit_withdrawal_ratio_values.fill(np.NINF)
        for x in [2, 4, 8, 12, 24, 48]:
            deposit_withdrawal_ratio_values = np.maximum(
                deposit_withdrawal_ratio_values,
                calculate_deposit_withdrawal_ratio(
                    X[f"rolling_sum_of_outgoing_payments_on_account_{x}h"].values,
                    X[f"rolling_sum_of_incoming_payments_on_account_{x}h"].values
                )
            )
        # assert np.NINF not in deposit_withdrawal_ratio_values
        assert np.isinf(deposit_withdrawal_ratio_values).any() == False
        return deposit_withdrawal_ratio_values

    @timer
    def transform_cashflow_volume_indicator(self, X):
        """
        Method to calculate volume of transactions on account
        - Formula: total_value_of_deposits+ requested_payment  + total_value_of_withdrawals
        - Calculate cashflow_value_indicator for rolling windows of 2, 4, 8, 12, 24, 48 hours and the maximum value among all windows is assumed as the final cashflow_value_indicator
        - Provides the total value of deposits and withdrawals on account
        - Value unbounded with no fixed range, with a minimum value of requested_payment
        """
        cashflow_volumes = np.empty(X.shape[0])
        cashflow_volumes.fill(np.NINF)
        for x in [2, 4, 8, 12, 24, 48]:
            cashflow_volumes = np.maximum(
                cashflow_volumes,
                np.add(X[f"rolling_sum_of_outgoing_payments_on_account_{x}h"].values,
                       X[f"rolling_sum_of_incoming_payments_on_account_{x}h"].values) + 1
            )
        # assert np.NINF not in cashflow_volumes
        assert np.isinf(cashflow_volumes).any() == False
        return cashflow_volumes

    @timer
    def calculate_payment_requested_ratio(self, requested_payment_value, feature_column):
        """Method to calculate ratio of a feature and payment requested on account"""
        return np.divide(requested_payment_value, np.add(requested_payment_value, feature_column))

    @timer
    def transform_payment_channel_most_used_for_withdrawals(self, X):
        """Method to calculate payment channel most used for withdrawals on account"""
        col_mapping = {
            'percentage_cardpayments_of_all_withdrawals_1y': 'card_payment',
            'percentage_fastpayments_of_all_withdrawals_1y': 'fast_payment',
            'percentage_directdebits_of_all_withdrawals_1y': 'direct_debit',
            'percentage_cardwithdrawals_of_all_withdrawals_1y': 'card_withdrawal',
            'percentage_outpayments_of_all_withdrawals_1y': 'out_payment',
        }
        return X.loc[:, list(col_mapping.keys())].idxmax(axis="columns").replace(col_mapping)

    @timer
    def transform_payment_channel_most_used_for_deposits(self, X):
        """Method to calculate payment channel most used for deposits on account"""
        col_mapping = {
            'percentage_fastpayments_of_all_deposits_1y': 'fast_payment',
            'percentage_cashdeposits_of_all_deposits_1y': 'cash_deposit',
            'percentage_inpayments_of_all_deposits_1y': 'in_payment',
        }
        return X.loc[:, list(col_mapping.keys())].idxmax(axis="columns").replace(col_mapping)

    @timer
    def transform_number_of_high_value_transactions_on_account_1y(self, X):
        """Method to calculate number of high value transactions on account"""
        return X.eval("number_of_high_card_payments_on_account_1y + number_of_high_card_withdrawals_on_account_1y + number_of_high_outgoing_fastpayments_on_account_1y + \
                       number_of_high_outpayments_on_account_1y + number_of_high_directdebits_on_account_1y + number_of_high_incoming_fastpayments_on_account_1y + \
                       number_of_maximum_cash_deposits_on_account_1y + number_of_high_inpayments_on_account_1y")

    @timer
    def transform_number_of_days_since_last_withdrawal_60d(self, rule_feature_request_timestamp,
                                                           last_payment_requested_received_at_60d):
        """Method to calculate number of days since last fast payment on account"""
        temp_series = ((rule_feature_request_timestamp.astype(float) - last_payment_requested_received_at_60d.astype(
            float)) / (60 * 60 * 24)).round()
        return np.clip(temp_series.values, 0,
                       60)  # Clipping max value to 60 to ensure cases where there is no payment request in last 60 days

    @timer
    def transform_is_previous_fastpayment_in_last_15mins(self, rule_feature_request_timestamp,
                                                         last_payment_requested_received_at_60d):
        """Method to check if last fast payment on account in last 15 mins prior to payment request"""
        return ((((rule_feature_request_timestamp.astype(float) - last_payment_requested_received_at_60d.astype(
            float)) / 60)) < 15).round().astype(int)

    @timer
    def transform_fastpmt_beneficiaries_benefactors_ratio_1y(self, number_of_fastpayment_beneficiaries_1y,
                                                             number_of_fastpayment_benefactors_1y):
        """Method to calculate ratio of beneficiaries to benefactors on account"""
        return number_of_fastpayment_beneficiaries_1y / (
                    number_of_fastpayment_beneficiaries_1y + number_of_fastpayment_benefactors_1y)
        
    @timer
    def transform_time_passed_since_confirmation_of_payee(self, rule_feature_request_timestamp,
                                                        timestamp_ntm_recipient_confirmed):
        timestamp_ntm_recipient_confirmed[timestamp_ntm_recipient_confirmed==None]=0
        time_passed_since_confirmation_of_payee = np.where(timestamp_ntm_recipient_confirmed,(rule_feature_request_timestamp - timestamp_ntm_recipient_confirmed), 365 * 24 * 60 * 60)
        time_passed_since_confirmation_of_payee = pd.Series(time_passed_since_confirmation_of_payee).fillna(365 * 24 * 60 * 60)
        return time_passed_since_confirmation_of_payee.astype(int).values

    @timer
    def calculate_pmt_initiated_hour_48h_range(self, X):
        """This method calculates the time difference between the maximum hour with min hour of pmt initiation."""
        return X["payment_initiated_hour_max_48h"] - X["payment_initiated_hour_min_48h"]

    @timer
    def calculate_ratio_pmts_outside_business_60days(self, X):
        """
          This method calculates the ratio of the number of payments made between 9am to 10pm in the last 60 days
          to the sum of outgoing payments from the account during the same period.
        """
        return X["number_of_payments_outside_9am_to_10pm_60d"] / (X["outgoing_payments_sum_account_60d"] + 1)

    @timer
    def calculate_incoming_pmt_requested_avg_pmt_value_with_payee_ratio_60days(self, X):
        """
          This method calculates the ratio of the requested payment value to the rolling mean of deposits
          from the counterparty in the last 60 days.
        """
        return (X['requested_payment_value'] / (X['rolling_mean_of_deposits_from_counter_party_60d'] + 1))

    @timer
    def calculate_outgoing_pmt_requested_avg_pmt_value_with_payee_ratio_60days(self, X):
        """
          This method calculates the ratio of the requested payment value to the rolling mean of withdrawals
          to the counterparty in the last 60 days.
        """
        return (X['requested_payment_value'] / (X['rolling_mean_of_withdrawals_to_counter_party_60d'] + 1))

    @timer
    def calculate_pmt_to_new_payees(self, withdrawals_to_cp, time_passed_since_confirmation_of_payee, hour=1):
        """ This method calculates the sum of payments over a threshold value to multiple new payees."""
        return withdrawals_to_cp * (1 * time_passed_since_confirmation_of_payee < (hour * 60 * 60))

    @timer
    def calculate_ratio_of_pmts_to_payee_of_overall_pmts(self, rolling_sum_of_withdrawals_to_cp,
                                                         outgoing_payments_sum_account):
        """
          This method calculates the ratio of the rolling sum of withdrawals to the counterparty
          to the sum of outgoing payments from the account in the last 60 days
        """
        return rolling_sum_of_withdrawals_to_cp / (outgoing_payments_sum_account + 1)

    # New function
    def calculate_days_since_payee_added_as_beneficiary(self, time_passed_since_confirmation_of_payee):
        return [(x / (24 * 60 * 60)) if x <= 365 * 24 * 60 * 60 else 365 for x in
                time_passed_since_confirmation_of_payee]

    def calculate_hours_since_payee_added_as_beneficiary(self, time_passed_since_confirmation_of_payee):
        return [(x / (60 * 60)) if x <= 48 * 60 * 60 else 48 for x in time_passed_since_confirmation_of_payee]

    @timer
    def composite_payment_features(self, X):
        """ Method to create composite feature for finding dormant account and counter party level features """

        X["incoming_payment_density_over_the_past_2h_vs_60days"] = (
                    (X["incoming_payments_sum_account_60d"] - X["rolling_sum_of_incoming_payments_on_account_2h"]) * X[
                "rolling_sum_of_incoming_payments_on_account_2h"])

        X["incoming_payment_density_over_the_past_2h_vs_60days_ratio"] = (X["incoming_payments_sum_account_60d"] - X[
            "rolling_sum_of_incoming_payments_on_account_2h"]) / (X["incoming_payments_sum_account_60d"] + 1)

        X["ratio_of_rolling_sum_of_wtd_2h_to_max_wtd_60d"] = (X["max_withdrawal_value_60d"] - X[
            "rolling_sum_of_withdrawals_on_account_2h"]) / (X["max_withdrawal_value_60d"] + 1)
        X["ratio_of_rolling_sum_of_deposits_2h_to_max_deposit_60d"] = (X["max_deposit_value_60d"] - X[
            "rolling_sum_of_deposits_on_account_2h"]) / (X["max_deposit_value_60d"] + 1)

        X["count_above_thr_deposite_from_counter_party_1000_60d"] = X[
            "rolling_sum_of_deposits_value_to_counter_party_abv_thresh_1000_60d"]  # * X['member_tenure_in_days']

        X['avg_no_payment_to_from_counter_party'] = ((X["rolling_count_of_payments_from_counter_party_48h"] + X[
            "rolling_count_of_payments_to_counter_party_48h"]) / 2)

        X['payment_transfer_rate_on_cop'] = np.log(
            X.requested_payment_value / (X["time_passed_since_confirmation_of_payee"] + 1))

        X["requested_payment_to_avg_deposit_60d_ratio"] = X.requested_payment_value / (
                    X['average_deposit_value_60d'] + 1)

        X["is_requested_payment_gt_60day_max_wtd"] = 1 * (X.requested_payment_value > X["max_withdrawal_value_60d"])

        return X

    @timer
    def transform_is_recipient_account_exempted(self, X):
        """Method to check if recipient account is exempted"""
        codes = set(self._exempted_accounts_sort_codes)
        return ((X.destination_account_number.astype(str).fillna('')) + (
            X.destination_account_sort_code.astype(str).fillna(''))).apply(
            lambda x: any(set(x).intersection(codes))) * 1

    def _is_clearlisted(self, input_data):
        request_timestamp: int = input_data.get('rule_feature_request_timestamp')  # request.timestamp from JBE
        risk_band: str = input_data.get('rule_feature_kyc_risk_band')  # HIGH/MEDIUM/LOW - request.riskRating
        # values from the last application/vnd.tide.company-(un)whitelisted.v1 events
        is_clear_listed: bool = input_data.get(
            'rule_feature_is_clearlisted')  # True/False - based on the type of the event, False on no event
        last_whitelisted_at: int = input_data.get(
            'rule_feature_clearlisting_last_updated_at')  # epoch timestamp of the last (un)whitelisted event, 0 on no data
        event_ignore_expiration: bool = input_data.get(
            'rule_feature_event_ignore_expiration')  # True/False, default False, False for events prior to `ignoreExpiration` attribute 2021-06-10
        # timestamp for permanent whitelisting status from static table (legacy Permanent whitelisting) (data materialized from TMv1 permanently whitelisted companies prior
        #  to the introduction of `ignoreExpiration` in the event)
        tm_ignore_expiration_at: int = input_data.get(
            'rule_feature_permenant_clearlisting_last_whitelisted_at')  # 0 or epoch value of 2020-07-28
        # value of 2020-07-28 is the introduction of perma-whitelist in TMv1 (ignore expiration was introduced 2021-06-10, so we take from 2021-06-11)
        IGNORE_EXPIRATION_FIELD_INTRODUCED: int = 1623358800  # epoch
        # END INPUT

        clear_listing_status = False  # we list only the clearlisting statuses
        # If company is part of legacy Permanent whitelisting
        # THEN clearlisted
        if (request_timestamp > tm_ignore_expiration_at > 0): 
            clear_listing_status = True
        return clear_listing_status
    
    def _is_source_dest_clearlisted(self, input_data):
        request_timestamp: int = input_data.get('rule_feature_request_timestamp')  # request.timestamp from JBE
        is_source_dest_clearlisted: bool = input_data.get('rule_feature_is_source_dest_clearlisted', False)  # True/False - based on the type of the event, False on no event
        expiry_ts: int = input_data.get('rule_feature_source_dest_expiry_ts')
        source_dest_clearlisted_amount: float = input_data.get('rule_feature_source_dest_amount')
        requested_payment_value: float = input_data.get('requested_payment_value')

        clear_listing_status = False
        if (is_source_dest_clearlisted) and ((request_timestamp < expiry_ts) and (requested_payment_value <= source_dest_clearlisted_amount)): 
            clear_listing_status = True
        return clear_listing_status
    
    @timer
    def transform_is_clearlisted(self, X):
        """Method to check if sender account is clearlisted"""
        return X.apply(self._is_clearlisted, axis=1).astype(int)

    @timer
    def transform_is_source_dest_clearlisted(self, X):
        """Method to check if sender and destination accounts pair are clearlisted"""
        return X.apply(self._is_source_dest_clearlisted, axis=1).astype(int)

    @timer
    def transform_is_matching_credit_event(self, X):
        """Method to check if payment request is matching credit event"""
        credit_line_status_values = np.zeros(X.shape[0])
        credit_line_amount_gt_0 = X.rule_feature_credit_line_amount.fillna(0) > 0
        if credit_line_amount_gt_0.any():
            credit_line_last_updated_at_gt_0 = X.rule_feature_credit_line_last_updated_at.fillna(0) > 0
            if credit_line_last_updated_at_gt_0.any():
                requested_payment_value_lt_rule_feature_credit_line_amount = X.requested_payment_value <= X.rule_feature_credit_line_amount
                if requested_payment_value_lt_rule_feature_credit_line_amount.any():
                    credit_line_status_values = credit_line_amount_gt_0 & credit_line_last_updated_at_gt_0 & requested_payment_value_lt_rule_feature_credit_line_amount
                    credit_line_status_values = credit_line_status_values & \
                                                ((X.rule_feature_request_timestamp.astype(
                                                    float) - X.rule_feature_credit_line_last_updated_at.astype(
                                                    float)) < self._credit_line_event_expiry_seconds)
                    credit_line_status_values = credit_line_status_values.astype(int)
        return credit_line_status_values

    @timer
    def transform_requested_payment_value(self, X):
        """Method to bin to avoid information leakage on requested_payment_value"""

        # num_bins = int(np.floor(1+np.log2(len(X))))
        # X.requested_payment_value = pd.cut(X.requested_payment_value, bins=num_bins, labels=False)
        # return (X.requested_payment_value/500) #float type for sample weights ints are sufficient
        return X.requested_payment_value

    @timer
    def rename_tecton_features(self, column_list):
        column_list = [col.replace("__", ".") for col in column_list]
        for index, val in enumerate(column_list):
            if '.' in val:
                feature_view, feature_name = val.split('.')
                assert len(feature_view) > 0 and len(feature_name) > 0
                full_feature_name = re.sub(r'_v\d+$', '',
                                           feature_view) + '.' + feature_name  # Removing version numbers from feature views
                renamed_feature = re.sub(r'_continuous$', '',
                                         full_feature_name)  # Remove continous keyword from feature names
                column_list[index] = renamed_feature
        return column_list

    @timer
    def map_feature_names(self, column_list):
        for index, val in enumerate(column_list):
            if val in FEATURE_MAPPER:
                column_list[index] = FEATURE_MAPPER[val]
        return column_list

    @timer
    def preprocess_features(self, X):
        """Method to pre-process input features"""
        # print("preprocessing begin")
        # Renaming features - remove version numbers and continous keywords
        X.columns = self.rename_tecton_features(X.columns.tolist())

        # coalescing of features
        X['is_registered'] = X['ifre_member_features.is_registered'].combine_first(
            X['company_core_features.is_registered'])
        X['company_type'] = X['ifre_member_features.company_type'].combine_first(
            X['company_core_features.company_type'])
        X['kyc_risk_band'] = X['whitelisting_features.risk_band'].combine_first(X['company_core_features.risk_band'])

        # Renaming features - using mapper to generate business friendly names
        X.columns = self.map_feature_names(X.columns.tolist())

        X = X.loc[:, INPUT_FEATURES_PRE_TRANSFORMATION]  # Filter the columns that are needed for transformation

        # The below is updated to get a performance boost over dateutil parser
        # if X.rule_feature_request_timestamp.dtype == 'O':
        #  X.rule_feature_request_timestamp = X.rule_feature_request_timestamp.apply(lambda x: int(dateutil.parser.parse(x).timestamp()))
        # else:

        X.rule_feature_request_timestamp = X.rule_feature_request_timestamp.values.astype('datetime64[s]').astype('int')
        X.rule_feature_source_dest_expiry_ts = X.rule_feature_source_dest_expiry_ts.values.astype('datetime64[s]').astype('int')

        X['member_area_code'] = self.transform_member_area_code(X.member_postcode)
        X.member_id_country_issue = self.transform_member_id_country_issue(X.member_id_country_issue.values)
        X.member_id_type = self.transform_member_id_type(X.member_id_type)
        X.registered_company_type = self.transform_registered_company_type(X.registered_company_type)
        # This is required to fill-in zero and avoid nulls in composite features
        # Below is not an imputation, rather replacing nulls with zero, as tecton doesn't do it
        # For example, if there are no deposits on account, sum/avg deposit is returned as null instead of zero by tecton
        X.fillna({x: 1 for x in FILLNA_COLS_WITH_ONE}, inplace=True)

        X.fillna({x: 0 for x in FILLNA_COLS_WITH_ZERO}, inplace=True)

        X['net_balance_indicator'] = self.transform_net_balance_indicator(X)
        X['cashflow_value_indicator'] = self.transform_cashflow_value_indicator(X)
        X['deposit_withdrawal_ratio'] = self.transform_deposit_withdrawal_ratio(X)
        X['cashflow_volume_indicator'] = self.transform_cashflow_volume_indicator(X)
        X['payments_initiated_hour_48h_range'] = self.calculate_pmt_initiated_hour_48h_range(X)
        X['ratio_pmts_outside_business_60d'] = self.calculate_ratio_pmts_outside_business_60days(X)

        X['time_passed_since_confirmation_of_payee'] = self.transform_time_passed_since_confirmation_of_payee(
            X.rule_feature_request_timestamp.values, X.timestamp_ntm_recipient_confirmed.values)
        X['days_since_confirmation_of_payee'] = self.calculate_days_since_payee_added_as_beneficiary(
            X.time_passed_since_confirmation_of_payee.values)
        X['hours_since_confirmation_of_payee'] = self.calculate_hours_since_payee_added_as_beneficiary(
            X.time_passed_since_confirmation_of_payee.values)
        X[
            'incoming_payment_requested_avg_pmt_value_with_payee_ratio_60d'] = self.calculate_incoming_pmt_requested_avg_pmt_value_with_payee_ratio_60days(
            X)
        X[
            'outgoing_payment_requested_avg_pmt_value_with_payee_ratio_60d'] = self.calculate_outgoing_pmt_requested_avg_pmt_value_with_payee_ratio_60days(
            X)

        X['mean_of_payments_to_new_payees_60d'] = self.calculate_pmt_to_new_payees(
            X.rolling_mean_of_withdrawals_to_counter_party_60d.values, X.time_passed_since_confirmation_of_payee.values)
        X['sum_of_payments_to_new_payees_48h'] = self.calculate_pmt_to_new_payees(
            X.rolling_sum_of_withdrawals_to_counter_party_48h.values, X.time_passed_since_confirmation_of_payee.values,
            48)

        X['rolling_sum_of_pmts_over_threshold_to_new_payee_48h'] = self.calculate_pmt_to_new_payees(
            X.rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_48h.values,
            X.time_passed_since_confirmation_of_payee.values, 48)
        X['rolling_sum_of_pmts_over_threshold_to_new_payee_60d'] = self.calculate_pmt_to_new_payees(
            X.rolling_sum_of_withdrawals_value_to_counter_party_abv_thresh_1000_60d.values,
            X.time_passed_since_confirmation_of_payee.values, 48)

        X['ratio_of_pmts_value_to_payee_of_overall_pmts_2h'] = self.calculate_ratio_of_pmts_to_payee_of_overall_pmts(
            X.rolling_sum_of_withdrawals_to_counter_party_2h.values, X.rolling_sum_of_withdrawals_on_account_2h.values)
        X['ratio_of_pmts_value_to_payee_of_overall_pmts_48h'] = self.calculate_ratio_of_pmts_to_payee_of_overall_pmts(
            X.rolling_sum_of_withdrawals_to_counter_party_48h.values,
            X.rolling_sum_of_withdrawals_on_account_48h.values)
        X['ratio_of_pmts_value_to_payee_of_overall_pmts_60d'] = self.calculate_ratio_of_pmts_to_payee_of_overall_pmts(
            X.rolling_sum_of_withdrawals_to_counter_party_60d.values,
            X.average_withdrawal_value_60d.values * X.outgoing_payments_sum_account_60d.values)

        X['payment_requested_mean_withdrawal_ratio_60d'] = self.calculate_payment_requested_ratio(
            X.requested_payment_value.values, X.average_withdrawal_value_60d.abs().values)
        X['payment_requested_max_withdrawal_ratio_60d'] = self.calculate_payment_requested_ratio(
            X.requested_payment_value.values, X.max_withdrawal_value_60d.abs().values)
        X['payment_requested_average_withdrawal_ratio_1y'] = self.calculate_payment_requested_ratio(
            X.requested_payment_value.values, X.average_withdrawal_value_1y.abs().values)
        X['payment_requested_max_withdrawal_ratio_1y'] = self.calculate_payment_requested_ratio(
            X.requested_payment_value.values, X.max_withdrawal_value_1y.abs().values)
        X['payment_requested_mean_deposit_ratio_60d'] = self.calculate_payment_requested_ratio(
            X.requested_payment_value.values, X.average_deposit_value_60d.values)
        X['payment_requested_max_deposit_ratio_60d'] = self.calculate_payment_requested_ratio(
            X.requested_payment_value.values, X.max_deposit_value_60d.values)
        X['payment_requested_average_deposit_ratio_1y'] = self.calculate_payment_requested_ratio(
            X.requested_payment_value.values, X.average_deposit_value_1y.values)
        X['payment_requested_max_deposit_ratio_1y'] = self.calculate_payment_requested_ratio(
            X.requested_payment_value.values, X.max_deposit_value_1y.values)
        X['payment_channel_most_used_for_withdrawals'] = self.transform_payment_channel_most_used_for_withdrawals(
            X).astype(str)
        X['payment_channel_most_used_for_deposits'] = self.transform_payment_channel_most_used_for_deposits(X).astype(
            str)
        X['number_of_high_value_transactions_1y'] = self.transform_number_of_high_value_transactions_on_account_1y(X)
        X['number_of_days_since_last_withdrawal_60d'] = self.transform_number_of_days_since_last_withdrawal_60d(
            X.rule_feature_request_timestamp, X.last_payment_requested_received_at_60d)
        X['is_previous_payment_in_last_15mins'] = self.transform_is_previous_fastpayment_in_last_15mins(
            X.rule_feature_request_timestamp, X.last_payment_requested_received_at_60d)
        X['fastpmt_beneficiaries_benefactors_ratio_1y'] = self.transform_fastpmt_beneficiaries_benefactors_ratio_1y(
            X.number_of_fastpayment_beneficiaries_1y, X.number_of_fastpayment_benefactors_1y)
        X['requested_payment_value'] = self.transform_requested_payment_value(X)

        X = self.composite_payment_features(X)

        X.loc[:, CATEGORICAL_COLS] = X.loc[:, CATEGORICAL_COLS].astype(str).apply(lambda col: col.str.upper())
        X.fillna(value=np.nan, inplace=True)
        X.replace({pd.NA: np.nan}, inplace=True)
        return X

    @timer
    def fit(self, X, y=None):
        """Method to fit the feature transformer with input features"""
        X_fit = X.copy()
        X_fit = self.preprocess_features(X_fit).loc[:, self.model_features]
        self.imputing_pipeline.fit(X_fit)
        self.imputed_feature_list = self.get_feature_list(X_fit, self.imputing_pipeline['imputation'][
            'columntransformer'].transformers_)
        if USE_SEPERATE_ENCODING:
            # print('Using_seperate_categorical_encoding')
            self.cat_encode_pipeline.fit(X_fit, X[TARGET_COL])
            self.cat_encode_feature_list = self.get_feature_list(X_fit, self.cat_encode_pipeline['cat-encode'][
                'columntransformer'].transformers_)
        return self

    @timer
    def transform(self, X):
        """Method to pre-process any given input data using the transformer pipeline"""
        X_transform = X.copy()
        X_transform1 = self.preprocess_features(X_transform)
        X_transform2 = pd.DataFrame(self.imputing_pipeline.transform(X_transform1.loc[:, self.model_features]),
                                    columns=self.imputed_feature_list)
        if USE_SEPERATE_ENCODING:
            # print('Using_seperate_categorical_encoding')
            X_transform2 = pd.DataFrame(self.cat_encode_pipeline.transform(X_transform2.loc[:, self.model_features]),
                                        columns=self.cat_encode_feature_list)
        if 'process_rule_fields' in X_transform.columns:  # Process rule fields only for evaluation periods and production inferencing
            X_transform2['rule_feature_is_recipient_account_exempted'] = self.transform_is_recipient_account_exempted(
                X_transform1.loc[:, ['destination_account_number', 'destination_account_sort_code']])
            X_transform2['rule_feature_is_matching_credit_event'] = self.transform_is_matching_credit_event(
                X_transform1)
            X_transform1.rule_feature_kyc_risk_band = X_transform1.rule_feature_kyc_risk_band.fillna("HIGH").astype(
                str).str.upper()
            X_transform1.rule_feature_is_clearlisted = X_transform1.rule_feature_is_clearlisted.fillna(False).astype(
                bool)
            X_transform1.rule_feature_clearlisting_last_updated_at = pd.to_numeric(
                X_transform1.rule_feature_clearlisting_last_updated_at, 'coerce').fillna(0).astype(int)
            X_transform1.rule_feature_event_ignore_expiration = X_transform1.rule_feature_event_ignore_expiration.fillna(
                False).astype(bool)
            X_transform1.rule_feature_permenant_clearlisting_last_whitelisted_at = pd.to_numeric(
                X_transform1.rule_feature_permenant_clearlisting_last_whitelisted_at, 'coerce').fillna(0).astype(int)
            X_transform1.rule_feature_is_source_dest_clearlisted = X_transform1.rule_feature_is_source_dest_clearlisted.fillna(False).astype(
                bool)
            X_transform2['rule_feature_is_clearlisted'] = self.transform_is_clearlisted(X_transform1)
            X_transform2['rule_feature_is_source_dest_clearlisted'] = self.transform_is_source_dest_clearlisted(X_transform1)


        if 'return_features_data' in X_transform.columns:
            X_transform2['model_explainability_features'] = X_transform2.to_dict(orient='records')
        return X_transform2