
import pandas as pd
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from sklearn.ensemble import RandomForestRegressor

class LongImpute:
    def __init__(self, df, time_var='fu_year', subject_var='projid'):
        self.df = df
        self.time_var = time_var
        self.subject_var = subject_var

    def loess_impute(self, columns_to_impute):
        df_combined = self.df.copy()
        for column_to_impute in columns_to_impute:
            new_col_name = f"{column_to_impute}_imputed_loess"
            df_combined[new_col_name] = df_combined[column_to_impute]
            for subj in df_combined[self.subject_var].unique():
                subj_df = df_combined[df_combined[self.subject_var] == subj].sort_values(by=self.time_var)
                if subj_df[column_to_impute].isna().all():
                    continue
                na_indices = subj_df[pd.isna(subj_df[column_to_impute])].index
                non_na_indices = subj_df[column_to_impute].dropna().index
                if len(non_na_indices) == 0:
                    continue
                smoothed_values = lowess(
                    subj_df.loc[non_na_indices, column_to_impute],
                    subj_df.loc[non_na_indices, self.time_var],
                    return_sorted=False
                )
                for idx in na_indices:
                    if idx == subj_df.index.min():
                        next_idx = subj_df.loc[idx:, column_to_impute].first_valid_index()
                        pred_value = subj_df.loc[next_idx, column_to_impute] if next_idx is not None else np.nan
                    elif idx == subj_df.index.max():
                        prev_idx = subj_df.loc[:idx, column_to_impute].last_valid_index()
                        pred_value = subj_df.loc[prev_idx, column_to_impute] if prev_idx is not None else np.nan
                    else:
                        closest_idx = np.argmin(np.abs(subj_df.loc[non_na_indices, self.time_var] - subj_df.loc[idx, self.time_var]))
                        pred_value = smoothed_values[closest_idx]
                    df_combined.loc[idx, new_col_name] = pred_value
        return df_combined

    def rf_impute(self, columns_to_impute):
        df_combined = self.df.copy()
        for column_to_impute in columns_to_impute:
            new_col_name = f"{column_to_impute}_imputed_rf"
            df_combined[new_col_name] = df_combined[column_to_impute]
            for subj in df_combined[self.subject_var].unique():
                subj_df = df_combined[df_combined[self.subject_var] == subj].sort_values(by=self.time_var)
                if subj_df[column_to_impute].isna().all():
                    continue
                na_indices = subj_df[pd.isna(subj_df[column_to_impute])].index
                non_na_indices = subj_df[column_to_impute].dropna().index
                if len(non_na_indices) == 0:
                    continue
                X_train = subj_df.loc[non_na_indices, [self.time_var]]
                y_train = subj_df.loc[non_na_indices, column_to_impute]
                rf = RandomForestRegressor(n_estimators=100, random_state=0)
                rf.fit(X_train, y_train)
                for idx in na_indices:
                    if idx == subj_df.index.min():
                        next_idx = subj_df.loc[idx:, column_to_impute].first_valid_index()
                        pred_value = subj_df.loc[next_idx, column_to_impute] if next_idx is not None else np.nan
                    elif idx == subj_df.index.max():
                        prev_idx = subj_df.loc[:idx, column_to_impute].last_valid_index()
                        pred_value = subj_df.loc[prev_idx, column_to_impute] if prev_idx is not None else np.nan
                    else:
                        X_test = subj_df.loc[[idx], [self.time_var]]
                        pred_value = rf.predict(X_test)[0]
                    df_combined.loc[idx, new_col_name] = pred_value
        return df_combined
