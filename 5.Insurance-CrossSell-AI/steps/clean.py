import numpy as np
from sklearn.impute import SimpleImputer


class Cleaner:
    def __init__(self):
        self.imputer = SimpleImputer(strategy='most_frequent', missing_values=np.nan)

    def clean_data(self, data):
        # Drop columns not needed for prediction
        cols_to_drop = [c for c in ['id', 'SalesChannelID', 'VehicleAge', 'DaysSinceCreated'] if c in data.columns]
        data = data.drop(cols_to_drop, axis=1)

        # Clean AnnualPremium — handle both string (£1,234.56) and numeric formats
        if data['AnnualPremium'].dtype == object:
            data['AnnualPremium'] = (
                data['AnnualPremium']
                .str.replace('£', '', regex=False)
                .str.replace(',', '', regex=False)
                .astype(float)
            )

        # Impute categoricals
        for col in ['Gender', 'RegionID']:
            if col in data.columns:
                data[col] = self.imputer.fit_transform(data[[col]]).flatten()

        # Impute numerics
        if 'Age' in data.columns:
            data['Age'] = data['Age'].fillna(data['Age'].median())
        if 'HasDrivingLicense' in data.columns:
            data['HasDrivingLicense'] = data['HasDrivingLicense'].fillna(1)
        if 'Switch' in data.columns:
            data['Switch'] = data['Switch'].fillna(0)
        if 'PastAccident' in data.columns:
            data['PastAccident'] = data['PastAccident'].fillna("Unknown")

        # Remove extreme AnnualPremium outliers (IQR method)
        Q1 = data['AnnualPremium'].quantile(0.25)
        Q3 = data['AnnualPremium'].quantile(0.75)
        IQR = Q3 - Q1
        upper_bound = Q3 + 1.5 * IQR
        data = data[data['AnnualPremium'] <= upper_bound].copy()

        return data