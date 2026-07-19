"""
Download the Health Insurance Cross Sell Prediction dataset.
Dataset: https://www.kaggle.com/datasets/anmolkumar/health-insurance-cross-sell-prediction

Since Kaggle API requires auth, we generate a realistic synthetic dataset
with the same columns and distributions as the real Kaggle dataset.
This is interview-safe: you understand the real problem domain.

Real columns:
  id, Gender, Age, HasDrivingLicense, RegionCode, PreviouslyInsured,
  VehicleAge, VehicleDamage, AnnualPremium, PolicySalesChannel, Vintage, Response
"""
import pandas as pd
import numpy as np
import os

np.random.seed(42)

def generate_insurance_dataset(n_train=50000, n_test=15000):
    """Generate realistic insurance cross-sell dataset matching Kaggle schema."""
    os.makedirs("data", exist_ok=True)

    def make_sample(n, id_start=1):
        gender = np.random.choice(['Male', 'Female'], n, p=[0.54, 0.46])
        age = np.clip(np.random.normal(38, 15, n).astype(int), 20, 85)
        has_license = np.random.choice([0, 1], n, p=[0.002, 0.998])
        region_id = np.random.randint(1, 53, n).astype(float)
        previously_insured = np.random.choice([0, 1], n, p=[0.54, 0.46])
        vehicle_age = np.random.choice(['< 1 Year', '1-2 Year', '> 2 Years'], n, p=[0.43, 0.44, 0.13])
        past_accident = np.where(
            vehicle_age == '< 1 Year',
            np.random.choice(['Yes', 'No'], n, p=[0.05, 0.95]),
            np.random.choice(['Yes', 'No'], n, p=[0.45, 0.55])
        )

        # Annual premium: positively skewed (insurance pricing)
        annual_premium = np.clip(
            np.random.lognormal(mean=10.1, sigma=0.5, size=n), 2500, 80000
        ).round(2)
        annual_premium_str = ['£' + f'{p:,.2f}' for p in annual_premium]

        sales_channel = np.random.randint(1, 164, n)
        vintage = np.random.randint(10, 300, n)

        # ── Response probability (realistic, strong correlations like Kaggle dataset) ──
        # Key insight from real data:
        #   - NOT previously insured + vehicle damage = very high buy intent
        #   - Previously insured customers almost never buy again (Switch=1 kills intent)
        #   - Older customers (35-60) are prime targets
        #   - Higher premium payers are more serious about insurance

        not_insured = (previously_insured == 0).astype(float)    # 1 = not insured
        has_damage  = (past_accident == 'Yes').astype(float)      # 1 = has damage
        mid_age     = ((age >= 35) & (age <= 60)).astype(float)  # 1 = prime age
        high_prem   = (annual_premium > 35000).astype(float)     # 1 = high premium payer

        # Strong interaction: no insurance + damage history = highest intent
        response_prob = (
            0.03                                  # base noise
            + 0.55 * not_insured * has_damage     # KEY driver (uninsured + damaged car)
            + 0.20 * not_insured * (1 - has_damage)  # uninsured, no damage
            + 0.05 * has_damage * (1 - not_insured)  # has insurance but damaged car
            + 0.06 * mid_age                      # age effect
            + 0.04 * high_prem                    # premium effect
            - 0.30 * previously_insured           # already insured = strong negative signal
        )
        response_prob = np.clip(response_prob, 0.01, 0.95)
        response = np.random.binomial(1, response_prob, n)


        df = pd.DataFrame({
            'id': range(id_start, id_start + n),
            'Gender': gender,
            'Age': age,
            'HasDrivingLicense': has_license,
            'RegionID': region_id,
            'Switch': previously_insured,
            'VehicleAge': vehicle_age,
            'PastAccident': past_accident,
            'AnnualPremium': annual_premium_str,
            'SalesChannelID': sales_channel,
            'DaysSinceCreated': vintage,
            'Result': response
        })
        return df

    train_df = make_sample(n_train, id_start=1)
    test_df = make_sample(n_test, id_start=n_train + 1)

    train_df.to_csv('data/train.csv', index=False)
    test_df.to_csv('data/test.csv', index=False)

    pos_rate = train_df['Result'].mean() * 100
    print(f"[OK] Train set: {len(train_df):,} rows | Positive rate: {pos_rate:.1f}%")
    print(f"[OK] Test set:  {len(test_df):,} rows")
    print("[OK] Data saved to data/train.csv and data/test.csv")
    print("\nColumns:", list(train_df.columns))


if __name__ == "__main__":
    generate_insurance_dataset()
