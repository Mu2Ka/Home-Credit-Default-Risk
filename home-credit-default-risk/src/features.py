import numpy as np
import pandas as pd


def build_bureau_features(bureau: pd.DataFrame) -> pd.DataFrame:
    bureau = bureau.copy()

    bureau["ACTIVE_CREDIT"] = (bureau["DAYS_CREDIT_ENDDATE"] > 0).astype(int)
    bureau["PAST_ENDDATE"] = (bureau["DAYS_CREDIT_ENDDATE"] < 0).astype(int)
    bureau["EARLY_CLOSE"] = (
            bureau["DAYS_ENDDATE_FACT"] > bureau["DAYS_CREDIT_ENDDATE"]
    ).astype(int)

    bureau_agg = bureau.groupby("SK_ID_CURR").agg(
        CREDIT_SUM=("AMT_CREDIT_SUM", "sum"),
        CREDIT_COUNT=("SK_ID_BUREAU", "count"),
        CREDIT_MEAN=("AMT_CREDIT_SUM", "mean"),
        CREDIT_MAX=("AMT_CREDIT_SUM", "max"),
        DAYS_CREDIT_MEAN=("DAYS_CREDIT", "mean"),
        DAYS_CREDIT_MOST_RECENT=("DAYS_CREDIT", "max"),
        DAYS_CREDIT_OLDEST=("DAYS_CREDIT", "min"),
        CREDIT_DAY_OVERDUE_MAX=("CREDIT_DAY_OVERDUE", "max"),
        CREDIT_DAY_OVERDUE_MIN=("CREDIT_DAY_OVERDUE", "min"),
        CREDIT_DAY_OVERDUE_MEAN=("CREDIT_DAY_OVERDUE", "mean"),
        CREDIT_DAY_OVERDUE_SUM=("CREDIT_DAY_OVERDUE", "sum"),
        DAYS_CREDIT_ENDDATE_MAX=("DAYS_CREDIT_ENDDATE", "max"),
        DAYS_CREDIT_ENDDATE_MIN=("DAYS_CREDIT_ENDDATE", "min"),
        DAYS_CREDIT_ENDDATE_MEAN=("DAYS_CREDIT_ENDDATE", "mean"),
        DAYS_CREDIT_ENDDATE_SUM=("DAYS_CREDIT_ENDDATE", "sum"),
        DAYS_ENDDATE_FACT_MEAN=("DAYS_ENDDATE_FACT", "mean"),
        ACTIVE_CREDIT_SUM=("ACTIVE_CREDIT", "sum"),
        ACTIVE_CREDIT_MEAN=("ACTIVE_CREDIT", "mean"),
        PAST_ENDDATE_SUM=("PAST_ENDDATE", "sum"),
        PAST_ENDDATE_MEAN=("PAST_ENDDATE", "mean"),
        EARLY_CLOSE_SUM=("EARLY_CLOSE", "sum"),
        EARLY_CLOSE_MEAN=("EARLY_CLOSE", "mean"),
    ).reset_index()

    type_credit = (
        bureau.groupby(["SK_ID_CURR", "CREDIT_TYPE"])
        .size()
        .unstack(fill_value=0)
        .reset_index()
    )

    return bureau_agg.merge(type_credit, on="SK_ID_CURR", how="left")


def merge_bureau_features(
        application_train: pd.DataFrame, bureau: pd.DataFrame
) -> pd.DataFrame:
    bureau_features = build_bureau_features(bureau)
    return application_train.merge(bureau_features, on="SK_ID_CURR", how="left")


def build_bureau_balance_features(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    bureau_balance = bureau_balance.copy()

    bb_agg = bureau_balance.groupby("SK_ID_BUREAU").agg({
        "MONTHS_BALANCE": ["min", "max", "count"]
    })
    bb_agg.columns = ["bb_min", "bb_max", "bb_count"]

    bureau_balance["status_to_num"] = bureau_balance["STATUS"].replace({
        "C": 0,
        "X": 0,
    }).astype(int)
    status_agg = bureau_balance.groupby("SK_ID_BUREAU")["status_to_num"].agg([
        "max",
        "mean",
        "sum",
    ])
    status_agg.columns = ["bb_status_max", "bb_status_mean", "bb_status_sum"]

    bureau_balance["is_bad"] = bureau_balance["STATUS"].isin(
        ["1", "2", "3", "4", "5"]
    ).astype(int)
    bad_agg = bureau_balance.groupby("SK_ID_BUREAU")["is_bad"].mean()
    bad_agg = bad_agg.rename("bb_bad_ratio")

    return (
        bb_agg
        .join(status_agg)
        .join(bad_agg)
    ).reset_index()


def build_bureau_balance_agg_features(
        bureau: pd.DataFrame, bureau_balance: pd.DataFrame
) -> pd.DataFrame:
    bb_features = build_bureau_balance_features(bureau_balance)
    bureau_full = bureau.merge(bb_features, on="SK_ID_BUREAU", how="left")

    return bureau_full.groupby("SK_ID_CURR").agg(
        BB_MIN_MIN=("bb_min", "min"),
        BB_MAX_MAX=("bb_max", "max"),
        BB_COUNT_MEAN=("bb_count", "mean"),
        BB_COUNT_SUM=("bb_count", "sum"),
        BB_STATUS_MAX_MAX=("bb_status_max", "max"),
        BB_STATUS_MEAN_MEAN=("bb_status_mean", "mean"),
        BB_STATUS_SUM_SUM=("bb_status_sum", "sum"),
        BB_BAD_RATIO_MEAN=("bb_bad_ratio", "mean"),
        BB_BAD_RATIO_MAX=("bb_bad_ratio", "max"),
    ).reset_index()


def merge_bureau_balance_features(
        application_train: pd.DataFrame,
        bureau: pd.DataFrame,
        bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    bureau_balance_features = build_bureau_balance_agg_features(
        bureau=bureau,
        bureau_balance=bureau_balance,
    )
    return application_train.merge(
        bureau_balance_features,
        on="SK_ID_CURR",
        how="left",
    )


def build_credit_card_features(
        credit_data: pd.DataFrame
) -> pd.DataFrame:
    credit_card_features = credit_data.copy()
    credit_card_features['balance_to_limit'] = (
            credit_card_features['AMT_BALANCE'] /
            credit_card_features['AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    )
    credit_card_features['payment_to_min'] = credit_card_features['AMT_PAYMENT_CURRENT'] / credit_card_features[
        'AMT_INST_MIN_REGULARITY'].replace(0, np.nan)
    credit_card_features['payment_to_balance'] = credit_card_features['AMT_PAYMENT_CURRENT'] / credit_card_features[
        'AMT_BALANCE'].replace(0, np.nan)
    credit_card_features['drawings_to_limit'] = credit_card_features['AMT_DRAWINGS_CURRENT'] / credit_card_features[
        'AMT_CREDIT_LIMIT_ACTUAL'].replace(0, np.nan)
    cc_agg = credit_card_features.groupby('SK_ID_CURR').agg(
        balance_to_limit_mean=('balance_to_limit', 'mean'),
        balance_to_limit_max=('balance_to_limit', 'max'),
        dpd_mean=('SK_DPD', 'mean'),
        dpd_max=('SK_DPD', 'max'),
        payment_to_min_mean=('payment_to_min', 'mean'),
        payment_to_min_max=('payment_to_min', 'max'),
        payment_to_balance_mean=('payment_to_balance', 'mean'),
        payment_to_balance_max=('payment_to_balance', 'max'),
        drawings_to_limit_mean=('drawings_to_limit', 'mean'),
        drawings_to_limit_max=('drawings_to_limit', 'max')
    ).reset_index()
    return cc_agg


def merge_credit_credit_features(application_train: pd.DataFrame, credit_card_balance: pd.DataFrame) -> pd.DataFrame:
    credit_card_features = build_credit_card_features(credit_card_balance)
    return application_train.merge(credit_card_features, on="SK_ID_CURR", how="left")


def build_installments_payments_features(
        credit_data: pd.DataFrame
) -> pd.DataFrame:
    ins_pay = credit_data
    ins_pay['days_late'] = ins_pay['DAYS_ENTRY_PAYMENT'] - ins_pay['DAYS_INSTALMENT']
    ins_pay['late_flag'] = (ins_pay['days_late'] > 0).astype(int)
    ins_pay['pay_diff'] = ins_pay['AMT_PAYMENT'] - ins_pay['AMT_INSTALMENT']
    ins_pay['pay_ratio'] = ins_pay['AMT_PAYMENT'] / ins_pay['AMT_INSTALMENT'].replace(0, np.nan)
    ins_pay['early_flag'] = (ins_pay['days_late'] < 0).astype(int)
    ins_pay['underpaid_flag'] = (ins_pay['pay_diff'] < 0).astype(int)
    ins_pay['overpaid_flag'] = (ins_pay['pay_diff'] > 0).astype(int)
    ins_pay['severe_late_30'] = (ins_pay['days_late'] > 30).astype(int)
    ins_agg = ins_pay.groupby('SK_ID_CURR').agg(
        INS_DAYS_LATE_MEAN=('days_late', 'mean'),
        INS_DAYS_LATE_MAX=('days_late', 'max'),
        INS_LATE_RATIO=('late_flag', 'mean'),
        INS_LATE_COUNT=('late_flag', 'sum'),
        INS_EARLY_RATIO=('early_flag', 'mean'),
        INS_PAY_DIFF_MEAN=('pay_diff', 'mean'),
        INS_PAY_DIFF_MIN=('pay_diff', 'min'),
        INS_PAY_RATIO_MEAN=('pay_ratio', 'mean'),
        INS_PAY_RATIO_MIN=('pay_ratio', 'min'),
        INS_UNDERPAID_RATIO=('underpaid_flag', 'mean'),
        INS_SEVERE_LATE_30_RATIO=('severe_late_30', 'mean'),
        INS_AMT_PAYMENT_SUM=('AMT_PAYMENT', 'sum'),
        INS_AMT_INSTALMENT_SUM=('AMT_INSTALMENT', 'sum'),
        INS_RECORD_COUNT=('SK_ID_PREV', 'count'),
        INS_PREV_LOAN_NUNIQUE=('SK_ID_PREV', 'nunique')
    ).reset_index()
    return ins_agg


def merge_installments_payments_features(application_train: pd.DataFrame,
                                         installments_payments: pd.DataFrame) -> pd.DataFrame:
    installments_payments_features = build_installments_payments_features(installments_payments)
    return application_train.merge(installments_payments_features, on="SK_ID_CURR", how="left")
