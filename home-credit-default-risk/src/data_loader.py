from pathlib import Path

import pandas as pd


def load_application_train(data_dir: str | Path = ".") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "application_train.csv")

def load_application_test(data_dir: str | Path = ".") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "application_test.csv")


def load_bureau(data_dir: str | Path = ".") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "bureau.csv")


def load_bureau_balance(data_dir: str | Path = ".") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "bureau_balance.csv")


def load_credit_card_balance(data_dir: str | Path = ".") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "credit_card_balance.csv")


def load_installments_payments(data_dir: str | Path = ".") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "installments_payments.csv")

def load_POS_CASH_balance(data_dir: str | Path = ".") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "POS_CASH_balance.csv")

def load_previous_application(data_dir: str | Path = ".") -> pd.DataFrame:
    data_dir = Path(data_dir)
    return pd.read_csv(data_dir / "previous_application.csv")


def load_core_tables(data_dir: str | Path = ".") -> dict[str, pd.DataFrame]:
    return {
        "application_test": load_application_test(data_dir),
        "test": load_application_test(data_dir),
        "application_train": load_application_train(data_dir),
        "bureau": load_bureau(data_dir),
        "bureau_balance": load_bureau_balance(data_dir),
        "credit_card_balance": load_credit_card_balance(data_dir),
        "installments_payments": load_installments_payments(data_dir),
        "POS_CASH_balance": load_POS_CASH_balance(data_dir),
        "previous_application": load_previous_application(data_dir),
    }

