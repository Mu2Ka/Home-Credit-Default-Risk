from data_loader import load_core_tables
from features import merge_bureau_balance_features, merge_bureau_features, merge_credit_credit_features



def main() -> None:
    tables = load_core_tables(".")
    train = merge_bureau_features(
        application_train=tables["application_train"],
        bureau=tables["bureau"],
    )
    train = merge_bureau_balance_features(
        application_train=train,
        bureau=tables["bureau"],
        bureau_balance=tables["bureau_balance"],
    )
    train = merge_credit_credit_features(
        application_train=train,
        credit_card_balance=tables["credit_card_balance"],
    )
    print(f"Dataset shape: {train.shape}")


if __name__ == "__main__":    main()
