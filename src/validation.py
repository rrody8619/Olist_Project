import pandas as pd
from great_expectations.dataset import PandasDataset

def validate_input_data(df: pd.DataFrame) -> bool:
    ge_df = PandasDataset(df)

    res_price = ge_df.expect_column_values_to_be_between("total_price", min_value=0, max_value=100000)
    res_freight = ge_df.expect_column_values_to_be_between("total_freight", min_value=0, max_value=10000)
    res_items = ge_df.expect_column_values_to_be_between("total_items", min_value=1, max_value=100)
    res_score = ge_df.expect_column_values_to_be_between("review_score", min_value=1.0, max_value=5.0)
    res_not_null = ge_df.expect_column_values_to_not_be_null("order_purchase_timestamp")

    all_passed = (
        res_price.success and 
        res_freight.success and 
        res_items.success and 
        res_score.success and 
        res_not_null.success
    )

    if not all_passed:
        raise ValueError("Data validation failed: Input payload violated Great Expectations rules.")

    return True