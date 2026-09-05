import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv(dotenv_path='.env')

user = os.getenv('DB_USER', 'postgres')
password = os.getenv('DB_PASSWORD', 'mysecretpassword')
host = os.getenv('DB_HOST', 'localhost')
port = os.getenv('DB_PORT', '5432')
db_name = os.getenv('DB_NAME', 'olist')

engine = create_engine(f'postgresql://{user}:{password}@{host}:{port}/{db_name}')


orders = pd.read_sql("SELECT * FROM orders;", engine)
customers = pd.read_sql("SELECT * FROM customers;", engine)
order_items = pd.read_sql("SELECT * FROM order_items;", engine)
order_payments = pd.read_sql("SELECT * FROM order_payments;", engine)
order_reviews = pd.read_sql("SELECT * FROM order_reviews;", engine)
products = pd.read_sql("SELECT * FROM products;", engine)
sellers = pd.read_sql("SELECT * FROM sellers;", engine)


items_agg = order_items.groupby('order_id').agg(
    total_price=('price', 'sum'),
    total_freight=('freight_value', 'sum'),
    total_items=('order_item_id', 'count'),
    seller_id=('seller_id', 'first'),
    product_id=('product_id', 'first')
).reset_index()

payments_agg = order_payments.groupby('order_id').agg(
    total_payment=('payment_value', 'sum'),
    payment_installments=('payment_installments', 'max'),
    primary_payment_type=('payment_type', 'first')
).reset_index()

reviews_clean = order_reviews.groupby('order_id').first().reset_index()


ml_df = orders.merge(customers, on='customer_id', how='left')
ml_df = ml_df.merge(items_agg, on='order_id', how='left')
ml_df = ml_df.merge(payments_agg, on='order_id', how='left')
ml_df = ml_df.merge(reviews_clean[['order_id', 'review_score']], on='order_id', how='left')
ml_df = ml_df.merge(products, on='product_id', how='left')
ml_df = ml_df.merge(sellers, on='seller_id', how='left')


os.makedirs('data/processed', exist_ok=True)
output_path = 'data/processed/ml_table.parquet'
ml_df.to_parquet(output_path, index=False)

print(f"Data successfully created and saved to {output_path}")