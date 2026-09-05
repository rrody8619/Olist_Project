import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_url)

print(" Testing Database Connection & Running SQL Queries...\n")

# 1. Querying orders table
orders_df = pd.read_sql("SELECT * FROM orders LIMIT 5;", con=engine)
print(" First 5 Orders:")
print(orders_df[['order_id', 'customer_id', 'order_status', 'order_delivered_customer_date']])
print("-" * 60)

# 2. Testing JOIN between orders and customers using customer_id
join_query = """
SELECT 
    o.order_id, 
    o.order_status, 
    c.customer_city, 
    c.customer_state
FROM orders o
JOIN customers c ON o.customer_id = c.customer_id
LIMIT 5;
"""

joined_df = pd.read_sql(join_query, con=engine)
print("🔗 SQL Join Test (Orders + Customers):")
print(joined_df)
print("All tests passed! Task 1 completed successfully!")