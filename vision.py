import os
import pandas as pd
from sqlalchemy import create_engine, inspect
from dotenv import load_dotenv

load_dotenv()

db_user = os.getenv("DB_USER")
db_password = os.getenv("DB_PASSWORD")
db_host = os.getenv("DB_HOST")
db_port = os.getenv("DB_PORT")
db_name = os.getenv("DB_NAME")

engine = create_engine(f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}")
inspector = inspect(engine)

print("📌 الجداول الموجودة والأعمدة التي تربط بينها:\n")

for table in inspector.get_table_names():
    columns = [col['name'] for col in inspector.get_columns(table)]
    print(f"🔹 Table: [{table}]")
    print(f"   Columns: {columns[:4]} ... (Total: {len(columns)} columns)")
    print("-" * 50)