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
folder_path = os.getenv("DATA_FOLDER")


db_url = f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
engine = create_engine(db_url)

print("⚡ Starting data ingestion to PostgreSQL...\n")

for file_name in os.listdir(folder_path):
    if file_name.endswith(".csv"):
        file_path = os.path.join(folder_path, file_name)

        
        table_name = file_name.replace("olist_", "").replace("_dataset", "").replace(".csv", "")

        print(f" Uploading file: {file_name} -> Table: [{table_name}]...")

        df = pd.read_csv(file_path)
        df.to_sql(name=table_name, con=engine, if_exists="replace", index=False)

        print(f" Successfully ingested {len(df):,} rows!\n")

print(" All dataset files have been successfully uploaded to PostgreSQL!")