"""Test a MongoDB connection using the MONGO_DB_URL environment variable."""
import os
from dotenv import load_dotenv
from pymongo import MongoClient

load_dotenv()
url = os.getenv("MONGO_DB_URL")
if not url:
    raise RuntimeError("Set MONGO_DB_URL in .env first")
client = MongoClient(url)
client.admin.command("ping")
print("MongoDB connection succeeded")
