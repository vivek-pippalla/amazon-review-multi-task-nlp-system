import os
import pymysql
from dotenv import load_dotenv
load_dotenv()
_db = None

def get_connection():
    global _db
    if _db is None or not _db.open:
        try:
            _db = pymysql.connect(
                host=os.getenv("DB_HOST"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME"),
                charset="utf8mb4",
                cursorclass=pymysql.cursors.DictCursor,
                auth_plugin_map={'caching_sha2_password': 'mysql_native_password'}
            )
        except pymysql.err.OperationalError as e:
            print(f"[DB ERROR] Could not connect to MySQL: {e}")
            raise
    return _db

def get_cursor():
    return get_connection().cursor()

def commit():
    get_connection().commit()
