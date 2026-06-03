import os
import threading
import pymysql
from dotenv import load_dotenv

load_dotenv()

# One connection per thread — fixes the global-singleton race condition that
# caused concurrent Flask requests to corrupt each other's DB state.
_local = threading.local()


def get_connection():
    if not hasattr(_local, "db") or _local.db is None or not _local.db.open:
        _local.db = pymysql.connect(
            host=os.getenv("DB_HOST"),
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            auth_plugin_map={"caching_sha2_password": "mysql_native_password"},
            autocommit=False,
        )
    return _local.db


def get_cursor():
    return get_connection().cursor()


def commit():
    get_connection().commit()


def close_connection():
    """Call at the end of each request to return the connection cleanly."""
    db = getattr(_local, "db", None)
    if db and db.open:
        db.close()
    _local.db = None
