# init_db.py
import sqlite3

conn = sqlite3.connect('database.db')
cursor = conn.cursor()

cursor.execute('''
CREATE TABLE IF NOT EXISTS products (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    description TEXT NOT NULL,
    effect TEXT NOT NULL,
    skin_type TEXT NOT NULL,
    image TEXT NOT NULL
)
''')

conn.commit()
conn.close()
print("Database initialized.")
