import json
import sqlite3

from app.core import DATA_DIR, COLLECTION


class Store:
    def __init__(self):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.path = DATA_DIR / "lab.sqlite3"
        with self.connect() as db:
            db.executescript('''
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT, collection TEXT, data TEXT,
                    PRIMARY KEY(id, collection));
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY, created TEXT, data TEXT, feedback TEXT);
            ''')

    def connect(self):
        return sqlite3.connect(self.path)

    def documents(self):
        with self.connect() as db:
            return [json.loads(row[0]) for row in db.execute(
                "SELECT data FROM documents WHERE collection=? ORDER BY rowid DESC", (COLLECTION,))]

    def save_document(self, document):
        with self.connect() as db:
            db.execute("INSERT OR REPLACE INTO documents VALUES (?, ?, ?)",
                       (document["id"], COLLECTION, json.dumps(document)))

    def remove_document(self, document_id):
        with self.connect() as db:
            db.execute("DELETE FROM documents WHERE id=? AND collection=?", (document_id, COLLECTION))

    def save_run(self, run):
        with self.connect() as db:
            db.execute("INSERT INTO runs VALUES (?, ?, ?, NULL)",
                       (run["id"], run["created"], json.dumps(run)))

    def runs(self):
        with self.connect() as db:
            return [{"id": row[0], "created": row[1], "question": json.loads(row[2])["question"],
                     "feedback": json.loads(row[3]) if row[3] else None}
                    for row in db.execute("SELECT * FROM runs ORDER BY created DESC LIMIT 100")]

    def run(self, run_id):
        with self.connect() as db:
            row = db.execute("SELECT data, feedback FROM runs WHERE id=?", (run_id,)).fetchone()
        if not row:
            return None
        return {**json.loads(row[0]), "feedback": json.loads(row[1]) if row[1] else None}

    def feedback(self, run_id, feedback):
        with self.connect() as db:
            result = db.execute("UPDATE runs SET feedback=? WHERE id=?", (json.dumps(feedback), run_id))
            return result.rowcount > 0
