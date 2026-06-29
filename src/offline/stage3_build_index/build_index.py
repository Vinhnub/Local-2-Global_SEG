import os
import sqlite3
import numpy as np
import time
from pathlib import Path

def create_db(db_path):
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute('DROP TABLE IF EXISTS queries')
    c.execute('DROP TABLE IF EXISTS database')
    c.execute('CREATE TABLE queries (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, feature BLOB)')
    c.execute('CREATE TABLE database (id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT UNIQUE, feature BLOB)')
    conn.commit()
    return conn

def insert_feature(conn, table, name, feature):
    blob = feature.astype(np.float32).tobytes()
    conn.execute(f'INSERT OR REPLACE INTO {table} (name, feature) VALUES (?, ?)', (name, blob))

def build_index(dataset_name, project_root):
    # Paths to extracted features
    base_dir = Path(project_root) / 'output' / 'stage1' / 'features' / 'fire' / dataset_name
    query_dir = base_dir / 'query'
    db_dir = base_dir / 'database'
    if not query_dir.is_dir() or not db_dir.is_dir():
        print(f'No features found for dataset {dataset_name}. Skipping indexing.')
        return
    # Ensure indexing output directory exists
    index_path = Path(project_root) / 'output' / 'indexing' / f'{dataset_name}.db'
    os.makedirs(index_path.parent, exist_ok=True)
    # Timing start
    start_time = time.time()
    conn = create_db(index_path)
    for npy_file in query_dir.glob('*.npy'):
        name = npy_file.stem
        feat = np.load(npy_file)
        insert_feature(conn, 'queries', name, feat)
    for npy_file in db_dir.glob('*.npy'):
        name = npy_file.stem
        feat = np.load(npy_file)
        insert_feature(conn, 'database', name, feat)
    conn.commit()
    conn.close()
    elapsed = time.time() - start_time
    hours = elapsed / 3600
    print(f'Index built and saved to {index_path}')
    print(f'Indexing time for {dataset_name}: {hours:.2f} hours ({elapsed:.0f} seconds)')
    # Optionally write timing to a summary file
    timing_file = Path(project_root) / 'output' / 'indexing' / 'timings.txt'
    with open(timing_file, 'a') as tf:
        tf.write(f'{dataset_name}: {hours:.2f} hours ({elapsed:.0f} seconds)\n')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Build SQLite index of FIRe features')
    parser.add_argument('--dataset', required=True, help='Dataset name (e.g., roxford5k or rparis6k)')
    parser.add_argument('--project_root', default=os.getcwd(), help='Root directory of the project')
    args = parser.parse_args()
    build_index(args.dataset, args.project_root)
