# src/database.py
"""
SQLite database for storing diagnosis history.
Works offline, creates 'logs/diagnoses.db' automatically.
"""

import sqlite3
import os
from datetime import datetime
import csv

# Path to database (in logs/ folder as per your project structure)
DB_PATH = os.path.join("logs", "diagnoses.db")

def init_db():
    """Create database and table if they don't exist""" 
    os.makedirs("logs", exist_ok=True)
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnostics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            plant_type TEXT NOT NULL,
            disease TEXT NOT NULL,
            confidence REAL NOT NULL,
            image_path TEXT,
            user_note TEXT
        )
    ''')
    
    conn.commit()
    conn.close()
    print("Database initialized:", DB_PATH)

def save_diagnosis(plant_type: str, disease: str, confidence: float,
                   image_path: str = None, user_note: str = ""):
    """Save one diagnosis record"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO diagnostics (timestamp, plant_type, disease, confidence, image_path, user_note)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, plant_type, disease, confidence, image_path, user_note))
    
    conn.commit()
    conn.close()
    
    print(f"Saved: {timestamp} | {plant_type} | {disease} ({confidence*100:.1f}%)")

def get_all_diagnostics() -> list:
    """Return all diagnosis records (newest first)"""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM diagnostics ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    
    conn.close()
    return rows

def export_to_csv(filename: str = "diagnostics_export.csv"):
    """Export all history to CSV file"""
    rows = get_all_diagnostics()
    if not rows:
        print("Database is empty. Nothing to export.")
        return
    
    headers = [
        "ID",
        "Date",
        "Plant",
        "Disease",
        "Confidence (%)",
        "Image Path",
        "Note"
    ]
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            confidence_percent = row[4] * 100 if row[4] is not None else 0
            writer.writerow([
                row[0], row[1], row[2], row[3],
                f"{confidence_percent:.1f}",
                row[5], row[6]
            ])
    
    print(f"Exported {len(rows)} records to {filename}")