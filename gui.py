# gui.py

import sys
import os
import sqlite3
from datetime import datetime

import torch
import torchvision.transforms as transforms
from PIL import Image

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QFileDialog, QTabWidget, QTableWidget, QTableWidgetItem,
    QHeaderView, QMessageBox, QTextEdit
)
from PyQt5.QtGui import QPixmap, QColor, QFont
from PyQt5.QtCore import Qt

from src.predict import predict_single_tensor
from src.disease_info import disease_info   

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DB_PATH = os.path.join("logs", "diagnoses.db")
os.makedirs("logs", exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS diagnostics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            plant TEXT,
            disease TEXT,
            confidence REAL,
            image_path TEXT
        )
    ''')
    conn.commit()
    conn.close()

def save_diagnosis(plant, disease, confidence, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO diagnostics (timestamp, plant, disease, confidence, image_path)
        VALUES (?, ?, ?, ?, ?)
    ''', (timestamp, plant, disease, confidence, image_path))
    conn.commit()
    conn.close()

def get_all_diagnostics():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM diagnostics ORDER BY timestamp DESC")
    rows = cursor.fetchall()
    conn.close()
    return rows


def export_to_csv():
    rows = get_all_diagnostics()
    if not rows:
        QMessageBox.information(None, "Экспорт", "Нет данных для экспорта")
        return

    filename, _ = QFileDialog.getSaveFileName(
        None,
        "Сохранить историю как CSV",
        "diagnostics_export.csv",
        "CSV Files (*.csv);;All Files (*)"
    )

    if not filename:
        return  

    try:
        with open(filename, "w", newline="", encoding="utf-8-sig") as f:
            f.write("ID;Дата;Растение;Болезнь;Точность (%);Путь к фото\n")
            for row in rows:
                fields = [
                    str(row[0]),
                    str(row[1]),
                    f'"{row[2]}"' if ',' in row[2] else row[2],
                    f'"{row[3]}"' if ',' in row[3] else row[3],
                    f"{row[4]*100:.1f}%",
                    row[5] or ""
                ]
                f.write(";".join(fields) + "\n")

        QMessageBox.information(None, "Успешно", f"Файл успешно сохранён:\n{filename}")
        print(f"Экспортировано в {filename}")
    except Exception as e:
        QMessageBox.warning(None, "Ошибка", f"Не удалось сохранить файл:\n{e}")

def export_to_txt():
    rows = get_all_diagnostics()
    if not rows:
        QMessageBox.information(None, "Экспорт", "Нет данных для экспорта")
        return

    filename, _ = QFileDialog.getSaveFileName(
        None,
        "Сохранить историю как TXT",
        "diagnostics_export.txt",
        "Text Files (*.txt);;All Files (*)"
    )

    if not filename:
        return

    try:
        with open(filename, "w", encoding="utf-8") as f:
            f.write("История диагностик\n")
            f.write(f"Всего записей: {len(rows)}\n\n")
            for row in rows:
                f.write(f"Запись #{row[0]}\n")
                f.write(f"Дата:          {row[1]}\n")
                f.write(f"Растение:      {row[2]}\n")
                f.write(f"Болезнь:       {row[3]}\n")
                f.write(f"Точность:      {row[4]*100:.1f}%\n")
                f.write(f"Путь к фото:   {row[5] or '—'}\n")
                f.write("-" * 50 + "\n\n")

        QMessageBox.information(None, "Успешно", f"Файл успешно сохранён:\n{filename}")
        print(f"Экспортировано в {filename}")
    except Exception as e:
        QMessageBox.warning(None, "Ошибка", f"Не удалось сохранить файл:\n{e}")

init_db()

class DragDropLabel(QLabel):
    def __init__(self, on_image_drop, parent=None):
        super().__init__(parent)
        self.on_image_drop = on_image_drop

        self.setText("Select or drag the image")
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("""
            background-color: #f0f0f0;
            border: 3px dashed #aaa;
            border-radius: 12px;
            font-size: 20px;
            color: #555;
        """)
        self.setMinimumSize(500, 500)

        self.setAcceptDrops(True)
        self.setAttribute(Qt.WA_AcceptDrops, True)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dragMoveEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        files = [u.toLocalFile() for u in event.mimeData().urls()]
        if files:
            self.on_image_drop(files[0])
            event.acceptProposedAction()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Диагностика болезней томатов и огурцов по фото листьев")
        self.setGeometry(100, 100, 1400, 900)

        self.current_image_path = None

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        self.tabs = QTabWidget()
        self.tabs.setFixedWidth(450)
        main_layout.addWidget(self.tabs)

        home_tab = QWidget()
        home_layout = QVBoxLayout(home_tab)
        home_layout.setAlignment(Qt.AlignTop)

        logo_layout = QVBoxLayout()
        logo_layout.setAlignment(Qt.AlignCenter)

        logo = QLabel()
        pixmap = QPixmap("logo.png")
        if pixmap.isNull():
            logo.setText("LeafPulse")
            logo.setStyleSheet("font-size: 40px; color: #4CAF50;")
        else:
            logo.setPixmap(pixmap.scaled(260, 260, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            logo.setStyleSheet("background-color: transparent;")
        logo.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(logo)

        title = QLabel("LeafPulse")
        title.setFont(QFont("Segoe UI", 28, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        logo_layout.addWidget(title)

        home_layout.addLayout(logo_layout)

        export_csv_btn = QPushButton("↓ download .csv")
        export_csv_btn.setStyleSheet("font-size: 16px; padding: 12px;")
        export_csv_btn.clicked.connect(export_to_csv)
        home_layout.addWidget(export_csv_btn)

        export_txt_btn = QPushButton("↓ download .txt")
        export_txt_btn.setStyleSheet("font-size: 16px; padding: 12px;")
        export_txt_btn.clicked.connect(export_to_txt)
        home_layout.addWidget(export_txt_btn)

        home_layout.addStretch()
        self.tabs.addTab(home_tab, "Главная")

        history_tab = QWidget()
        history_layout = QVBoxLayout(history_tab)

        self.history_table = QTableWidget()
        self.history_table.setColumnCount(5)
        self.history_table.setHorizontalHeaderLabels(["Дата", "Растение", "Болезнь", "Точность (%)", "Фото"])
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        history_layout.addWidget(self.history_table, stretch=1)

        clear_btn = QPushButton("Очистить всю историю")
        clear_btn.setStyleSheet("font-size: 16px; padding: 12px;")
        clear_btn.clicked.connect(self.clear_history)
        history_layout.addWidget(clear_btn)

        self.tabs.addTab(history_tab, "История")

        right_widget = QWidget()
        right_layout = QHBoxLayout(right_widget)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        self.drag_area = DragDropLabel(self.load_image)
        left_layout.addWidget(self.drag_area, stretch=1)

        choose_btn = QPushButton("Выбрать изображение")
        choose_btn.setStyleSheet("font-size: 18px; padding: 12px;")
        choose_btn.clicked.connect(self.open_file)
        left_layout.addWidget(choose_btn)

        self.diagnose_btn = QPushButton("Диагностировать")
        self.diagnose_btn.setStyleSheet("font-size: 18px; padding: 12px; background-color: #4CAF50; color: white;")
        self.diagnose_btn.clicked.connect(self.run_diagnosis)
        self.diagnose_btn.setEnabled(False)
        left_layout.addWidget(self.diagnose_btn)

        right_layout.addWidget(left_widget, stretch=4)

        result_widget = QWidget()
        result_layout = QVBoxLayout(result_widget)
        result_layout.setAlignment(Qt.AlignTop)

        self.plant_box = QLabel("Растение: —")
        self.plant_box.setStyleSheet("font-size: 22px; padding: 15px; border-radius: 10px; background-color: #e0e0e0;")
        self.plant_box.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.plant_box)

        self.disease_box = QLabel("Диагноз заболевания: —")
        self.disease_box.setStyleSheet("font-size: 20px; padding: 15px; border-radius: 10px; background-color: #e0e0e0;")
        self.disease_box.setAlignment(Qt.AlignCenter)
        result_layout.addWidget(self.disease_box)

        self.description_box = QTextEdit()
        self.description_box.setReadOnly(True)
        self.description_box.setStyleSheet("font-size: 18px; padding: 15px; border-radius: 10px; background-color: #f8f8f8;")
        self.description_box.setMinimumHeight(400)
        result_layout.addWidget(self.description_box)

        right_layout.addWidget(result_widget, stretch=2)

        main_layout.addWidget(right_widget, stretch=4)

        self.load_history()

    def open_file(self):
        fname = QFileDialog.getOpenFileName(self, 'Выберите фото', '', 'Images (*.png *.jpg *.jpeg *.bmp)')
        if fname[0]:
            self.load_image(fname[0])

    def load_image(self, path):
        self.current_image_path = path

        pixmap = QPixmap(path).scaled(600, 600, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.drag_area.setPixmap(pixmap)
        self.drag_area.setText("")

        self.plant_box.setText("Растение: —")
        self.plant_box.setStyleSheet("font-size: 22px; padding: 15px; border-radius: 10px; background-color: #e0e0e0;")
        self.disease_box.setText("Диагноз заболевания: —")
        self.description_box.clear()

        self.diagnose_btn.setEnabled(True)

    def run_diagnosis(self):
        if not self.current_image_path:
            return

        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            img_tensor = transform(Image.open(self.current_image_path).convert("RGB")).unsqueeze(0).to(device)

            plant, disease, confidence = predict_single_tensor(img_tensor)

            if disease.lower() == "healthy":
                color = QColor("#C2EABD")
            else:
                color = QColor("#FE6847")

            self.plant_box.setText(f"Растение: {plant}")
            self.plant_box.setStyleSheet(f"font-size: 22px; padding: 15px; border-radius: 10px; background-color: {color.name()};")

            self.disease_box.setText(f"Диагноз заболевания: {disease} (точность {confidence:.1%})")

            info = disease_info.get(disease)
            if info:
                text = f"Описание:\n{info['description']}\n\n"
                text += "Симптомы:\n"
                for s in info["symptoms"]:
                    text += f"- {s}\n"
                text += f"\nРекомендации:\n{info['treatment']}"
            else:
                text = "Нет информации о заболевании"

            self.description_box.setText(text)

            save_diagnosis(plant, disease, confidence, self.current_image_path)
            self.load_history()

        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось обработать фото:\n{e}")

    def load_history(self):
        rows = get_all_diagnostics()
        self.history_table.setRowCount(len(rows))
        for i, row in enumerate(rows):
            self.history_table.setItem(i, 0, QTableWidgetItem(row[1]))
            self.history_table.setItem(i, 1, QTableWidgetItem(row[2]))
            self.history_table.setItem(i, 2, QTableWidgetItem(row[3]))
            self.history_table.setItem(i, 3, QTableWidgetItem(f"{row[4]*100:.1f}%"))
            self.history_table.setItem(i, 4, QTableWidgetItem(os.path.basename(row[5]) if row[5] else "—"))

    def clear_history(self):
        reply = QMessageBox.question(self, "Очистка", "Удалить всю историю?", QMessageBox.Yes | QMessageBox.No)
        if reply == QMessageBox.Yes:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("DELETE FROM diagnostics")
            conn.commit()
            conn.close()
            self.load_history()
            QMessageBox.information(self, "Готово", "История очищена")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())