import csv
from PyQt5.QtWidgets import QWidget, QFileDialog, QTableWidget, QVBoxLayout, QPushButton, QTableWidgetItem


CP_STRINGS = [
  '_cp_masks', '_cp_outlines', '_cp_flows', '_cp_cellprob'
]
MAIN_CHANNEL_CHOICES = [
  ('average all channels', 0), ('0=red', 1), ('1=green', 2), ('2=blue', 3),
  ('3', 4), ('4', 5), ('5', 6), ('6', 7), ('7', 8), ('8', 9)
]
OPTIONAL_NUCLEAR_CHANNEL_CHOICES = [
  ('none', 0), ('0=red', 1), ('1=green', 2), ('2=blue', 3),
  ('3', 4), ('4', 5), ('5', 6), ('6', 7), ('7', 8), ('8', 9)
]


def csv_export_table(table_widget: QTableWidget):
  options = QFileDialog.Options()
  file_name, _ = QFileDialog.getSaveFileName(None, "Save CSV", "", "CSV Files (*.csv);;All Files (*)", options=options)

  if not bool(file_name): return

  with open(file_name, mode='w', newline='') as file:
    writer = csv.writer(file, delimiter=";")

    headers = [table_widget.horizontalHeaderItem(i).text() for i in range(table_widget.columnCount())]
    writer.writerow(headers)

    for row in range(table_widget.rowCount()):
      row_data = [table_widget.item(row, column).text() for column in range(table_widget.columnCount())]
      writer.writerow(row_data)


def create_table_with_csv_export(header, data) -> QTableWidget:
  container_widget = QWidget()
  layout = QVBoxLayout()
  table_widget = QTableWidget()
  export_button = QPushButton("export to csv")

  table_widget.setRowCount(len(data))
  table_widget.setColumnCount(len(header))
  table_widget.setHorizontalHeaderLabels(header)

  for i in range(len(data)):
    for j in range(len(data[i])):
      table_widget.setItem(i, j, QTableWidgetItem(str(data[i][j])))
  
  layout.addWidget(table_widget)
  layout.addWidget(export_button)
  
  export_button.clicked.connect(lambda: csv_export_table(table_widget))

  container_widget.setLayout(layout)

  return container_widget

