import os
import csv
from datetime import datetime
import shutil
import threading

class Logger:
    def __init__(self, folder_path, file_name, level="INFO", rotation=None, archive_freq=None):
        self.folder_path = folder_path
        self.archive_folder = os.path.join(folder_path, "Archive")
        self.file_name = file_name
        self.level = level
        self.rotation = rotation
        self.archive_freq = archive_freq
        self.lock = threading.Lock()

        if not os.path.exists(self.folder_path):
            os.makedirs(self.folder_path)
        if not os.path.exists(self.archive_folder):
            os.makedirs(self.archive_folder)

        self.determine_last_time_suffix()
        self.create_log_file()
        self.initial_archive_logs()

    def determine_last_time_suffix(self):
        max_suffix = ""
        current_time = datetime.now()

        for file in os.listdir(self.folder_path):
            if file.startswith(self.file_name) and file.endswith(".csv"):
                parts = file.replace(self.file_name + '_', '').rsplit('.', 1)[0]
                try:
                    file_time = datetime.strptime(parts, self.rotation_format())
                    if file_time > current_time:
                        continue
                    if max_suffix == "" or file_time.strftime(self.rotation_format()) > max_suffix:
                        max_suffix = file_time.strftime(self.rotation_format())
                except ValueError:
                    continue

        self.time_suffix = max_suffix if max_suffix else self.current_time_format()

    def current_time_format(self):
        if self.rotation == "hourly":
            return datetime.now().strftime("%Y_%m_%d__%H")
        elif self.rotation == "daily":
            return datetime.now().strftime("%Y_%m_%d")
        else:
            return ""

    def rotation_format(self):
        return "%Y_%m_%d__%H" if self.rotation == "hourly" else "%Y_%m_%d"

    def create_log_file(self):
        with self.lock:
            self.file_path = os.path.join(self.folder_path, f"{self.file_name}_{self.time_suffix}.csv" if self.time_suffix else f"{self.file_name}.csv")
            file_exists = os.path.exists(self.file_path)
            with open(self.file_path, mode='a', newline='') as file:
                if not file_exists:
                    writer = csv.writer(file, delimiter='\t')
                    writer.writerow(['DateTime', 'LogLevel', 'LogMessage'])

    def update_file_name(self):
        old_time_suffix = self.time_suffix
        self.time_suffix = self.current_time_format()
        if self.time_suffix != old_time_suffix:
            self.archive_logs()
            self.create_log_file()

    def log(self, log_message, level="INFO"):
        if self.should_log(level):
            self.update_file_name()
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with self.lock:
                with open(self.file_path, mode='a', newline='') as file:
                    writer = csv.writer(file, delimiter='\t')
                    writer.writerow([current_datetime, level, log_message])
                    print(f"{current_datetime}\t{level}\t{log_message}")

    def should_log(self, level):
        levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        return levels.index(level) >= levels.index(self.level)

    def archive_logs(self):
        self.move_old_logs()

    def initial_archive_logs(self):
        self.move_old_logs(consider_current=False)

    def move_old_logs(self, consider_current=True):
        if self.archive_freq:
            for file_name in os.listdir(self.folder_path):
                if file_name.endswith(".csv") and (not consider_current or file_name != os.path.basename(self.file_path)):
                    date_part = file_name.replace(f"{self.file_name}_", "", 1)
                    date_part = os.path.splitext(date_part)[0]
                    if '__' in date_part:
                        file_date = date_part.split('__')[0] if self.rotation == "hourly" else date_part
                    else:
                        file_date = date_part  # No hour included

                    if self.archive_freq == "daily" and file_date != self.time_suffix.split('__')[0]:
                        shutil.move(
                            os.path.join(self.folder_path, file_name),
                            os.path.join(self.archive_folder, file_name)
                        )
                    elif self.archive_freq == "hourly" and file_date != self.time_suffix:
                        shutil.move(
                            os.path.join(self.folder_path, file_name),
                            os.path.join(self.archive_folder, file_name)
                        )

if __name__ == "__main__":
    logger = Logger("logs_test", "logfile_test", level="DEBUG") #, rotation="daily", archive_freq="daily")
    logger.log("This is a DEBUG log message.", level="DEBUG")
    logger.log("This is an INFO log message.", level="INFO")
    logger.log("This is a WARNING log message.", level="WARNING")
    logger.log("This is an ERROR log message.", level="ERROR")
    logger.log("This is a CRITICAL log message.", level="CRITICAL")
