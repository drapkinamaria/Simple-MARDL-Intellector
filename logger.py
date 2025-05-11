import datetime


class Logger:
    def __init__(self, path: str):
        self.path = path
        with open(self.path, "w") as f:
            f.write(f"=== Training log started at {datetime.datetime.now()} ===\n")

    def log(self, text: str):
        with open(self.path, "a") as f:
            f.write(text + "\n")

    def log_empty_line(self):
        with open(self.path, "a") as f:
            f.write("\n")
