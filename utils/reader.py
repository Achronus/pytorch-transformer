import gzip


class RawTextReader:
    """Reads raw data from a GZip file."""
    def __init__(self, filename: str) -> None:
        self.filename = filename

    def read_data(self) -> str:
        """Reads a single file within a GZip file (without an extension) and returns its data as a string."""
        with gzip.open(self.filename, 'r') as f:
            data = f.read()
        return str(data)
