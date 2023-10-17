class MissingRequiredParameterError(Exception):
    """Raised when a required key is missing from kwargs for a specific class."""

    def __init__(self, class_name: str, missing_key: str):
        self.class_name = class_name
        self.missing_key = missing_key
        super().__init__(
            f"'{self.missing_key}' parameter is missing, which is required for the '{self.class_name}' layer.")


class ProteinTooBig(Exception):
    """Exception raised for errors in the input salary.

    Attributes:
        salary -- input salary which caused the error
        message -- explanation of the error
    """

    def __init__(self, size, pdb, message="Protein size is too big to parse"):
        self.size = size
        self.pdb = pdb
        self.message = message
        super().__init__(self.message + f" {pdb} size is {str(size)}")