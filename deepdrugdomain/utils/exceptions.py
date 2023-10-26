class MissingRequiredParameterError(Exception):
    """Raised when a required key is missing from kwargs for a specific class."""

    def __init__(self, class_name: str, missing_key: str):
        self.class_name = class_name
        self.missing_key = missing_key
        super().__init__(
            f"'{self.missing_key}' parameter is missing, which is required for the '{self.class_name}'.")


class ProteinTooBig(Exception):
    """
    Exception raised when the size of a protein is too large for the system's memory to handle.
    
    Attributes:
        size (int): Size of the protein.
        pdb (str): Identifier or name of the protein.
        message (str, optional): Default error message for the exception. Default is "Protein size is too big to parse".

    Example:
        >>> if protein_size > MAX_SIZE:
        ...     raise ProteinTooBig(protein_size, protein_pdb, "Custom error message")
        ...
    """

    def __init__(self, size, pdb, message="Protein size is too big to parse"):
        self.size = size
        self.pdb = pdb
        self.message = message
        super().__init__(self.message + f" {pdb} size is {str(size)}")