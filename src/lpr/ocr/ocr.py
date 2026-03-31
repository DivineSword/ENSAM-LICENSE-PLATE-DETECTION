import easyocr

_reader = None


def _get_reader() -> easyocr.Reader:
    """Initialise the EasyOCR reader once and reuse it (lazy loading)."""
    global _reader
    if _reader is None:
        _reader = easyocr.Reader(["en"])
    return _reader


def read_text(image):
    reader = _get_reader()
    results = reader.readtext(image)
    for (bbox, text, prob) in results:
        if prob > 0.4:
            return text
    return None
