import logging


def truncate_string(s, max_length=500):
    """Truncate long strings for concise printing."""
    if isinstance(s, str) and len(s) > max_length:
        return s[:max_length] + "..."
    return s

# Configure logger  
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)  # Choose your default level (INFO, DEBUG, etc.)


# Optionally add a console handler if you don't have one already
if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s] %(name)s - %(message)s")
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)