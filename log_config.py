import logging

# Set up basic logging configuration
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,  # Adjust as necessary
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        # handlers=[
        #     logging.StreamHandler()  # Also log to the console
        # ]
    )