import logging

# Set logging level
LEVEL = logging.DEBUG

# Set logger for application
logging.basicConfig(
    level=LEVEL,
    format="%(levelname)s %(message)s",
    # format="%(asctime)s %(levelname)s %(message)s",  # Add datetime
    datefmt="%Y-%m-%d %H:%M:%S",
    # filename="app.log"  # track logs
)

logger = logging.getLogger('logger')


def enable_logging(logger: logging.Logger, flag: bool) -> None:
    logger.disabled = True

    if flag:
        logger.disabled = False
