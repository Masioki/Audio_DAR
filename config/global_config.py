import datetime
import logging
import os
from pathlib import PurePath, Path

import colorlog
from dotenv import load_dotenv

load_dotenv()

SAMPLE_RATE = int(os.environ.get("SAMPLE_RATE", 16000))
FRAME_SIZE = float(os.environ.get("FRAME_SIZE", 0.025))
HF_CONFIG = {
    "token": os.environ.get("HF_TOKEN"),
    "write_permission": bool(os.environ.get("HF_WRITE_PERMISSION", False))
}
SPEAKER_COLUMN_NAME = "speaker"
TEXT_COLUMN_NAME = "text"
AUDIO_COLUMN_NAME = "audio"
CONVERSATION_COLUMN_NAME = "conversation"
LABEL_COLUMN_NAME = "label"

# logging
LOG_DIR = Path(os.environ.get("LOG_DIR", "./logs"))
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_DIR.mkdir(parents=True, exist_ok=True)

log = logging.getLogger(__name__)
log_filename = PurePath(LOG_DIR,
                        f"logs_{datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')}.log")
file_formatter = logging.Formatter('%(asctime)s.%(msecs)03d %(levelname)-5s --- %(name)s : %(message)s',
                                   datefmt='%Y-%m-%d %H:%M:%S')
file_handler = logging.FileHandler(log_filename)
file_handler.setFormatter(file_formatter)

console_formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s%(asctime)s.%(msecs)03d %(levelname)-5s --- %(name)s : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red'
    }
)
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(console_formatter)
log.addHandler(file_handler)
log.addHandler(stream_handler)
log.setLevel(LOG_LEVEL)
log.info(
    f"Running with configuration: SAMPLE_RATE={SAMPLE_RATE}, HF_WRITE_PERMISSION={HF_CONFIG['write_permission']}, LOG_LEVEL={LOG_LEVEL}")
