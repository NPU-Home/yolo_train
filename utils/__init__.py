from datetime import datetime
from .split_dataset import *


def get_current_time_in_min():
    current = datetime.now()
    year = current.year
    month = f"{current.month:02d}"
    day = f"{current.day:02d}"
    hour = f"{current.hour:02d}"
    minute = f"{current.minute:02d}"

    formatted_time = f"{year % 100}{month}{day}_{hour}{minute}"
    return formatted_time
