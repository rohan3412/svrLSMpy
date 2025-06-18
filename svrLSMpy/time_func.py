from datetime import datetime
def get_current_datetime_for_filename():
    now = datetime.now()
    # Format the date and time as a string, using '-' and '_' to avoid invalid file name characters
    formatted_datetime = now.strftime("%Y-%m-%d_%I%M_%p")  # Use %I for 12-hour format and %p for AM/PM
    file_time = f"({formatted_datetime})"
    return file_time

def easy_time(seconds):
    seconds = int(seconds)
    hour, remainder = divmod(seconds, 3600)
    minute, seconds = divmod(remainder, 60)

    parts = []
    if hour:
        if hour==1:
            parts.append(f"{hour} HR")
        else:
            parts.append(f"{hour} HRS")
    if minute:
        parts.append(f"{minute} MIN")
    if seconds:
        parts.append(f"{seconds} SEC")

    return " ".join(parts)
