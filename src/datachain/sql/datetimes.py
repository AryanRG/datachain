from datetime import MAXYEAR, MINYEAR, datetime, timezone


def _normalize_datetime_for_storage(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value

    try:
        return value.astimezone(timezone.utc).replace(tzinfo=None)
    except (OverflowError, ValueError, OSError):
        if value.year == MAXYEAR:
            return datetime.max.replace(tzinfo=timezone.utc).replace(tzinfo=None)
        if value.year == MINYEAR:
            return datetime.min.replace(tzinfo=timezone.utc).replace(tzinfo=None)
        raise


def serialize_datetime(value: datetime) -> str:
    return _normalize_datetime_for_storage(value).isoformat(" ")
