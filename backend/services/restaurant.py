import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from backend.config import get_settings, Settings


class RestaurantTemplate:
    """Loads static restaurant knowledge base and exposes helper lookups."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        kb_path = os.path.join(self.settings.DATA_DIR, "restaurant_kb.json")
        if os.path.exists(kb_path):
            with open(kb_path, "r", encoding="utf-8") as fp:
                self._kb = json.load(fp)
        else:
            self._kb = {"menu": [], "restaurant_info": {}, "faqs": []}

    @property
    def menu(self) -> List[Dict[str, str]]:
        return self._kb.get("menu", [])

    @property
    def info(self) -> Dict[str, str]:
        return self._kb.get("restaurant_info", {})

    @property
    def faqs(self) -> List[Dict[str, str]]:
        return self._kb.get("faqs", [])

    def get_event_for_day(self, weekday: str) -> Optional[Dict[str, str]]:
        weekday_lower = weekday.lower()
        for event in self.settings.WEEKLY_EVENTS:
            if event["day"].lower() == weekday_lower:
                return event
        return None

    def generate_slots_for_date(self, target_date: datetime) -> List[str]:
        """Return list of slot strings (HH:MM) for the date respecting config."""
        start_time = datetime.strptime(
            self.settings.RESERVATION_SERVICE_START, "%H:%M"
        ).time()
        end_time = datetime.strptime(
            self.settings.RESERVATION_SERVICE_END, "%H:%M"
        ).time()

        cursor = datetime.combine(target_date.date(), start_time)
        cutoff = datetime.combine(target_date.date(), end_time)
        slots: List[str] = []
        while cursor <= cutoff:
            slots.append(cursor.strftime("%H:%M"))
            cursor += timedelta(minutes=self.settings.RESERVATION_SLOT_MINUTES)
        return slots


_template: Optional[RestaurantTemplate] = None


def get_restaurant_template() -> RestaurantTemplate:
    global _template
    if _template is None:
        _template = RestaurantTemplate()
    return _template