"""Tools for the Bella Agent."""
import json
from datetime import datetime
from pathlib import Path
from typing import List

from langchain_core.tools import tool

DATA_DIR = Path(__file__).parent.parent / "data"
TICKETS_FILE = DATA_DIR / "tickets.json"
RESTAURANT_KB_FILE = DATA_DIR / "restaurant_kb.json"

# FIX #12: Cache restaurant KB in memory to avoid disk I/O on every menu query
_RESTAURANT_KB_CACHE: dict = None

def get_existing_tickets() -> List[dict]:
    """Get all existing tickets."""
    if not TICKETS_FILE.exists():
        return []
    with open(TICKETS_FILE) as f:
        return json.load(f)

def load_restaurant_kb() -> dict:
    """Load the restaurant knowledge base. Cached after first load."""
    global _RESTAURANT_KB_CACHE
    
    if _RESTAURANT_KB_CACHE is not None:
        return _RESTAURANT_KB_CACHE
    
    if not RESTAURANT_KB_FILE.exists():
        _RESTAURANT_KB_CACHE = {"menu": [], "faq": []}
        return _RESTAURANT_KB_CACHE
    
    with open(RESTAURANT_KB_FILE, 'r', encoding='utf-8') as f:
        _RESTAURANT_KB_CACHE = json.load(f)
    
    return _RESTAURANT_KB_CACHE

@tool
def check_availability(
    party_size: int, date_str: str, time_str: str | None = None
) -> str:
    """
    Check if a table is available for a given party size, date, and optionally time.

    Args:
        party_size: The number of people in the party.
        date_str: The date of the reservation, e.g., "YYYY-MM-DD".
        time_str: The time of the reservation, e.g., "HH:MM".
    """
    # In a real system, this would check a database. Here we'll use a simple logic.
    # For simplicity, let's assume the restaurant has a capacity of 50
    # and is open from 17:00 to 23:00.

    try:
        reservation_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return "It seems the date or time format is incorrect. Please use YYYY-MM-DD and HH:MM."

    if not (17 <= reservation_dt.hour < 23):
        return f"Sorry, we are closed at {time_str}. Our opening hours are from 17:00 to 23:00."

    tickets = get_existing_tickets()
    current_bookings = sum(
        t["party_size"]
        for t in tickets
        if datetime.fromisoformat(t["datetime"]).date() == reservation_dt.date()
        and abs(
            (datetime.fromisoformat(t["datetime"]) - reservation_dt).total_seconds()
        )
        < 2 * 3600  # 2-hour window
    )

    if current_bookings + party_size > 50:
        return f"I'm sorry, but we are fully booked around {time_str} on {date_str}. Is there another time that would work for you?"

    return f"Yes, we have a table available for {party_size} on {date_str} at {time_str}."


@tool
def make_booking(
    name: str, party_size: int, date_str: str, time_str: str, notes: str | None = None
) -> str:
    """
    Make a reservation for a given party size, date, and time.

    Args:
        name: The name of the person making the booking.
        party_size: The number of people in the party.
        date_str: The date of the reservation, e.g., "YYYY-MM-DD".
        time_str: The time of the reservation, e.g., "HH:MM".
        notes: Any special requests or notes for the booking.
    """
    tickets = get_existing_tickets()

    try:
        reservation_dt = datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return "It seems the date or time format is incorrect. Please use YYYY-MM-DD and HH:MM."

    ticket = {
        "id": len(tickets) + 1,
        "name": name,
        "party_size": party_size,
        "datetime": reservation_dt.isoformat(),
        "notes": notes or "",
    }
    tickets.append(ticket)

    with open(TICKETS_FILE, "w") as f:
        json.dump(tickets, f, indent=2)
    
    notes_msg = f" I've also noted your request: '{notes}'." if notes else ""
    return f"I've made a booking for {name} for {party_size} people on {date_str} at {time_str}.{notes_msg} The booking ID is {ticket['id']}. We look forward to seeing you!"


@tool
def cancel_booking(booking_id: int) -> str:
    """
    Cancel a booking given a booking ID.

    Args:
        booking_id: The ID of the booking to cancel.
    """
    tickets = get_existing_tickets()
    ticket_to_cancel = next((t for t in tickets if t["id"] == booking_id), None)

    if not ticket_to_cancel:
        return f"I'm sorry, I couldn't find a booking with the ID {booking_id}. Please double-check the ID."

    tickets = [t for t in tickets if t["id"] != booking_id]
    with open(TICKETS_FILE, "w") as f:
        json.dump(tickets, f, indent=2)

    return f"Your booking with ID {booking_id} has been successfully canceled."


@tool
def find_booking(name: str | None = None, booking_id: int | None = None) -> str:
    """
    Find a booking by name or booking ID.

    Args:
        name: The name of the person who made the booking.
        booking_id: The ID of the booking.
    """
    if not name and not booking_id:
        return "Please provide a name or a booking ID to find a reservation."

    tickets = get_existing_tickets()
    results = []
    if booking_id:
        results = [t for t in tickets if t["id"] == booking_id]
    elif name:
        results = [t for t in tickets if t["name"].lower() == name.lower()]

    if not results:
        search_term = f"ID {booking_id}" if booking_id else f"the name '{name}'"
        return f"I'm sorry, I couldn't find any bookings under {search_term}."

    return (
        "I found the following bookings:\n"
        + "\n".join(
            [
                f"  - ID: {t['id']}, Name: {t['name']}, Party: {t['party_size']}, Time: {t['datetime']}"
                for t in results
            ]
        )
    )


@tool
def update_booking(
    booking_id: int | str,
    new_party_size: int | None = None,
    new_date_str: str | None = None,
    new_time_str: str | None = None,
    name: str | None = None,
) -> str:
    """
    Update an existing booking's party size, date, time, or name.

    Args:
        booking_id: The ID of the booking to update.
        new_party_size: The new number of people in the party.
        new_date_str: The new date for the reservation.
        new_time_str: The new time for the reservation.
        name: The new name for the reservation.
    """
    # Robustness: Handle case where Agent passes a JSON string as the first argument
    # because it thinks the tool takes a single string input.
    if isinstance(booking_id, str):
        booking_id = booking_id.strip()
        if booking_id.startswith("{") and booking_id.endswith("}"):
            try:
                data = json.loads(booking_id)
                # Remap fields
                booking_id = data.get("booking_id", booking_id)
                new_party_size = data.get("new_party_size", new_party_size)
                new_date_str = data.get("new_date_str", new_date_str)
                new_time_str = data.get("new_time_str", new_time_str)
                name = data.get("name", name)
            except json.JSONDecodeError:
                pass # Try to use as is (maybe it's a string ID?)
        
        # Try to convert to int if it's numeric
        try:
             booking_id = int(booking_id)
        except (ValueError, TypeError):
             pass # Keep as string or whatever it ended up as

    tickets = get_existing_tickets()
    ticket_idx = -1
    for i, t in enumerate(tickets):
        if t["id"] == booking_id:
            ticket_idx = i
            break

    if ticket_idx == -1:
        return f"I'm sorry, I couldn't find a booking with the ID {booking_id}. Please double-check the ID."

    ticket = tickets[ticket_idx]
    original_datetime = datetime.fromisoformat(ticket["datetime"])

    if new_party_size is not None:
        ticket["party_size"] = new_party_size
    
    if name is not None:
        ticket["name"] = name

    new_date = new_date_str or original_datetime.strftime("%Y-%m-%d")
    new_time = new_time_str or original_datetime.strftime("%H:%M")
    ticket["datetime"] = datetime.strptime(
        f"{new_date} {new_time}", "%Y-%m-%d %H:%M"
    ).isoformat()

    tickets[ticket_idx] = ticket
    with open(TICKETS_FILE, "w") as f:
        json.dump(tickets, f, indent=2)

    return f"I've updated booking {booking_id}. The new details are: Party of {ticket['party_size']} on {new_date} at {new_time}."


@tool
def search_menu(query: str) -> str:
    """
    Search the restaurant's menu and FAQ for information.

    Args:
        query: The search query, e.g., "What is in the pasta?", "Do you have vegan options?", "What are your opening hours?".
    """
    kb = load_restaurant_kb()
    menu_items = kb.get("menu", [])
    faqs = kb.get("faq", [])

    results = []

    # Search menu items
    for item in menu_items:
        if query.lower() in item.get("name", "").lower() or \
           query.lower() in item.get("description", "").lower():
            results.append(f"Menu Item: {item.get('name')} - {item.get('description')} (Price: {item.get('price')})")
    
    # Search FAQs
    for faq in faqs:
        if query.lower() in faq.get("question", "").lower() or \
           query.lower() in faq.get("answer", "").lower():
            results.append(f"FAQ: {faq.get('question')} - {faq.get('answer')}")

    if results:
        return "\n".join(results)
    else:
        return f"I couldn't find anything related to '{query}' on the menu or in the FAQs."