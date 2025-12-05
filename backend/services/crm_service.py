import json
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from functools import lru_cache
from pathlib import Path
import aiofiles

from config import get_settings
from models import Ticket, TicketStatus, BookingIntent

class CRMService:
    """
    In-memory CRM service with JSON persistence for managing reservation tickets.
    Uses aiofiles for non-blocking file I/O.
    """
    
    def __init__(self):
        self.settings = get_settings()
        self.tickets: Dict[str, Ticket] = {}
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self):
        """Initialize CRM by loading existing tickets from the JSON file asynchronously."""
        if self._initialized:
            return
        
        print("🔄 Initializing CRM Service...")
        async with self._lock:
            tickets_file = Path(self.settings.TICKETS_FILE)
            if tickets_file.exists():
                try:
                    async with aiofiles.open(tickets_file, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        data = json.loads(content)
                        for ticket_data in data.get('tickets', []):
                            ticket = Ticket(**ticket_data)
                            self.tickets[ticket.id] = ticket
                    print(f"  ✓ Loaded {len(self.tickets)} existing tickets from {tickets_file}.")
                except (json.JSONDecodeError, TypeError) as e:
                    print(f"  ⚠ Could not load or parse tickets file, starting fresh: {e}")
                    self.tickets = {}
            else:
                print(f"  > Tickets file not found at {tickets_file}. Starting fresh.")
        
        self._initialized = True
        print("✅ CRM Service initialized.")
    
    async def create_ticket_from_intent(self, booking_intent: BookingIntent, call_id: str, summary: str = "") -> Ticket:
        """Creates a CRM ticket directly from a booking intent."""
        if not self._initialized:
            await self.initialize()

        async with self._lock:
            status = TicketStatus.CONFIRMED if booking_intent.is_complete() else TicketStatus.PENDING
            
            ticket = Ticket(
                call_id=call_id,
                customer_name=booking_intent.customer_name,
                customer_phone=booking_intent.phone,
                intent="reservation",
                details=booking_intent.model_dump(exclude_none=True),
                status=status,
                transcript= [], # Initialize transcript
                summary=summary
            )
            
            self.tickets[ticket.id] = ticket
            await self._save_tickets()
            print(f"  ✓ Created ticket: {ticket.id} with status {ticket.status}")
            return ticket
            
    async def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Get a single ticket by its ID."""
        return self.tickets.get(ticket_id)
    
    async def list_tickets(self, status: Optional[TicketStatus] = None) -> List[Ticket]:
        """List all tickets, optionally filtering by status."""
        tickets = list(self.tickets.values())
        if status:
            tickets = [t for t in tickets if t.status == status]
        
        tickets.sort(key=lambda t: t.timestamp, reverse=True)
        return tickets

    async def _save_tickets(self):
        """Persist the current state of all tickets to the JSON file asynchronously."""
        try:
            tickets_data = {
                "last_updated": datetime.utcnow().isoformat(),
                "tickets": [ticket.model_dump(mode='json') for ticket in self.tickets.values()],
            }
            
            async with aiofiles.open(self.settings.TICKETS_FILE, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(tickets_data, indent=2, default=str))
        except Exception as e:
            print(f"  ❌ Failed to save tickets to file: {e}")

@lru_cache()
def get_crm_service() -> CRMService:
    """Get a cached singleton instance of the CRMService."""
    return CRMService()