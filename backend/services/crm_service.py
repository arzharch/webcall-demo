import json
import asyncio
from typing import List, Optional, Dict, Any
from datetime import datetime
from functools import lru_cache
import logging
import os

from backend.config import get_settings
from backend.models import Ticket, TicketStatus

logger = logging.getLogger(__name__)

class CRMService:
    """In-memory CRM with JSON persistence"""
    
    def __init__(self):
        self.settings = get_settings()
        self.tickets: Dict[str, Ticket] = {}
        self.is_initialized = False
    
    async def initialize(self):
        """Load tickets from JSON"""
        try:
            if os.path.exists(self.settings.TICKETS_FILE):
                with open(self.settings.TICKETS_FILE, 'r') as f:
                    data = json.load(f)
                    for ticket_dict in data.get("tickets", []):
                        ticket = Ticket(**ticket_dict)
                        self.tickets[ticket.id] = ticket
            
            self.is_initialized = True
            logger.info(f"CRM Service initialized with {len(self.tickets)} tickets")
        except Exception as e:
            logger.error(f"CRM initialization error: {e}")
    
    async def create_ticket_from_intent(self, booking_intent: BookingIntent, call_id: str, summary: str = "") -> Ticket:
        """Creates a CRM ticket directly from a booking intent."""
        if not self.is_initialized:
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
                summary=summary
            )
            
            self.tickets[ticket.id] = ticket
            await self._save_tickets()
            print(f"  ✓ Created ticket: {ticket.id} with status {ticket.status}")
            return ticket
            
    async def get_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """Get ticket by ID"""
        return self.tickets.get(ticket_id)
    
    async def list_tickets(
        self,
        status: Optional[TicketStatus] = None
    ) -> List[Ticket]:
        """List tickets, optionally filtered by status"""
        tickets = list(self.tickets.values())
        
        if status:
            tickets = [t for t in tickets if t.status == status]
        
        tickets.sort(key=lambda t: t.timestamp, reverse=True)
        return tickets
    
    async def update_ticket(
        self,
        ticket_id: str,
        updates: Dict[str, Any]
    ) -> Optional[Ticket]:
        """Update ticket"""
        if ticket_id not in self.tickets:
            return None
        
        ticket = self.tickets[ticket_id]
        
        for key, value in updates.items():
            if hasattr(ticket, key):
                setattr(ticket, key, value)
        
        await self._save_tickets()
        return ticket
    
    async def _save_tickets(self):
        """Save tickets to JSON"""
        try:
            data = {
                "tickets": [t.model_dump() for t in self.tickets.values()]
            }
            
            with open(self.settings.TICKETS_FILE, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving tickets: {e}")

@lru_cache()
def get_crm_service() -> CRMService:
    """Get cached CRM service instance"""
    return CRMService()
