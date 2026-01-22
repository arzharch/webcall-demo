import aiofiles
import os
import asyncio
from datetime import datetime
from typing import Optional, List
import logging
from functools import lru_cache

from config import get_settings
from models import CallTranscript, Ticket

logger = logging.getLogger(__name__)

class StorageService:
    """JSON file storage for transcripts and tickets"""
    
    def __init__(self):
        self.settings = get_settings()
        logger.info("✅ Storage Service initialized")
    
    async def save_transcript(self, transcript: CallTranscript) -> bool:
        """
        Save call transcript to JSON file
        
        Args:
            transcript: CallTranscript object
        
        Returns:
            Success status
        """
        try:
            filename = f"{transcript.call_id}.json"
            filepath = os.path.join(self.settings.TRANSCRIPTS_DIR, filename)
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(transcript.model_dump_json(indent=2))
            
            logger.info(f"💾 Saved transcript: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Save transcript error: {e}", exc_info=True)
            return False
    
    async def load_transcript(self, call_id: str) -> Optional[CallTranscript]:
        """
        Load transcript from file
        
        Args:
            call_id: Call ID
        
        Returns:
            CallTranscript or None
        """
        try:
            filename = f"{call_id}.json"
            filepath = os.path.join(self.settings.TRANSCRIPTS_DIR, filename)
            
            if not os.path.exists(filepath):
                return None
            
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                content = await f.read()
                return CallTranscript.model_validate_json(content)
        
        except Exception as e:
            logger.error(f"❌ Load transcript error: {e}", exc_info=True)
            return None
    
    async def list_transcripts(self) -> List[str]:
        """
        List all transcript call IDs
        
        Returns:
            List of call IDs
        """
        try:
            # Run blocking I/O in executor
            loop = asyncio.get_event_loop()
            files = await loop.run_in_executor(
                None,
                lambda: os.listdir(self.settings.TRANSCRIPTS_DIR)
            )
            return [f.replace('.json', '') for f in files if f.endswith('.json')]
        except Exception as e:
            logger.error(f"❌ List transcripts error: {e}", exc_info=True)
            return []
    
    async def save_ticket(self, ticket: Ticket) -> bool:
        """
        Save booking ticket to JSON file
        
        Args:
            ticket: Ticket object
        
        Returns:
            Success status
        """
        try:
            filename = f"{ticket.ticket_id}.json"
            filepath = os.path.join(self.settings.TICKETS_DIR, filename)
            
            async with aiofiles.open(filepath, 'w', encoding='utf-8') as f:
                await f.write(ticket.model_dump_json(indent=2))
            
            logger.info(f"🎫 Saved ticket: {filename}")
            return True
        
        except Exception as e:
            logger.error(f"❌ Save ticket error: {e}", exc_info=True)
            return False
    
    async def load_ticket(self, ticket_id: str) -> Optional[Ticket]:
        """
        Load ticket from file
        
        Args:
            ticket_id: Ticket ID
        
        Returns:
            Ticket or None
        """
        try:
            filename = f"{ticket_id}.json"
            filepath = os.path.join(self.settings.TICKETS_DIR, filename)
            
            if not os.path.exists(filepath):
                return None
            
            async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                content = await f.read()
                return Ticket.model_validate_json(content)
        
        except Exception as e:
            logger.error(f"❌ Load ticket error: {e}", exc_info=True)
            return None
    
    async def list_tickets(self) -> List[Ticket]:
        """
        List all tickets
        
        Returns:
            List of Ticket objects
        """
        try:
            # Run blocking I/O in executor
            loop = asyncio.get_event_loop()
            files = await loop.run_in_executor(
                None,
                lambda: os.listdir(self.settings.TICKETS_DIR)
            )
            
            tickets = []
            for filename in files:
                if filename.endswith('.json'):
                    filepath = os.path.join(self.settings.TICKETS_DIR, filename)
                    async with aiofiles.open(filepath, 'r', encoding='utf-8') as f:
                        content = await f.read()
                        ticket = Ticket.model_validate_json(content)
                        tickets.append(ticket)
            
            return tickets
        
        except Exception as e:
            logger.error(f"❌ List tickets error: {e}", exc_info=True)
            return []

@lru_cache()
def get_storage_service() -> StorageService:
    """Get singleton storage service"""
    return StorageService()
