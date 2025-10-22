# app/api/repository/conversation_repositoy.py
from typing import Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.db.entities import Conversation
from app.api.db.schemas import ConversationCreate, ConversationBase, ConversationUpdate
from app.api.repository.conversation_interface import IConversationRepository


class ConversationRepository(IConversationRepository):
    def __init__(self, session: AsyncSession) -> None:
        self.session = session

    async def list(self, skip: int, take: int) -> list[Conversation]:
        async with self.session.begin():
            result = await self.session.execute(
                select(Conversation).offset(skip).limit(take)
            )
        return result.scalars().all()

    async def get(self, conversation_id:int) -> Conversation | None:
        async with self.session.begin():
            result = await self.session.execute(
                select(Conversation).where(Conversation.id == conversation_id)
            )
        return result.scalars().first()

    async def create(self, conversation: ConversationCreate) -> Conversation:
        new_conversation = Conversation(**conversation.model_dump())
        self.session.add(new_conversation)
        await self.session.commit()      # explicit commit
        await self.session.refresh(new_conversation)
        return new_conversation

    async def update(self, conversation_id: int, updated_conversation: ConversationUpdate) -> Conversation | None:
        conversation = await self.get(conversation_id)
        if not conversation:
            return None
        for key, value in updated_conversation.model_dump().items():
            setattr(conversation, key, value)
        await self.session.commit()       # commit changes
        await self.session.refresh(conversation)
        return conversation

    async def delete(self, conversation_id: int) -> None:
        conversation = await self.get(conversation_id)
        if not conversation:
            return
        self.session.delete(conversation)  # synchronous delete
        await self.session.commit()