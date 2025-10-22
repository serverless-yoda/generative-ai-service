# app/api/core/postgres/service.py

from typing import Annotated
from fastapi import Depends, HTTPException, status
from sqlalchemy import select

from app.api.db.database import DBSessionDep
from app.api.db.entities import Conversation
from app.api.db.schemas import ConversationCreate, ConversationUpdate, ConversationOut
from app.api.repository.conversation_repository import ConversationRepository

async def get_conversation(
    conversation_id: int, session: DBSessionDep
) -> Conversation:
    conversation = await ConversationRepository(session).get(conversation_id)
    
    if not conversation:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return conversation

GetConversationDependency = Annotated[Conversation, Depends(get_conversation)]


async def list_conversations(session: DBSessionDep, skip: int =0, take: int = 20) -> list[ConversationOut]:
    conversations = await ConversationRepository(session).list(skip, take)
    return [
        ConversationOut.model_validate(conversation)
        for conversation in conversations
    ]


async def retrieve_conversation(conversation : GetConversationDependency) -> ConversationOut:
    return ConversationOut.model_validate(conversation)

async def create_conversation(conversation:ConversationCreate, session:DBSessionDep) -> ConversationOut:
    new_conversation = await ConversationRepository(session).create(
                        conversation
                    )
    return ConversationOut.model_validate(new_conversation)

async def update_conversation(upd_conversation:ConversationUpdate,
            conversation: GetConversationDependency, 
            session: DBSessionDep) -> ConversationOut:
    updated_conversation = await ConversationRepository(session).update(
        conversation.id, upd_conversation
    )
    return ConversationOut.model_validate(updated_conversation)


async def delete_conversation(conversation:GetConversationDependency, session:DBSessionDep) -> None:
    await ConversationRepository(session).delete(conversation.id)