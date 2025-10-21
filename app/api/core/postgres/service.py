# app/api/core/postgres/service.py
#from _typeshed import AnnotationForm
from typing import Annotated
from app.api.routes.postgres import conversation
from fastapi import Depends, HTTPException, status
from sqlalchemy import select

from app.api.db.database import DBSessionDep
from app.api.db.entities import Conversation
from app.api.db.schemas import ConversationCreate, ConversationUpdate, ConversationOut


async def get_conversation(
    conversation_id: int, session: DBSessionDep
) -> Conversation:
    async with session.begin():
        result = await session.execute(
            select(Conversation).where(Conversation.id == conversation_id)
        )
        conversation = result.scalars().first()
    
    if not conversation:
        raise HTTPException(
            status_code = status.HTTP_404_NOT_FOUND,
            detail="Conversation not found"
        )
    
    return conversation

GetConversationDependency = Annotated[Conversation, Depends(get_conversation)]


async def list_conversations(session: DBSessionDep, skip: int =0, take: int = 20) -> list[ConversationOut]:
    async with session.begin():
        result = await session.execute(select(Conversation).offset(skip).limit(take))
    return [
        ConversationOut.model_validate(conversation)
        for conversation in result.scalars().all()
    ]


async def retrieve_conversation(conversation : GetConversationDependency) -> ConversationOut:
    return ConversationOut.model_validate(conversation)

async def create_conversation(conversation:ConversationCreate, session:DBSessionDep) -> ConversationOut:
    new_conversation = Conversation(**conversation.model_dump())
    async with session.begin():
        session.add(new_conversation)
        #await session.commit()
        await session.refresh(new_conversation)
    return ConversationOut.model_validate(new_conversation)

async def update_conversation(upd_conversation:ConversationUpdate,
            conversation: GetConversationDependency, 
            session: DBSessionDep) -> ConversationOut:
    for key, value in upd_conversation.model_dump().items():
        setattr(conversation, key, value)
    async with session.begin():
        #await session.commit()
        await session.refresh(conversation)
    return ConversationOut.model_validate(conversation)


async def delete_conversation(conversation:GetConversationDependency, session:DBSessionDep) -> None:
    async with session.begin():
        await session.delete(conversation)
        #await session.commit()