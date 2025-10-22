# app/api/routes/postgres/conversation.py

from fastapi import APIRouter, Depends, status,Response
from app.api.db.database import DBSessionDep
from app.api.db.schemas import ConversationCreate, ConversationUpdate, ConversationOut
from app.api.core.postgres.service import (
    create_conversation,
    retrieve_conversation,
    update_conversation,
    delete_conversation,
    list_conversations,
    get_conversation
)

router = APIRouter()

@router.get("/conversations", response_model=list[ConversationOut])
async def list_all(session: DBSessionDep, skip: int = 0, take: int = 20):
    return await list_conversations(session, skip, take)

@router.get("/conversations/{conversation_id}", response_model=ConversationOut)
async def retrieve(conversation_id: int, session: DBSessionDep):
    conversation = await get_conversation(conversation_id, session)
    return await retrieve_conversation(conversation)

@router.post("/conversations", status_code=status.HTTP_201_CREATED, response_model=ConversationOut)
async def create(conversation: ConversationCreate, session: DBSessionDep):
    return await create_conversation(conversation, session)

@router.put("/conversations/{conversation_id}", response_model=ConversationOut)
async def update(conversation_id: int, upd_conversation: ConversationUpdate, session: DBSessionDep):
    conversation = await get_conversation(conversation_id, session)
    return await update_conversation(upd_conversation, conversation, session)

@router.delete("/conversations/{conversation_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete(conversation_id: int, session: DBSessionDep):
    conversation = await get_conversation(conversation_id, session)
    await delete_conversation(conversation, session)
    return Response(status_code=status.HTTP_204_NO_CONTENT)  # Explicit response