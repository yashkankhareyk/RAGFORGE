"""
memory.py
=========
Conversational memory for RAGForge using the modern LangChain API.

Uses RunnableWithMessageHistory (NOT deprecated ConversationChain).
Uses InMemoryChatMessageHistory (NOT deprecated ConversationBufferMemory).

Memory strategy: Window buffer — keeps last K turns.
Why window over summary:
  - Summary requires an extra LLM call (costs tokens + latency)
  - For RAG, recent context is almost always what matters
  - Window of 6 turns = ~3 exchanges = enough for follow-up questions
  - Zero extra API cost

Session design:
  - Each browser tab = one session_id (UUID generated on frontend)
  - Multiple sessions supported simultaneously
  - Sessions live in-process (restart = fresh memory, which is fine)
  - Upgrade path: swap InMemoryChatMessageHistory for RedisChatMessageHistory
    for persistence across restarts with zero code change

Verified API: langchain-core >= 0.3.0
"""

import logging
from typing import Dict
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

logger = logging.getLogger(__name__)

# In-process session store — maps session_id -> InMemoryChatMessageHistory
# Replace with Redis/Mongo for persistence across restarts
_session_store: Dict[str, InMemoryChatMessageHistory] = {}

# Window size — how many past messages to keep
# 6 messages = 3 human turns + 3 AI responses
WINDOW_K = 6


def get_session_history(session_id: str) -> InMemoryChatMessageHistory:
    """
    Returns (or creates) the message history for a given session_id.
    Called automatically by RunnableWithMessageHistory on every invoke.
    """
    if session_id not in _session_store:
        _session_store[session_id] = InMemoryChatMessageHistory()
        logger.info(f"New session created: {session_id[:8]}...")
    return _session_store[session_id]


def trim_history(session_id: str):
    """
    Keeps only the last WINDOW_K messages for a session.
    Called after each turn to prevent unbounded growth.
    """
    history = get_session_history(session_id)
    if len(history.messages) > WINDOW_K:
        # Keep only the last WINDOW_K messages
        history.messages = history.messages[-WINDOW_K:]


def clear_session(session_id: str):
    """Clears all history for a session (user clicks 'New Chat')."""
    if session_id in _session_store:
        del _session_store[session_id]
        logger.info(f"Session cleared: {session_id[:8]}...")


def get_session_count() -> int:
    """Returns number of active sessions (for the dashboard)."""
    return len(_session_store)


def get_all_sessions() -> list:
    """Returns session IDs and their turn counts (for the dashboard)."""
    return [
        {
            "session_id": sid[:8] + "...",
            "turns": len(hist.messages) // 2,
        }
        for sid, hist in _session_store.items()
    ]


def build_conversational_rag_chain(rag_chain, llm):
    """
    Wraps a RAG chain with RunnableWithMessageHistory.

    The chain receives:
      - question: current user question
      - history:  MessagesPlaceholder filled by RunnableWithMessageHistory
      - context:  retrieved docs (injected before calling the chain)

    Returns a chain that can be invoked with session_id config.
    """
    # Prompt that includes conversation history
    CONVERSATIONAL_RAG_PROMPT = ChatPromptTemplate.from_messages([
        ("system",
         """You are a helpful assistant. Use ONLY the context below to answer.
If the answer is not in the context, say: "I don't have enough information."
Never make up facts. Never use knowledge outside the context.

Context:
{context}"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ])

    chain = CONVERSATIONAL_RAG_PROMPT | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    return chain_with_history