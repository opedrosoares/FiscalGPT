#!/usr/bin/env python3
"""
Persistência local (SQLite) para interações do chatbot e feedback do usuário.
Armazena: pergunta do usuário, resposta do assistente, fontes, metadados e feedback (👍/👎).
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


def _ensure_parent_dir(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


class LocalDB:
    """Camada simples de acesso ao SQLite para interações do chatbot."""

    def __init__(self, db_path: Optional[Path] = None) -> None:
        try:
            # Preferência: usar caminho do config, se existir
            if db_path is None:
                try:
                    from chatbot.config.config import LOCAL_DB_PATH  # type: ignore
                    db_path = Path(LOCAL_DB_PATH)
                except Exception:
                    # Fallback para shared/data/chatbot_feedback.db na raiz do projeto
                    db_path = Path(__file__).parent.parent.parent / 'shared' / 'data' / 'chatbot_feedback.db'
            self.db_path = Path(db_path)
            _ensure_parent_dir(self.db_path)
            self._initialize()
        except Exception as exc:
            raise RuntimeError(f"Falha ao inicializar banco local: {exc}")

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    user_question TEXT NOT NULL,
                    assistant_answer TEXT NOT NULL,
                    sources_json TEXT,
                    metadata_json TEXT,
                    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    feedback INTEGER -- 1 = up, -1 = down, NULL = sem feedback
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_interactions_created_at
                ON interactions (created_at DESC);
                """
            )
            conn.commit()

    # --------------------------- Operações públicas ---------------------------
    def save_interaction(
        self,
        session_id: str,
        user_question: str,
        assistant_answer: str,
        sources: Optional[List[Dict[str, Any]]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_at: Optional[datetime] = None,
    ) -> int:
        """Salva uma interação e retorna o id gerado."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                """
                INSERT INTO interactions (
                    session_id, user_question, assistant_answer, sources_json, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    session_id,
                    user_question,
                    assistant_answer,
                    json.dumps(sources or [], ensure_ascii=False),
                    json.dumps(metadata or {}, ensure_ascii=False),
                    (created_at or datetime.now()).isoformat(timespec='seconds'),
                ),
            )
            conn.commit()
            return int(cur.lastrowid)

    def set_feedback(self, interaction_id: int, feedback: Optional[int]) -> None:
        """Define feedback (1, -1 ou None) para uma interação."""
        if feedback not in (None, -1, 1):
            raise ValueError("feedback deve ser 1, -1 ou None")
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(
                "UPDATE interactions SET feedback = ? WHERE id = ?",
                (feedback, interaction_id),
            )
            conn.commit()

    def get_interaction(self, interaction_id: int) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT * FROM interactions WHERE id = ?", (interaction_id,))
            row = cur.fetchone()
            return dict(row) if row else None

    def list_interactions(
        self,
        limit: int = 200,
        feedback: Optional[int] = None,
        session_id: Optional[str] = None,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Lista interações com filtros simples."""
        clauses: List[str] = []
        params: List[Any] = []
        if feedback in (-1, 0, 1):
            if feedback == 0:
                clauses.append("feedback IS NULL")
            else:
                clauses.append("feedback = ?")
                params.append(feedback)
        if session_id:
            clauses.append("session_id = ?")
            params.append(session_id)
        if since:
            clauses.append("created_at >= ?")
            params.append(since.isoformat(timespec='seconds'))
        if until:
            clauses.append("created_at <= ?")
            params.append(until.isoformat(timespec='seconds'))

        where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = (
            f"SELECT id, session_id, user_question, assistant_answer, sources_json, metadata_json, created_at, feedback "
            f"FROM interactions {where_sql} ORDER BY datetime(created_at) DESC LIMIT ?"
        )
        params.append(limit)

        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute(sql, tuple(params))
            rows = cur.fetchall()
            return [dict(r) for r in rows]

    def get_stats(self) -> Dict[str, int]:
        """Retorna contadores simples de feedbacks e total."""
        with self._connect() as conn:
            cur = conn.cursor()
            cur.execute("SELECT COUNT(1) AS c FROM interactions")
            total = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(1) AS c FROM interactions WHERE feedback = 1")
            up = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(1) AS c FROM interactions WHERE feedback = -1")
            down = int(cur.fetchone()[0])
            cur.execute("SELECT COUNT(1) AS c FROM interactions WHERE feedback IS NULL")
            none = int(cur.fetchone()[0])
        return {"total": total, "up": up, "down": down, "none": none}


__all__ = ["LocalDB"]


