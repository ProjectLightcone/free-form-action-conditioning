from __future__ import annotations

import sqlite3
from pathlib import Path


class DedupeIndex:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._create()

    def close(self) -> None:
        self.conn.close()

    def upsert_scene(self, scene_hash: str, scene_id: int, split: str) -> int:
        self.conn.execute(
            "insert or ignore into scenes(scene_hash, scene_id, split) values (?, ?, ?)",
            (scene_hash, scene_id, split),
        )
        self.conn.commit()
        return int(self.conn.execute("select scene_id from scenes where scene_hash = ?", (scene_hash,)).fetchone()[0])

    def upsert_event(self, event_hash: str, event_id: int, scene_hash: str) -> int:
        self.conn.execute(
            "insert or ignore into events(event_hash, event_id, scene_hash) values (?, ?, ?)",
            (event_hash, event_id, scene_hash),
        )
        self.conn.commit()
        return int(self.conn.execute("select event_id from events where event_hash = ?", (event_hash,)).fetchone()[0])

    def upsert_outcome(self, outcome_hash: str, outcome_id: int, event_hash: str) -> int:
        self.conn.execute(
            "insert or ignore into outcomes(outcome_hash, outcome_id, event_hash) values (?, ?, ?)",
            (outcome_hash, outcome_id, event_hash),
        )
        self.conn.commit()
        return int(self.conn.execute("select outcome_id from outcomes where outcome_hash = ?", (outcome_hash,)).fetchone()[0])

    def upsert_rendering(
        self,
        rendering_hash: str,
        condition_id: int,
        event_hash: str,
        family: str,
        template_id: str,
    ) -> int:
        self.conn.execute(
            """
            insert or ignore into renderings(rendering_hash, condition_id, event_hash, family, template_id)
            values (?, ?, ?, ?, ?)
            """,
            (rendering_hash, condition_id, event_hash, family, template_id),
        )
        self.conn.commit()
        return int(
            self.conn.execute("select condition_id from renderings where rendering_hash = ?", (rendering_hash,)).fetchone()[0]
        )

    def _create(self) -> None:
        self.conn.executescript(
            """
            create table if not exists scenes(
              scene_hash text primary key,
              scene_id integer not null,
              split text not null
            );
            create table if not exists events(
              event_hash text primary key,
              event_id integer not null,
              scene_hash text not null
            );
            create table if not exists outcomes(
              outcome_hash text primary key,
              outcome_id integer not null,
              event_hash text not null
            );
            create table if not exists renderings(
              rendering_hash text primary key,
              condition_id integer not null,
              event_hash text not null,
              family text not null,
              template_id text not null
            );
            """
        )
        self.conn.commit()


def split_from_hash(identity: str, val_fraction: float, salt: str = "scene-split-v1") -> str:
    bucket = int(identity[:12], 16) / float(0xFFFFFFFFFFFF)
    salted = (bucket + (int.from_bytes(salt.encode("utf-8"), "little") % 10_000) / 10_000.0) % 1.0
    return "val" if salted < val_fraction else "train"
