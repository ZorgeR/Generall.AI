"""Authorization state: allow-list, block-list and invite codes.

Persisted to ``data/userlist.json`` with the same shape as before::

    {"users": [...], "blocked_users": [...], "invites": {inviter: {code: {created_at, used_by}}}}
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from datetime import datetime
from typing import Iterable

from bot.config import config

logger = logging.getLogger(__name__)


class AuthStore:
    def __init__(
        self,
        path: str,
        base_chat_ids: Iterable[str] = (),
        admin_id: str | None = None,
        allow_all: bool = False,
    ) -> None:
        self.path = path
        self.base_chat_ids = set(base_chat_ids)
        self.admin_id = admin_id
        self.allow_all = allow_all
        self.authorized: set[str] = set(self.base_chat_ids)
        self.blocked: set[str] = set()
        self.invites: dict[str, dict[str, dict]] = {}

    # ---- persistence -----------------------------------------------------
    def load(self) -> None:
        try:
            if os.path.exists(self.path):
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.authorized = set(self.base_chat_ids) | set(map(str, data.get("users", [])))
                self.blocked = set(map(str, data.get("blocked_users", [])))
                self.invites = data.get("invites", {}) or {}
                logger.info(
                    "Loaded %d authorized users, %d blocked from %s",
                    len(self.authorized), len(self.blocked), self.path,
                )
            else:
                self.save()
                logger.info("Created new %s", self.path)
        except Exception as e:  # noqa: BLE001
            logger.error("Error loading authorized users: %s", e)

    def save(self) -> None:
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            with open(self.path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "users": sorted(self.authorized),
                        "blocked_users": sorted(self.blocked),
                        "invites": self.invites,
                    },
                    f,
                    indent=2,
                )
        except Exception as e:  # noqa: BLE001
            logger.error("Error saving authorized users: %s", e)

    # ---- queries ---------------------------------------------------------
    def is_admin(self, user_id: str) -> bool:
        return bool(self.admin_id) and str(user_id) == self.admin_id

    def is_authorized(self, user_id: str) -> bool:
        user_id = str(user_id)
        if user_id in self.blocked:
            return False
        if self.allow_all or self.is_admin(user_id):
            return True
        return user_id in self.authorized

    # ---- mutations -------------------------------------------------------
    def authorize(self, user_id: str) -> None:
        self.authorized.add(str(user_id))
        self.save()

    def block(self, user_id: str) -> None:
        user_id = str(user_id)
        self.authorized.discard(user_id)
        self.blocked.add(user_id)
        self.save()

    def unblock(self, user_id: str) -> None:
        user_id = str(user_id)
        self.blocked.discard(user_id)
        self.authorized.add(user_id)
        self.save()

    # ---- invites ---------------------------------------------------------
    def generate_invite(self, user_id: str) -> str:
        code = uuid.uuid4().hex[:8]
        self.invites.setdefault(str(user_id), {})[code] = {
            "created_at": datetime.now().isoformat(),
            "used_by": None,
        }
        self.save()
        return code

    def unused_invite_count(self, user_id: str) -> int:
        return sum(1 for inv in self.invites.get(str(user_id), {}).values() if inv.get("used_by") is None)

    def total_unused_invites(self) -> int:
        return sum(
            1 for invites in self.invites.values() for inv in invites.values() if inv.get("used_by") is None
        )

    def find_invite(self, code: str) -> str | None:
        """Return the inviter id for an unused code, else None."""
        for inviter, invites in self.invites.items():
            if code in invites and invites[code].get("used_by") is None:
                return inviter
        return None

    def use_invite(self, code: str, used_by: str) -> str | None:
        inviter = self.find_invite(code)
        if inviter is None:
            return None
        self.invites[inviter][code]["used_by"] = str(used_by)
        self.authorized.add(str(used_by))
        self.save()
        return inviter


auth = AuthStore(
    path=os.path.join(config.data_dir, "userlist.json"),
    base_chat_ids=config.chat_ids,
    admin_id=config.admin_id,
    allow_all=config.allow_all_users,
)
