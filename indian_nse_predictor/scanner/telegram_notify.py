"""Send Plotly-generated PNG via Telegram Bot API."""

from __future__ import annotations

import logging
from pathlib import Path

import requests

logger = logging.getLogger(__name__)


def send_photo_png(
    image_path: Path,
    caption: str,
    bot_token: str,
    chat_id: str,
    timeout: int = 60,
) -> bool:
    if not bot_token or not chat_id:
        logger.warning("TELEGRAM_BOT_TOKEN or TELEGRAM_CHAT_ID missing; skip send.")
        return False
    url = f"https://api.telegram.org/bot{bot_token}/sendPhoto"
    try:
        with image_path.open("rb") as photo:
            r = requests.post(
                url,
                data={"chat_id": chat_id, "caption": caption[:1024]},
                files={"photo": photo},
                timeout=timeout,
            )
        if r.status_code != 200:
            logger.error("Telegram API %s: %s", r.status_code, r.text[:500])
            return False
        return True
    except Exception as e:
        logger.exception("Telegram send failed: %s", e)
        return False
