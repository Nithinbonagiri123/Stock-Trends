"""
Telegram (requests) and Email (smtplib) notifications for BUY signals.
"""

from __future__ import annotations

import logging
import smtplib
from email.mime.text import MIMEText

import requests

from .config_auto import (
    EMAIL_FROM,
    EMAIL_SMTP_HOST,
    EMAIL_SMTP_PASSWORD,
    EMAIL_SMTP_PORT,
    EMAIL_SMTP_USER,
    EMAIL_TO,
    TELEGRAM_BOT_TOKEN,
    TELEGRAM_CHAT_ID,
)

logger = logging.getLogger(__name__)


def send_telegram(message: str, timeout: float = 30.0) -> bool:
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        logger.warning("Telegram not configured (TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID).")
        return False
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    try:
        r = requests.post(
            url,
            json={"chat_id": TELEGRAM_CHAT_ID, "text": message[:4000]},
            timeout=timeout,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


def send_email(subject: str, body: str, timeout: float = 30.0) -> bool:
    if not all([EMAIL_SMTP_HOST, EMAIL_SMTP_USER, EMAIL_FROM, EMAIL_TO]):
        logger.warning("Email not fully configured; skipping SMTP send.")
        return False
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = EMAIL_FROM
    msg["To"] = EMAIL_TO
    try:
        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT, timeout=timeout) as smtp:
            smtp.starttls()
            if EMAIL_SMTP_PASSWORD:
                smtp.login(EMAIL_SMTP_USER, EMAIL_SMTP_PASSWORD)
            smtp.sendmail(EMAIL_FROM, [EMAIL_TO], msg.as_string())
        return True
    except Exception as e:
        logger.error("Email send failed: %s", e)
        return False


def alert_buy_signal(
    ticker: str,
    predicted_price: float,
    rsi: float,
    predicted_return: float,
    confidence: float,
) -> None:
    text = (
        f"BUY signal (automated)\n"
        f"Ticker: {ticker}\n"
        f"Predicted T+1 price: {predicted_price:.4f}\n"
        f"Predicted return: {predicted_return * 100:.2f}%\n"
        f"RSI: {rsi:.2f}\n"
        f"Confidence (heuristic): {confidence:.1f}/100\n"
    )
    send_telegram(text)
    send_email(f"[Quant] BUY {ticker}", text)
