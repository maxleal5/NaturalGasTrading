"""Alerting module — Slack and email notifications."""
import os
import logging
from datetime import datetime, timezone
from slack_sdk import WebhookClient
from slack_sdk.errors import SlackApiError

logger = logging.getLogger(__name__)


def send_slack_alert(message: str, webhook_url: str = None) -> bool:
    """Send a Slack notification via incoming webhook."""
    url = webhook_url or os.getenv("SLACK_WEBHOOK_URL")
    if not url:
        logger.warning("SLACK_WEBHOOK_URL not set; skipping Slack alert.")
        return False
    try:
        client = WebhookClient(url)
        resp = client.send(text=message)
        if resp.status_code == 200:
            return True
        logger.error("Slack alert failed: %s %s", resp.status_code, resp.body)
        return False
    except SlackApiError as exc:
        logger.error("SlackApiError: %s", exc)
        return False
    except Exception as exc:
        logger.error("Unexpected error sending Slack alert: %s", exc)
        return False


def format_dag_failure_alert(task_name: str, failure_reason: str, last_success: datetime = None) -> str:
    """Format a standard DAG failure alert message."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    last_ok = last_success.strftime("%Y-%m-%d %H:%M UTC") if last_success else "Unknown"
    return (
        f":red_circle: *NatGas Platform Alert* — `{task_name}` FAILED\n"
        f"*Time:* {ts}\n"
        f"*Reason:* {failure_reason}\n"
        f"*Last successful run:* {last_ok}"
    )


def format_model_drift_alert(model_name: str, region: str, lead_days: int,
                              residual_bias: float, z_score: float, consecutive: int) -> str:
    """Format a model drift / stability alert message."""
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f":warning: *Model Change Detected* — `{model_name}`\n"
        f"*Time:* {ts}\n"
        f"*Region:* {region} | *Lead:* {lead_days}d\n"
        f"*Residual bias:* {residual_bias:.4f} | *Z-score:* {z_score:.2f} | "
        f"*Consecutive violations:* {consecutive}\n"
        f"Action: MOS window narrowed to 7 days. Please inspect ECMWF/Google release notes."
    )
