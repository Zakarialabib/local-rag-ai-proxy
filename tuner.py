import asyncio
import httpx
import structlog
import time
from typing import Dict, Any, List

logger = structlog.get_logger()

class AdaptiveTuner:
    """Lazy metrics collector — updates only on demand, not on a timer loop."""

    def __init__(self, lmstudio_base: str = "http://localhost:1234"):
        self.lmstudio_base = lmstudio_base
        self.last_stats: Dict[str, Any] = {}
        self._poll_interval = 30
        self._last_poll = 0.0

    async def poll_once(self) -> Dict[str, Any]:
        """Call after a chat completion to snapshot real metrics."""
        now = time.time()
        if now - self._last_poll < self._poll_interval:
            return self.last_stats
        self._last_poll = now
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"{self.lmstudio_base}/api/v1/models")
                if resp.status_code == 200:
                    models = resp.json()
                    loaded = [
                        m for m in (models.get("data", []) if isinstance(models, dict) else models)
                        if m.get("state") == "loaded"
                    ]
                    self.last_stats = {
                        "loaded_models": [m.get("id") for m in loaded],
                        "count": len(loaded),
                        "models": models,
                    }
                    logger.info("tuner_poll", count=len(loaded), models=[m.get("id") for m in loaded])
        except Exception as e:
            logger.error("tuner_poll_error", error=str(e))
        return self.last_stats

    async def monitor_and_tune(self):
        """Run once on startup; subsequent updates are lazy (per-request)."""
        logger.info("adaptive_tuner_started", base_url=self.lmstudio_base)
        await self.poll_once()

if __name__ == "__main__":
    tuner = AdaptiveTuner()
    asyncio.run(tuner.monitor_and_tune())
