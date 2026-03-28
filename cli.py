import argparse
import sys
import os
import subprocess
import time
import structlog
import logging
import httpx
from urllib.parse import urlparse

# Set up structured logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format="%(message)s")
structlog.configure(
    processors=[
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory(),
    cache_logger_on_first_use=False,
)
logger = structlog.get_logger()


def start_lmstudio_server():
    logger.info("starting_lmstudio_server")
    try:
        return subprocess.Popen(["lms", "server", "start"])
    except FileNotFoundError as exc:
        raise RuntimeError("LM Studio CLI `lms` was not found in PATH.") from exc


def run_proxy_server():
    logger.info("launching_headless_bridge")
    import uvicorn
    from proxy import app

    raw_host = os.getenv("BRIDGE_HOST", "127.0.0.1")
    if "://" in raw_host:
        host = urlparse(raw_host).hostname or "127.0.0.1"
    else:
        host = raw_host.split(":", 1)[0] or "127.0.0.1"

    raw_port = os.getenv("BRIDGE_PORT", "8080")
    try:
        port = int(raw_port)
    except Exception:
        if "://" in raw_port:
            port = urlparse(raw_port).port or 8080
        else:
            try:
                port = int(raw_port.rsplit(":", 1)[1])
            except Exception:
                port = 8080
    uvicorn.run(app, host=host, port=port)


def wait_for_lmstudio_server(timeout_seconds: int = 30):
    base_url = os.getenv("LMSTUDIO_BASE_URL", "http://").rstrip("/")
    deadline = time.time() + timeout_seconds
    last_error = None
    while time.time() < deadline:
        try:
            response = httpx.get(f"{base_url}/api/v1/models", timeout=2.0)
            if response.status_code < 500:
                logger.info("lmstudio_server_ready", base_url=base_url)
                return
        except Exception as exc:
            last_error = exc
        time.sleep(1)
    raise RuntimeError(f"LM Studio server did not become ready at {base_url}: {last_error}")


def load_model_via_bridge(model_id: str):
    import asyncio
    from proxy import proxy

    async def _load():
        await proxy.bridge.load_model(model_id)

    asyncio.run(_load())
    logger.info("bridge_model_loaded_from_cli", model=model_id)

def main():
    parser = argparse.ArgumentParser(description="LM Studio bridge, tuner, and config tools")
    parser.add_argument("--tui", action="store_true", help="Launch the Textual TUI interface")
    parser.add_argument("--cli", action="store_true", help="Launch basic CLI (Legacy)")
    parser.add_argument("--serve-proxy", action="store_true", help="Run the headless FastAPI bridge/proxy server")
    parser.add_argument("--start-lms", action="store_true", help="Start the LM Studio local server with the lms CLI")
    parser.add_argument("--load-model", type=str, help="Load a model through the bridge without opening the GUI")
    parser.add_argument("--debug", action="store_true", help="Enable structured debug logging")
    args = parser.parse_args()

    if args.debug:
        logging.basicConfig(filename="lmstudio-config-v3.log", level=logging.DEBUG)
        structlog.configure(
            processors=[
                structlog.processors.add_log_level,
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.JSONRenderer()
            ],
            wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        )

    if args.start_lms:
        start_lmstudio_server()
        wait_for_lmstudio_server()

    if args.load_model:
        load_model_via_bridge(args.load_model)

    if args.serve_proxy:
        run_proxy_server()
    elif args.tui:
        logger.info("launching_tui_interface")
        from tui import LMStudioConfigApp
        app = LMStudioConfigApp()
        app.run()
    elif args.cli:
        logger.info("launching_legacy_cli")
        print("Legacy CLI mode is deprecated. Use --tui or launch without flags for GUI.")
    else:
        logger.info("launching_modern_gui")
        try:
            from gui import LMStudioConfigGUI
            app = LMStudioConfigGUI()
            app.mainloop()
        except ImportError as e:
            logger.error("gui_launch_failed", error=str(e))
            print("\nError: Could not launch GUI. Please install requirements: pip install -r requirements.txt")
            print("Alternatively, launch the TUI with: python cli.py --tui\n")

if __name__ == "__main__":
    main()
