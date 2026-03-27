import argparse
import sys
import structlog
import logging

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

def main():
    parser = argparse.ArgumentParser(description="LM Studio Config Wizard - V3")
    parser.add_argument("--tui", action="store_true", help="Launch the Textual TUI interface")
    parser.add_argument("--cli", action="store_true", help="Launch basic CLI (Legacy)")
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

    if args.tui:
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
