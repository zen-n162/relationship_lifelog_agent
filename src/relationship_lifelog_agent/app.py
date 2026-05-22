from __future__ import annotations

import argparse
import os

from relationship_lifelog_agent.config import gradio_launch_kwargs, load_config
from relationship_lifelog_agent.ui.chat_ui import build_chat_ui


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Relationship Lifelog Agent UI.")
    parser.add_argument("--config", default=None, help="Optional private config path.")
    parser.add_argument("--port", type=int, default=None, help="Override the local Gradio port for this run.")
    parser.add_argument("--smoke", action="store_true", help="Build the app without starting a server.")
    return parser


def main(argv: list[str] | None = None) -> None:
    os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
    args = build_arg_parser().parse_args(argv)
    settings = load_config(args.config)
    demo = build_chat_ui(settings)
    launch_kwargs = gradio_launch_kwargs(settings, port=args.port)
    if args.smoke:
        print(f"relationship_lifelog_agent UI built with share=False port={launch_kwargs['server_port']}")
        return
    demo.launch(**launch_kwargs)


if __name__ == "__main__":
    main()
