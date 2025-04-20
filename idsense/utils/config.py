import argparse
import logging
import os
import sys

from .io import load_json


def _parse_cli_arguments():
    """Parses command-line arguments for the application."""

    parser = argparse.ArgumentParser(description="Digital Signage CLI")

    parser.add_argument(
        "--config", type=str, default="config.json", help="Path to configuration file"
    )

    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    parser.add_argument("--model", choices=["imdb"], help="Model to use")

    return parser.parse_args()


class Config:
    _inner = None

    @staticmethod
    def load(default_config_path="config.default.json"):
        """Loads configuration from a default and user-provided config file."""

        args = _parse_cli_arguments()
        if not os.path.exists(default_config_path):
            logging.error(
                f"Default configuration file not found: {default_config_path}"
            )
            sys.exit(1)
        Config._inner = load_json(default_config_path)
        if os.path.exists(args.config):
            user_config = load_json(args.config)
            Config._inner.update(user_config)
        else:
            logging.warning(f"Config file `{args.config}` not found. Using default.")
        if args.model:
            Config._inner["model"] = args.model
        Config._inner["debug"] = args.debug

    @staticmethod
    def get(key=None):
        """Retrieves the value for a specified key from the loaded configuration."""

        if Config._inner is None:
            raise ValueError(
                "Configuration not initialized. Please call `Config.load()` first."
            )
        if key:
            return Config._inner[key]
        return Config._inner
