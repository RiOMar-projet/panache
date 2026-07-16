"""Tests for panache.cli — covers build_parser and main."""
from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from panache.cli import build_parser, main


class BuildParserTests(unittest.TestCase):

    def test_returns_argument_parser(self):
        self.assertIsInstance(build_parser(), argparse.ArgumentParser)

    def test_config_argument_parsed(self):
        args = build_parser().parse_args(["myconfig.json"])
        self.assertEqual(args.config, "myconfig.json")


class MainTests(unittest.TestCase):

    def test_main_calls_load_run_config_and_run_batch(self):
        mock_cfg = MagicMock()
        mock_path = Path("/fake/Results.csv")
        with patch("panache.cli.load_run_config", return_value=mock_cfg) as mock_load, \
             patch("panache.cli.run_batch", return_value=mock_path) as mock_run, \
             patch("sys.argv", ["panache", "myconfig.json"]):
            main()
        mock_load.assert_called_once()
        mock_run.assert_called_once_with(mock_cfg)


if __name__ == "__main__":
    unittest.main()
