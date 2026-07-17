"""Smoke tests for the CLI."""

import pytest
from ecal.cli import main


class TestCLI:
    def test_version(self, capsys):
        with pytest.raises(SystemExit, match="0"):
            main(["--version"])
        captured = capsys.readouterr()
        assert "ecal 0.1.0" in captured.out

    def test_estimate_mlp(self, capsys):
        main(["estimate", "--model", "MLP", "--layers", "3", "--epochs", "5",
              "--samples", "100", "--inferences", "100"])
        captured = capsys.readouterr()
        assert "eCAL:" in captured.out

    def test_estimate_json(self, capsys):
        main(["estimate", "--model", "MLP", "--layers", "3", "--epochs", "5",
              "--samples", "100", "--json"])
        import json
        captured = capsys.readouterr()
        result = json.loads(captured.out)
        assert "total" in result

    def test_estimate_with_hardware(self, capsys):
        main(["estimate", "--model", "MLP", "--layers", "3", "--epochs", "5",
              "--samples", "100", "--hardware", "generic_cpu"])
        captured = capsys.readouterr()
        assert "eCAL:" in captured.out

    def test_profiles(self, capsys):
        main(["profiles"])
        captured = capsys.readouterr()
        assert "generic_cpu" in captured.out
        assert "apple_m2" in captured.out

    def test_no_command_exits(self):
        with pytest.raises(SystemExit, match="1"):
            main([])
