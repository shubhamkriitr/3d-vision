import pytest
from visn.main import CommandLineHandlerV1

class TestCommandLineHandlerV1:
    def test_handle(self):
        assert hasattr(CommandLineHandlerV1(), "name")