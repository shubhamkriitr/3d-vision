from argparse import ArgumentParser
from xmlrpc.client import boolean

class BaseCommandLineHandler:
    def __init__(self, name="main") -> None:
        self.name = name

    def handle(self):
        raise NotImplementedError()

class CommandLineHandlerV1(BaseCommandLineHandler):
    def __init__(self, name="main") -> None:
        super().__init__(name)
        self.parser = ArgumentParser()
        self.parser.add_argument("--test", action="store_true")

    def handle(self):
        args = self.parser.parse_args()
        if args.test:
            print("="*80+"\n"
            +"Improving relative pose estimation by estimating gravity\n"
            +"="*80+"\n")

if __name__ == "__main__":
    handler = CommandLineHandlerV1()
    handler.handle()
