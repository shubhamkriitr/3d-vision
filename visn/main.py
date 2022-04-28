from argparse import ArgumentParser
from visn.process.pipeline import BasePipeline
from visn.utils import logger
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
        self.parser.add_argument("--pipeline", "-P", type=str, 
                                 default="BasePipeline")

    def handle(self):
        args = self.parser.parse_args()
        if args.test:
            print("="*80+"\n"
            +"Improving relative pose estimation by estimating gravity\n"
            +"="*80+"\n")
            return
        pipeline = None
        if args.pipeline == "BasePipeline":
            pipeline = BasePipeline() # TODO read and pass config to it
        
        outputs = pipeline.run()
        import pickle
        with open('outputs.pickle', 'wb') as handle:
            pickle.dump(outputs, handle, protocol=pickle.HIGHEST_PROTOCOL)
        

if __name__ == "__main__":
    handler = CommandLineHandlerV1()
    handler.handle()
