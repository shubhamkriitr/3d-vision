class BasePipeline:
    def __init__(self, config = None, **kwargs) -> None:
        self._init_from_config(config)
        
    
    def _init_from_config(self, config):
        if config is None:
            self.config = {
               
            }
        else:
            self.config = config


if __name__ == "__main__":
    pass