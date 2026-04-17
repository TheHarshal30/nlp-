class BaseEmbedder:
    def load(self, model_path):
        raise NotImplementedError

    def encode(self, texts, batch_size=32):
        raise NotImplementedError

    @property
    def name(self):
        return self.__class__.__name__
