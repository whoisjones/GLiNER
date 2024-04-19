from transformers import PretrainedConfig


class VAEConfig(PretrainedConfig):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
