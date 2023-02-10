from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from pythonfiles import tokenizer


def loadModelInitial():
    device = "cpu"
    model = Wav2Vec2ForCTC.from_pretrained(".\\nepalimodel\\model_0.1_dropout_5_10sec").to(device)
    processor = Wav2Vec2Processor.from_pretrained(".\\nepalimodel\\processor_0.1_dropout_5_10sec")
    return model, processor, device
    