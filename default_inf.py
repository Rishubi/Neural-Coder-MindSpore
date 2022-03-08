from model import GPTJForCausalLM, GPTJConfig
import mindspore
import mindspore.numpy as np
from mindspore import context

if __name__ == "__main__":
    print("loading model")
    model_ms = GPTJForCausalLM(GPTJConfig())
    print("model loaded")
    ids = np.arange(2011).astype(mindspore.int32)
    print("begin inference")
    out = model_ms(ids)
    print(out)