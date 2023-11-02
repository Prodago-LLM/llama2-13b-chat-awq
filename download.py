from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

MODEL_NAME_OR_PATH = "TheBloke/Llama-2-13B-chat-AWQ"
DEVICE = "cuda:0"


def download_model() -> tuple:
    """Download the model and tokenizer."""
    model = AutoAWQForCausalLM.from_quantized(model_name_or_path, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
    return model, tokenizer


if __name__ == "__main__":
    download_model()
