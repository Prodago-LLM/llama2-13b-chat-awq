from vllm import LLM

MODEL_NAME_OR_PATH = "TheBloke/Llama-2-13B-chat-AWQ"


def download_model() -> tuple:
    """Download the model and tokenizer."""
    llm = LLM(model=MODEL_NAME_OR_PATH, quantization="awq")
    return llm


if __name__ == "__main__":
    download_model()
