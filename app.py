from potassium import Potassium, Request, Response
from vllm import LLM, SamplingParams

MODEL_NAME_OR_PATH = "TheBloke/Llama-2-13B-chat-AWQ"

app = Potassium("Llama2-13B-Chat-AWQ")


@app.init
def init() -> dict:
    """Initialize the application with the model and tokenizer."""
    llm = LLM(model=MODEL_NAME_OR_PATH, quantization="awq")

    return {
        "llm": llm,
    }


@app.handler()
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate text from a prompt."""
    llm = context.get("llm")
    max_new_tokens = request.json.get("max_new_tokens", 512)
    temperature = request.json.get("temperature", 0.7)
    prompt = request.json.get("prompt")
    prompt_template = f'''{prompt}
    '''

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        max_tokens=max_new_tokens
    )

    result = llm.generate_text(
        prompt=prompt_template,
        sampling_params=sampling_params
    )

    return Response(json={"outputs": result[0].outputs[0].text}, status=200)


if __name__ == "__main__":
    app.serve()
