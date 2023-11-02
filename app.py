from potassium import Potassium, Request, Response
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

MODEL_NAME_OR_PATH = "TheBloke/Llama-2-13B-chat-AWQ"
DEVICE = "cuda:0"

app = Potassium("Llama2-13B-Chat-AWQ")


@app.init
def init() -> dict:
    """Initialize the application with the model and tokenizer."""
    model = AutoAWQForCausalLM.from_quantized(MODEL_NAME_OR_PATH, fuse_layers=True,
                                              trust_remote_code=False, safetensors=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME_OR_PATH, trust_remote_code=False)

    return {
        "model": model,
        "tokenizer": tokenizer
    }


@app.handler()
def handler(context: dict, request: Request) -> Response:
    """Handle a request to generate text from a prompt."""
    model = context.get("model")
    tokenizer = context.get("tokenizer")
    max_new_tokens = request.json.get("max_new_tokens", 512)
    temperature = request.json.get("temperature", 0.7)
    prompt = request.json.get("prompt")
    prompt_template = f'''{prompt}
    '''
    # input_ids = tokenizer(prompt_template, return_tensors='pt').input_ids.cuda()

    # output = model.generate(inputs=input_ids,
    #                         temperature=temperature,
    #                         do_sample=True,
    #                         top_p=0.95,
    #                         top_k=40,
    #                         max_new_tokens=max_new_tokens)
    # result = tokenizer.decode(output[0])

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        repetition_penalty=1.1
    )

    result = pipe(prompt_template, return_full_text=False)[0]['generated_text']
    return Response(json={"outputs": result}, status=200)


if __name__ == "__main__":
    app.serve()
