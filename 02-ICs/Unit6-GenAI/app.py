from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, TextIteratorStreamer
import torch
from flask import Flask, render_template, request, Response, stream_with_context
import json
import threading

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_SYSTEM = """You are an advertising specialist for BikeEase, a company that sells and rents bikes. Given the bike specifications, discount information, and marketing theme, generate a compelling and intuitive advertisement text. Highlight the key features and benefits effectively.
"""

MODELS_LIST = [
    "EleutherAI/gpt-j-6B",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "facebook/opt-1.3B",
    "tiiuae/falcon-rw-1b",
    "microsoft/phi-2"
]

# Cache for loaded models to avoid reloading every time if the user picks the same one
# However, for a simple app, we might just load each time as requested by the prompt "model loads"
model_cache = {}

def get_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name]
    
    # load the model/tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    
    # Choose the correct model class based on the model name
    if "t5" in model_name.lower():
        model_class = AutoModelForSeq2SeqLM
    else:
        model_class = AutoModelForCausalLM
        
    model = model_class.from_pretrained(model_name,
                                        dtype=torch.float16 if device.type == "cuda" else torch.float32
                                        ).to(device).eval()
    model_cache[model_name] = (model, tok)
    return model, tok


@app.route("/", methods=["GET"])
def index():
    selected_model = MODELS_LIST[1]
    return render_template(
        "index.html",
        models=MODELS_LIST,
        selected_model=selected_model,
        specs="",
        discount="",
        theme="",
        advertisement=None,
        error=None
    )

@app.route("/generate")
def generate():
    model_name = request.args.get("model_name")
    specs = request.args.get("specs")
    discount = request.args.get("discount")
    theme = request.args.get("theme")

    def stream():
        try:
            yield f"data: {json.dumps({'status': 'Loading model...', 'progress': 10})}\n\n"
            model, tok = get_model(model_name)
            yield f"data: {json.dumps({'status': 'Model loaded. Preparing prompt...', 'progress': 40})}\n\n"

            prompt = f"""{PROMPT_SYSTEM}
    Specs: {specs} 
    Discount: {discount} 
    Theme: {theme}
    
    Write the advertisement text now.
    """
            inputs = tok(prompt, return_tensors="pt").to(device)
            input_ids = inputs["input_ids"]
            
            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            
            # For progress percentage, we can estimate based on max_new_tokens
            max_new_tokens = 150
            
            generation_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                eos_token_id=tok.eos_token_id,
                pad_token_id=tok.eos_token_id
            )

            # Run generation in a separate thread
            thread = threading.Thread(target=model.generate, kwargs=generation_kwargs)
            thread.start()

            yield f"data: {json.dumps({'status': 'Generating text...', 'progress': 50})}\n\n"

            generated_text = ""
            tokens_generated = 0
            for new_text in streamer:
                generated_text += new_text
                tokens_generated += 1
                # Progress from 50% to 100%
                progress = min(50 + int((tokens_generated / max_new_tokens) * 50), 99)
                yield f"data: {json.dumps({'status': 'Generating...', 'progress': progress, 'text': generated_text})}\n\n"

            yield f"data: {json.dumps({'status': 'Done!', 'progress': 100, 'text': generated_text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(stream()), mimetype="text/event-stream")


if __name__ == "__main__":
    app.run(debug=True)

