from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, TextIteratorStreamer
import torch
from flask import Flask, render_template, request, Response, stream_with_context
import json
import threading
from datetime import datetime
import os
from pathlib import Path

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_SYSTEM = """
You are an advertising specialist for BikeEase, a company that sells and rents bikes. 
Given the bike specifications, discount information, and marketing theme, 
generate a compelling and intuitive advertisement text. Highlight the key features and benefits effectively.
Return only these sections exactly once, in order, no extra text:\n
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

REPORT_DIR = Path(__file__).parent / 'reports'
REPORT_PATH = REPORT_DIR / 'last_report.json'

def get_model(model_name):
    if model_name in model_cache:
        return model_cache[model_name]
    
    # load the model/tokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok is None:
        # Defensive: if tokenizer failed to load, surface an explicit error
        raise RuntimeError(f"Tokenizer failed to load for model: {model_name}")

    # Choose the correct model class based on the model name
    if "t5" in model_name.lower():
        model_class = AutoModelForSeq2SeqLM
        # For T5, we might need to load in float32 for stability on CPU
        model = model_class.from_pretrained(model_name).to(device).eval()
    elif "gpt-j" in model_name.lower():
        model_class = AutoModelForCausalLM
        # GPT-J-6B is huge, use 8-bit or half precision if possible, but on CPU we are limited.
        # Let's try to at least use low_cpu_mem_usage.
        model = model_class.from_pretrained(model_name,
                                            dtype=torch.float32 if device.type == "cpu" else torch.float16,
                                            low_cpu_mem_usage=True
                                            ).to(device).eval()
    else:
        model_class = AutoModelForCausalLM
        model = model_class.from_pretrained(model_name,
                                            dtype=torch.float16 if device.type == "cuda" else torch.float32,
                                            low_cpu_mem_usage=True
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
    Website url  
    
    Write the advertisement text now.
    """
            # Use Chat template for TinyLlama
            if "tinyllama" in model_name.lower():
                prompt = f"<|system|>\n{PROMPT_SYSTEM}</s>\n<|user|>\nSpecs: {specs}\nDiscount: {discount}\nTheme: {theme}</s>\n<|assistant|>\n"
            elif "phi-2" in model_name.lower():
                prompt = f"Instruct: {PROMPT_SYSTEM}\nSpecs: {specs}\nDiscount: {discount}\nTheme: {theme}\nOutput:"
            elif "flan-t5" in model_name.lower():
                prompt = f"{PROMPT_SYSTEM}\nSpecs: {specs}\nDiscount: {discount}\nTheme: {theme}"
            
            inputs = tok(prompt, return_tensors="pt")
            if inputs is None:
                raise RuntimeError("Tokenizer returned no inputs for the prompt")
            inputs = inputs.to(device)
            input_ids = inputs["input_ids"]
            
            # Use local pad/eos ids (defensive) to avoid modifying tokenizer object and silence static checks
            pad_token_id = getattr(tok, 'pad_token_id', None)
            eos_token_id = getattr(tok, 'eos_token_id', None)
            if pad_token_id is None:
                pad_token_id = eos_token_id if eos_token_id is not None else 0
            if eos_token_id is None:
                eos_token_id = pad_token_id

            streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)
            
            # Increased max_new_tokens to prevent early cutoff
            max_new_tokens = 512
            
            generation_kwargs = dict(
                input_ids=input_ids,
                streamer=streamer,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.15,
                no_repeat_ngram_size=4,
                pad_token_id=pad_token_id,
                eos_token_id=eos_token_id
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
                if new_text.strip() or tokens_generated % 5 == 0:
                    yield f"data: {json.dumps({'status': 'Generating...', 'progress': progress, 'text': generated_text})}\n\n"

            if not generated_text:
                yield f"data: {json.dumps({'error': 'Model returned empty output. Try a different model or prompt.'})}\n\n"
            else:
                # Build a simple summary JSON that other parts of the app can read
                try:
                    REPORT_DIR.mkdir(parents=True, exist_ok=True)
                    summary = {
                        'report_title': f'Run Summary - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
                        'report_subtitle': 'Automatically generated after model run',
                        'report_date': datetime.now().strftime('%B %d, %Y'),
                        'author': 'AutoGenerator',
                        'executive_summary': (generated_text[:1000] + '...') if len(generated_text) > 1000 else generated_text,
                        'findings': [],
                        'recommendations': [],
                        'notes': f'Generated {len(generated_text)} characters; tokens_estimate={tokens_generated}',
                        'metrics': {
                            'Chars': str(len(generated_text)),
                            'Tokens_Est': str(tokens_generated)
                        },
                        'summary_table': [
                            {'label': 'Generated Chars', 'value': str(len(generated_text))},
                            {'label': 'Token Estimate', 'value': str(tokens_generated)}
                        ],
                        'version': '1.0',
                        'generated_text': generated_text
                    }

                    # Add recommendations based on the ratio of chars to tokens
                    try:
                        tokens_val = float(max_new_tokens)/float(tokens_generated) if tokens_generated else 0.0
                        ratio = (len(generated_text) / tokens_val) if tokens_val > 0 else 0.0
                    except Exception:
                        ratio = 0.0

                    # Assumption: interpret thresholds as follows:
                    # - ratio >= 3 => very good model
                    # - 2 <= ratio < 3 => moderate model
                    # - ratio < 2 => not recommended
                    if ratio >= 3.0:
                        rec = (
                            "Model quality appears very good (chars/tokens ratio >= 3). "
                            "Consider this model for running extended text generations and more derivations."
                        )
                    elif ratio >= 2.0:
                        rec = (
                            "Model quality appears moderate (chars/tokens ratio between 2 and 3). "
                            "Consider additional tuning (e.g., more data, hyperparameter tweaks) before production use."
                        )
                    else:
                        rec = (
                            "Model is not recommended (chars/tokens ratio < 2). "
                            "Do not use for real world production use-case; review model configuration, prompt, or Tokenizer settings."
                        )

                    # Insert the recommendation and a short quality note into the summary
                    summary['recommendations'] = [rec]
                    summary['metrics']['Chars_per_Token'] = f"{ratio:.2f}"

                    # Simple chart config: show token/char measures
                    chart_config = {
                        'type': 'bar',
                        'data': {
                            'labels': ['Chars', 'Tokens_Est'],
                            'datasets': [{
                                'label': 'Run metrics',
                                'data': [len(generated_text), tokens_generated],
                                'backgroundColor': ['#4e79a7', '#f28e2b']
                            }]
                        },
                        'options': {'responsive': True}
                    }

                    # Save JSON including chart config (atomic write)
                    out = dict(summary=summary, chart_config=chart_config)
                    temp_path = REPORT_PATH.with_suffix('.tmp')
                    with open(temp_path, 'w', encoding='utf-8') as fh:
                        json.dump(out, fh, ensure_ascii=False, indent=2)
                    # Atomic replace
                    os.replace(str(temp_path), str(REPORT_PATH))
                except Exception as e:
                    # If writing the report fails, continue but include an error in the stream
                    yield f"data: {json.dumps({'status': 'Warning: failed to write report', 'error': str(e)})}\n\n"

                yield f"data: {json.dumps({'status': 'Done!', 'progress': 100, 'text': generated_text})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return Response(stream_with_context(stream()), mimetype="text/event-stream")


# render the polished summary report template
@app.route('/report')
def report_demo():
    # If an auto-generated report JSON exists, read it and use that data
    report_data = None
    try:
        if REPORT_PATH.exists():
            with open(REPORT_PATH, 'r', encoding='utf-8') as fh:
                obj = json.load(fh)
                # Expecting {'summary': {...}, 'chart_config': {...}}
                report_data = obj.get('summary', None)
                chart_config = obj.get('chart_config', None)
                if chart_config is not None:
                    metrics_chart_json = json.dumps(chart_config)
                else:
                    metrics_chart_json = None
        else:
            report_data = None
            metrics_chart_json = None
    except Exception:
        report_data = None
        metrics_chart_json = None

    if report_data is None:
        # Sample fallback data to demonstrate the template
        sample_data = {
            'report_title': 'Model Execution Report Summary',
            'report_subtitle': 'Model run summary generated for demonstration purposes',
            'report_date': datetime.now().strftime('%B %d, %Y'),
            'author': 'Automated Report System',
            'executive_summary': 'This report summarizes the recent model run and highlights the primary findings, key metrics, and recommended next steps for stakeholders.',
            'findings': [
                {'title': 'Improved Accuracy', 'detail': 'Validation accuracy increased by 4.3% compared to previous run.'},
                {'title': 'Lower Latency', 'detail': 'Average inference latency reduced by 18%.'}
            ],
            'recommendations': [
                'Deploy new model to staging and monitor memory usage.',
                'Increase batch size for throughput improvements where latency budget allows.',
                'Collect more labeled samples for edge cases to improve robustness.'
            ],
            'notes': 'All experiments were run on a mixed GPU/CPU environment. See logs for per-run details.',
            'metrics': {'Accuracy': '92.3%', 'Throughput': '480 req/s', 'Latency (p95)': '135 ms'},
            'summary_table': [
                {'label': 'Total Runs', 'value': '42'},
                {'label': 'Successful', 'value': '40'},
                {'label': 'Failed', 'value': '2'}
            ],
            'version': '1.0'
        }
        return render_template('model_report.html', **sample_data, metrics_chart_json=json.dumps(chart_config) if 'chart_config' in locals() else None)
    else:
        # Use the generated report data
        return render_template('model_report.html', **report_data, metrics_chart_json=metrics_chart_json)


@app.route("/summary")
def summary():
    return render_template("summary.html")

if __name__ == "__main__":
    app.run(debug=True)
