from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, TextIteratorStreamer, pipeline
import torch
from flask import Flask, render_template, request, Response, stream_with_context
import json
import threading
from datetime import datetime
import os
from pathlib import Path
from collections import OrderedDict

from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import BaseCallbackHandler

app = Flask(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PROMPT_SYSTEM = """
You are an advertising specialist for BikeEase, a company that sells and rents bikes. 
Given the bike specifications, discount information, and marketing theme, 
generate a compelling and intuitive advertisement text. Highlight the key features and benefits effectively.
Return only these sections exactly once, in order, no extra text:\n
"""

MODELS_LIST = [
    "EleutherAI/gpt-neo-125M",
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "google/flan-t5-base",
    "google/flan-t5-large",
    "facebook/opt-125m",
    "tiiuae/falcon-rw-1b",
    "microsoft/phi-2"
]

# Cache for loaded models to avoid reloading every time if the user picks the same one
# Use an OrderedDict to implement a simple LRU eviction policy and per-model locks
CACHE_MAX = 4
model_cache = OrderedDict()  # model_name -> {'llm': ..., 'lock': threading.Lock()}
cache_lock = threading.Lock()

REPORT_DIR = Path(__file__).parent / 'reports'
REPORT_PATH = REPORT_DIR / 'last_report.json'


def get_llm(model_name, force_reload: bool = False):
    """Load and cache LangChain LLM in a thread-safe way.

    Returns HuggingFacePipeline instance.
    If `force_reload` is True the model will be reloaded.
    """
    if not model_name:
        raise ValueError("model_name is required")

    # cached data: return cached LLM if present (and not force_reload)
    with cache_lock:
        if not force_reload and model_name in model_cache:
            entry = model_cache.pop(model_name)
            # move to the end to mark as recently used
            model_cache[model_name] = entry
            if entry.get('llm') is not None:
                return entry['llm']
        else:
            # create an empty cache entry with lock
            entry = {'llm': None, 'lock': threading.Lock()}
            model_cache[model_name] = entry
            # enforce cache size
            if len(model_cache) > CACHE_MAX:
                # pop the least recently used item
                try:
                    model_cache.popitem(last=False)
                except Exception as ex:
                    pass

    # Acquire the per-model lock while we perform the potentially expensive load
    with entry['lock']:
        # Another thread may have finished loading while we waited for the lock
        if entry.get('llm') is not None and not force_reload:
            return entry['llm']

        # load the tokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        if tok is None:
            raise RuntimeError(f"Tokenizer failed to load for model: {model_name}")

        # Choose the correct model class based on the model name
        if "t5" in model_name.lower():
            task = "text2text-generation"
            model_class = AutoModelForSeq2SeqLM
            model = model_class.from_pretrained(model_name).to(device).eval()
        else:
            task = "text-generation"
            model_class = AutoModelForCausalLM
            if "gpt-j" in model_name.lower():
                dtype = torch.float32 if device.type == "cpu" else torch.float16
            else:
                dtype = torch.float16 if device.type == "cuda" else torch.float32

            model = model_class.from_pretrained(model_name,
                                                torch_dtype=dtype,
                                                low_cpu_mem_usage=True
                                                ).to(device).eval()

        pipe = pipeline(
            task,
            model=model,
            tokenizer=tok,
            device=device if device.type == "cuda" else -1,
            max_new_tokens=512,
            do_sample=True,
            temperature=0.7,
            top_p=0.95,
        )

        llm = HuggingFacePipeline(pipeline=pipe)

        # store into the cache entry
        entry['llm'] = llm

    # Ensure the entry is marked as recently used (move to end)
    with cache_lock:
        if model_name in model_cache:
            model_cache.pop(model_name)
        model_cache[model_name] = entry

    return llm


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
    ad_prompt = PROMPT_SYSTEM if request.args.get("ad_prompt") is None else request.args.get("ad_prompt")

    def stream():
        class StreamHandler(BaseCallbackHandler):
            def __init__(self):
                self.generated_text = ""
                self.tokens_generated = 0

            def on_llm_new_token(self, token: str, **kwargs) -> None:
                self.generated_text += token
                self.tokens_generated += 1
                progress = min(50 + int((self.tokens_generated / 512) * 50), 99)
                # Note: We can't yield from here as it's a callback, but we can update a shared state
                # However, for Flask streaming we need a different approach if we want real-time.
                # HuggingFacePipeline with stream() is better.

        try:
            yield f"data: {json.dumps({'status': 'Loading model...', 'progress': 10})}\n\n"
            llm = get_llm(model_name)
            yield f"data: {json.dumps({'status': 'Model loaded. Preparing prompt...', 'progress': 40})}\n\n"

            # Use PromptTemplate for cleaner interface
            if "tinyllama" in model_name.lower():
                template = "<|system|>\n{system_prompt}</s>\n<|user|>\nSpecs: {specs}\nDiscount: {discount}\nTheme: {theme}</s>\n<|assistant|>\n"
            elif "phi-2" in model_name.lower():
                template = "Instruct: {system_prompt}\nSpecs: {specs}\nDiscount: {discount}\nTheme: {theme}\nOutput:"
            elif "flan-t5" in model_name.lower():
                template = "{system_prompt}\nSpecs: {specs}\nDiscount: {discount}\nTheme: {theme}"
            else:
                template = "{system_prompt}\nSpecs: {specs}\nDiscount: {discount}\nTheme: {theme}\nWebsite url\nWrite the advertisement text now.\n"

            prompt_template = PromptTemplate.from_template(template)
            prompt = prompt_template.format(
                system_prompt=ad_prompt,
                specs=specs,
                discount=discount,
                theme=theme
            )

            yield f"data: {json.dumps({'status': 'Generating text...', 'progress': 50})}\n\n"

            generated_text = ""
            tokens_generated = 0

            # Use llm.stream() for real LangChain streaming
            # Workaround for Seq2Seq models which have a bug in langchain-huggingface stream()
            if "t5" in model_name.lower():
                # For Seq2Seq models, we use a manual streaming approach to avoid the 'inputs' TypeError
                # which occurs in langchain-huggingface's internal threading logic.
                from threading import Thread

                pipe = llm.pipeline
                tokenizer = pipe.tokenizer
                model = pipe.model

                # Use pipeline's config to stay consistent with original setup
                gen_config = pipe.model.generation_config

                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                # For Seq2Seq models, skip_prompt=True can sometimes cause issues or be unnecessary
                # as the prompt is not in the decoder output.
                streamer = TextIteratorStreamer(tokenizer, skip_prompt=False, skip_special_tokens=True)

                generation_kwargs = dict(
                    **inputs,
                    streamer=streamer,
                    max_new_tokens=pipe.kwargs.get("max_new_tokens", 512),
                    do_sample=pipe.kwargs.get("do_sample", True),
                    temperature=pipe.kwargs.get("temperature", 0.7),
                    top_p=pipe.kwargs.get("top_p", 0.95),
                )

                thread = Thread(target=model.generate, kwargs=generation_kwargs)
                thread.start()

                for chunk in streamer:
                    if not chunk:
                        continue
                    generated_text += chunk
                    tokens_generated += 1
                    progress = min(50 + int((tokens_generated / 512) * 50), 99)
                    yield f"data: {json.dumps({'status': 'Generating...', 'progress': progress, 'text': generated_text})}\n\n"
            else:
                for chunk in llm.stream(prompt):
                    generated_text += chunk
                    tokens_generated += 1
                    progress = min(50 + int((tokens_generated / 512) * 50), 99)
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
                        'executive_summary': (generated_text[:1000] + '...') if len(
                            generated_text) > 1000 else generated_text,
                        'findings': [],
                        'recommendations': [],
                        'notes': f'Generated {len(generated_text)} characters; tokens_estimate={tokens_generated}',
                        'metrics': {
                            'Chars': str(len(generated_text)),
                            'Tokens_Est': str(tokens_generated),
                            'words': str(len(generated_text.split()))
                        },
                        'summary_table': [
                            {'label': 'Generated Chars', 'value': str(len(generated_text))},
                            {'label': 'Token Estimate', 'value': str(tokens_generated)},
                            {'label': 'Words', 'value': str(len(generated_text.split()))}
                        ],
                        'version': '1.0',
                        'generated_text': generated_text
                    }

                    # Add recommendations based on the ratio of chars to tokens
                    try:
                        tokens_val = float(512) / float(tokens_generated) if tokens_generated else 0.0
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
def report_view():
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
                {'title': 'Improved Accuracy',
                 'detail': 'Validation accuracy increased by 4.3% compared to previous run.'},
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
        return render_template('model_report.html', **sample_data,
                               metrics_chart_json=json.dumps(chart_config) if 'chart_config' in locals() else None)
    else:
        # Use the generated report data
        return render_template('model_report.html', **report_data, metrics_chart_json=metrics_chart_json)


@app.route("/summary")
def summary_view():
    return render_template("summary.html")


@app.route('/_cache')
def cache_status():
    # Debug-only endpoint to view cache contents
    if not app.debug:
        return Response(json.dumps({'error': 'cache inspection disabled'}), status=403, mimetype='application/json')
    with cache_lock:
        entries = []
        for k, v in model_cache.items():
            entries.append({'model_name': k, 'loaded': bool(v.get('llm') is not None)})
    return Response(json.dumps({'cache_max': CACHE_MAX, 'size': len(entries), 'entries': entries}, indent=2),
                    mimetype='application/json')


if __name__ == "__main__":
    app.run(debug=True)
