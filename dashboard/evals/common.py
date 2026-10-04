"""What every live eval shares: choosing the model, and timing/usage stats.

Each eval runs the production model path (llm.build_chat_model) with the
configured provider/model, or with --provider/--model overrides, so the
same golden cases can be run against any supported model.
"""

import statistics
import time

from llm import load_llm_config


def add_llm_arguments(parser):
    parser.add_argument("--provider", help="LLM provider (default: LLM_PROVIDER, else anthropic)")
    parser.add_argument("--model", help="model name (default: the provider's *_MODEL setting or default)")


def llm_config_from_args(args):
    return load_llm_config(args.provider, args.model)


def print_header(title, config, runs):
    print(f"{title}: {config.label}, {runs} run(s) per case", flush=True)


class Timer:
    """with Timer() as t: ...; t.seconds"""

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.seconds = time.perf_counter() - self.start


def usage(messages):
    """(input_tokens, output_tokens) summed over the AI messages' standard
    LangChain usage_metadata -- both providers report it -- or (None, None)
    if no message did."""
    input_tokens, output_tokens, reported = 0, 0, False
    for message in messages:
        usage_metadata = getattr(message, "usage_metadata", None)
        if not usage_metadata:
            continue
        reported = True
        input_tokens += usage_metadata.get("input_tokens", 0)
        output_tokens += usage_metadata.get("output_tokens", 0)
    if not reported:
        return None, None
    return input_tokens, output_tokens


def performance(records):
    """Median latency and mean tokens over `records`, each a dict with
    seconds, input_tokens, output_tokens (None when not reported)."""
    seconds, inputs, outputs = [], [], []
    for record in records:
        if record.get("seconds") is not None:
            seconds.append(record["seconds"])
        if record.get("input_tokens") is not None:
            inputs.append(record["input_tokens"])
        if record.get("output_tokens") is not None:
            outputs.append(record["output_tokens"])
    return {
        "median_seconds": statistics.median(seconds) if seconds else None,
        "mean_input_tokens": statistics.mean(inputs) if inputs else None,
        "mean_output_tokens": statistics.mean(outputs) if outputs else None,
        "usage_reported": f"{len(inputs)}/{len(records)}",
    }


def print_performance(perf, unit="run"):
    def fmt(value, spec):
        return "n/a" if value is None else format(value, spec)

    print(f"{'Median latency:':<26}{fmt(perf['median_seconds'], '.1f')}s per {unit}")
    print(f"{'Mean input tokens:':<26}{fmt(perf['mean_input_tokens'], ',.0f')}")
    print(f"{'Mean output tokens:':<26}{fmt(perf['mean_output_tokens'], ',.0f')}")
    print(f"{'Usage reported:':<26}{perf['usage_reported']} {unit}s")
