from .profiler import Profiler

try:
    from .benchmark_utils import compare_benchmarks
except ModuleNotFoundError as exc:
    if exc.name != "pandas":
        raise

    def compare_benchmarks(*args, **kwargs):
        raise ModuleNotFoundError("pandas is required to compare benchmarks")
