"""S0 resource construction only; proposal/VERIFY scheduling is a later patch."""

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.runtime import DeviceRuntime


class DualModelRunner:
    def __init__(self, target_config: Config, draft_config: Config):
        if target_config is draft_config:
            raise ValueError("Models need separate mutable Config objects")
        for config in (target_config, draft_config):
            if config.tensor_parallel_size != 1 or config.enable_prefix_cache:
                raise ValueError("S0 resources require TP=1 and prefix cache disabled")
            if config.kv_cache_memory_bytes is None:
                raise ValueError("Set each model's explicit KV budget before loading")
        for field in ("kvcache_block_size", "max_model_len", "max_num_seqs", "enforce_eager"):
            if getattr(target_config, field) != getattr(draft_config, field):
                raise ValueError("Target/draft must agree on " + field)
        self.runtime = DeviceRuntime(0, 1)
        self.runners = []
        try:
            # Load both sets of weights before either runner consumes the KV budget.
            self.target = ModelRunner(target_config, 0, [], runtime=self.runtime, defer_cache=True)
            self.runners.append(self.target)
            self.draft = ModelRunner(draft_config, 0, [], runtime=self.runtime, defer_cache=True)
            self.runners.append(self.draft)
            self.target.initialize_cache()
            self.draft.initialize_cache()
        except BaseException as error:
            try:
                self.exit()
            except BaseException as cleanup_error:
                raise error from cleanup_error
            raise

    def exit(self):
        errors = []
        for runner in reversed(self.runners):
            try:
                runner.exit()
            except Exception as error:
                errors.append(error)
        self.runners.clear()
        try:
            self.runtime.close()
        except Exception as error:
            errors.append(error)
        if errors:
            raise errors[0]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        try:
            self.exit()
        except BaseException as cleanup_error:
            if exc_value is not None:
                raise exc_value from cleanup_error
            raise
