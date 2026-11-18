"""
ONNX Runtime-compatible shim that runs a TorchScript model natively in PyTorch.

The legacy evaluation scripts (tanuh-eval-scripts in GCS) load the submitted
model with `onnxruntime.InferenceSession(model_path)`. For TorchScript jobs the
Processing TEE puts this directory first on the eval script's PYTHONPATH, so
`import onnxruntime` resolves here and the same script calls run model.pt with
torch.jit.load — no conversion to ONNX, and the GCS scripts stay unchanged.
ONNX jobs never see this module.

The model's input declaration (parsed from its embedded tanuh.json by
lib/torchscript_spec.py) arrives in the TANUH_TORCHSCRIPT_SPEC env var as JSON.

Covered API: InferenceSession(...).get_inputs() / get_outputs() / run() /
get_providers() / get_modelmeta(), plus get_available_providers(),
get_device(), SessionOptions, RunOptions, GraphOptimizationLevel,
ExecutionMode. Anything else raises a clear error naming the unsupported API.
"""

import enum
import json
import os

import numpy as np
import torch

__version__ = f"torchscript-shim (torch {torch.__version__})"

_TORCH_DTYPES = {"float32": torch.float32, "float16": torch.float16, "uint8": torch.uint8}
_ORT_TYPES = {"float32": "tensor(float)", "float16": "tensor(float16)", "uint8": "tensor(uint8)"}
_CUDA = "CUDAExecutionProvider"
_CPU = "CPUExecutionProvider"


def _spec():
    raw = os.environ.get("TANUH_TORCHSCRIPT_SPEC")
    if not raw:
        raise RuntimeError("TorchScript shim: TANUH_TORCHSCRIPT_SPEC is not set")
    return json.loads(raw)


class NodeArg:
    def __init__(self, name, shape, type_):
        self.name = name
        self.shape = shape
        self.type = type_

    def __repr__(self):
        return f"NodeArg(name={self.name!r}, type={self.type!r}, shape={self.shape!r})"


class ModelMetadata:
    producer_name = "pytorch"
    graph_name = "torchscript"
    domain = ""
    description = "TorchScript model run natively by the Processing TEE"
    version = 0
    custom_metadata_map = {}


class GraphOptimizationLevel(enum.IntEnum):
    ORT_DISABLE_ALL = 0
    ORT_ENABLE_BASIC = 1
    ORT_ENABLE_EXTENDED = 2
    ORT_ENABLE_ALL = 99


class ExecutionMode(enum.IntEnum):
    ORT_SEQUENTIAL = 0
    ORT_PARALLEL = 1


class _Options:
    """Accepts any option an eval script sets; none affect PyTorch execution."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return lambda *args, **kwargs: None


class SessionOptions(_Options):
    pass


class RunOptions(_Options):
    pass


def get_available_providers():
    return [_CUDA, _CPU] if torch.cuda.is_available() else [_CPU]


def get_device():
    return "GPU" if torch.cuda.is_available() else "CPU"


def _provider_names(providers):
    names = []
    for p in providers or []:
        names.append(p[0] if isinstance(p, (tuple, list)) else p)
    return names


def _flatten_outputs(out):
    """Same output naming as the new Processing TEE's flatten_torch_outputs."""
    if isinstance(out, torch.Tensor):
        items = {"output": out}
    elif isinstance(out, (tuple, list)):
        items = {f"output_{i}": t for i, t in enumerate(out)}
    elif isinstance(out, dict):
        items = {str(k): v for k, v in out.items()}
    else:
        raise RuntimeError(f"model returned {type(out).__name__}; expected a tensor, tuple or dict of tensors")
    result = {}
    for k, v in items.items():
        if not isinstance(v, torch.Tensor):
            raise RuntimeError(f"output {k!r} is {type(v).__name__}, expected a tensor")
        result[k] = v.detach().float().cpu().numpy()
    return result


class InferenceSession:
    def __init__(self, path_or_bytes, sess_options=None, providers=None, provider_options=None, **kwargs):
        if not isinstance(path_or_bytes, (str, os.PathLike)):
            raise RuntimeError("TorchScript shim: InferenceSession needs the model path, not bytes")
        spec = _spec()
        self._dtype = spec.get("dtype", "float32")
        layout = spec.get("layout", "NCHW")
        c, h, w = int(spec["channels"]), int(spec["height"]), int(spec["width"])
        chw = [c, h, w] if layout == "NCHW" else [h, w, c]

        # Honour an explicit provider list the way ONNX Runtime would: CUDA only
        # when it is requested (or nothing was requested) and available.
        requested = _provider_names(providers)
        want_cuda = (not requested or _CUDA in requested) and torch.cuda.is_available()
        self._device = "cuda" if want_cuda else "cpu"
        self._providers = [_CUDA, _CPU] if want_cuda else [_CPU]

        try:
            self._model = torch.jit.load(str(path_or_bytes), map_location=self._device)
            self._model.eval()
        except Exception as exc:
            raise RuntimeError(f"TorchScript model failed to load: {exc}") from exc

        self._inputs = [NodeArg("input", ["batch", *chw], _ORT_TYPES[self._dtype])]
        # One dummy forward pass to learn the output names and shapes.
        dummy = torch.zeros([1, *chw], dtype=_TORCH_DTYPES[self._dtype], device=self._device)
        with torch.inference_mode():
            sample = _flatten_outputs(self._model(dummy))
        self._outputs = [NodeArg(name, ["batch", *arr.shape[1:]], "tensor(float)") for name, arr in sample.items()]

    def get_inputs(self):
        return list(self._inputs)

    def get_outputs(self):
        return list(self._outputs)

    def get_overridable_initializers(self):
        return []

    def get_providers(self):
        return list(self._providers)

    def get_provider_options(self):
        return {p: {} for p in self._providers}

    def get_modelmeta(self):
        return ModelMetadata()

    def get_session_options(self):
        return SessionOptions()

    def set_providers(self, providers=None, provider_options=None):
        return None

    def run(self, output_names, input_feed, run_options=None):
        if len(input_feed) != 1:
            raise RuntimeError(f"TorchScript models take one input; got {sorted(input_feed)}")
        batch = np.ascontiguousarray(next(iter(input_feed.values())))
        x = torch.from_numpy(batch).to(device=self._device, dtype=_TORCH_DTYPES[self._dtype])
        with torch.inference_mode():
            outputs = _flatten_outputs(self._model(x))
        names = list(output_names) if output_names else [o.name for o in self._outputs]
        missing = [n for n in names if n not in outputs]
        if missing:
            raise RuntimeError(f"unknown output name(s) {missing}; the model's outputs are {list(outputs)}")
        return [outputs[n] for n in names]

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        raise AttributeError(f"onnxruntime.InferenceSession.{name} is not supported for TorchScript models")


def __getattr__(name):
    if name.startswith("__"):
        raise AttributeError(name)
    raise AttributeError(f"onnxruntime.{name} is not supported for TorchScript models")
