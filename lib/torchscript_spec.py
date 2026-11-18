"""
TorchScript model checks for the legacy Processing TEE.

Ported from the new Processing TEE (internal/modelpkg/modelpkg.go
checkTorchScript and tools/infer/infer.py read_embedded_spec /
parse_input_spec) so a .pt is accepted on the same terms:

  * it must be a TorchScript archive written by torch.jit.save (a zip holding
    */code/* and constants.pkl), not a torch.save checkpoint;
  * it must declare its input size in an embedded extra/tanuh.json (or
    extra/input_spec.json), e.g.
        torch.jit.save(m, f, _extra_files={"tanuh.json": json.dumps(
            {"input_shape": [3, 224, 224]})})

The archive is only read as a zip here; nothing in it is executed.
In the legacy pipeline the evaluation script does the preprocessing, so only
the input shape, layout and dtype are used (resize and normalisation keys are
ignored rather than rejected).
"""

import json
import re
import zipfile

EMBEDDED_SPEC_NAMES = ("tanuh.json", "input_spec.json")

NOT_TORCHSCRIPT = "model is not a TorchScript archive (save it with torch.jit.save, not torch.save)"
INPUT_SIZE_UNKNOWN = (
    "the TorchScript model's input size is unknown: embed it with "
    "torch.jit.save(m, f, _extra_files={\"tanuh.json\": '{\"input_shape\": [3, 224, 224]}'})"
)

DTYPE_ALIASES = {"float32": "float32", "float": "float32", "fp32": "float32", "f32": "float32",
                 "float16": "float16", "half": "float16", "fp16": "float16", "f16": "float16",
                 "uint8": "uint8", "u8": "uint8"}
LAYOUT_ALIASES = {"nchw": "NCHW", "chw": "NCHW", "channels_first": "NCHW",
                  "nhwc": "NHWC", "hwc": "NHWC", "channels_last": "NHWC"}
SHAPE_KEYS = ("input_shape", "shape", "input_size", "image_size", "img_size", "size", "resolution")


class TorchScriptModelError(ValueError):
    """The uploaded TorchScript model is unusable (a user model error)."""


def _norm_key(k):
    return re.sub(r"[^a-z0-9]+", "_", str(k).lower()).strip("_")


def _positive_int(v, what, source):
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v != int(v) or int(v) <= 0:
        raise TorchScriptModelError(f"{source}: {what} must be a positive whole number, got {v!r}")
    return int(v)


def parse_input_spec(raw, source):
    """Read a flexible input declaration into {channels, height, width,
    layout, dtype}, holding only what it states (same keys as the new code)."""
    if not isinstance(raw, dict):
        raise TorchScriptModelError(f"{source} must be a JSON object")
    keys = {_norm_key(k): v for k, v in raw.items()}
    out = {}
    layout = keys.get("layout", keys.get("data_format"))
    if layout is not None:
        out["layout"] = LAYOUT_ALIASES.get(str(layout).lower())
        if out["layout"] is None:
            raise TorchScriptModelError(f"{source}: layout must be NCHW or NHWC, got {layout!r}")

    shape = next((keys[k] for k in SHAPE_KEYS if k in keys), None)
    if isinstance(shape, (int, float)) and not isinstance(shape, bool):
        out["height"] = out["width"] = _positive_int(shape, "size", source)
    elif isinstance(shape, dict):
        dims = {_norm_key(k): v for k, v in shape.items()}
        h, w = dims.get("height", dims.get("h")), dims.get("width", dims.get("w"))
        if h is None or w is None:
            raise TorchScriptModelError(f"{source}: size object needs height and width, got {shape!r}")
        out["height"], out["width"] = _positive_int(h, "height", source), _positive_int(w, "width", source)
    elif isinstance(shape, (list, tuple)):
        dims = list(shape)
        if len(dims) == 4:
            dims = dims[1:]  # [N, ...]: the batch size is not part of the spec
        dims = [_positive_int(d, "every dimension", source) for d in dims]
        if len(dims) == 3:
            lay = out.get("layout")
            if lay is None:  # channels are the 1-or-3 end
                lay = "NHWC" if dims[2] in (1, 3) and dims[0] not in (1, 3) else "NCHW"
                out["layout"] = lay
            c, h, w = dims if lay == "NCHW" else (dims[2], dims[0], dims[1])
            out.update(channels=c, height=h, width=w)
        elif len(dims) == 2:
            out["height"], out["width"] = dims
        elif len(dims) == 1:
            out["height"] = out["width"] = dims[0]
        else:
            raise TorchScriptModelError(
                f"{source}: input shape must be [H, W], [C, H, W] or [N, C, H, W], got {shape!r}")
    elif shape is not None:
        raise TorchScriptModelError(f"{source}: cannot read input size {shape!r}")
    for key in ("height", "width"):
        if key in keys:
            out[key] = _positive_int(keys[key], key, source)
    for key in ("channels", "num_channels", "in_channels"):
        if key in keys:
            out["channels"] = _positive_int(keys[key], "channels", source)
    if "dtype" in keys:
        out["dtype"] = DTYPE_ALIASES.get(str(keys["dtype"]).lower())
        if out["dtype"] is None:
            raise TorchScriptModelError(
                f"{source}: dtype must be float32, float16 or uint8, got {keys['dtype']!r}")
    return out


def check_torchscript(model_path):
    """Validate a TorchScript archive and return its input spec:
    {"channels", "height", "width", "layout", "dtype", "source"}."""
    try:
        with zipfile.ZipFile(model_path) as zf:
            names = zf.namelist()
            if not any("/code/" in n for n in names) or not any(n.endswith("constants.pkl") for n in names):
                raise TorchScriptModelError(NOT_TORCHSCRIPT)
            embedded, source = {}, None
            for wanted in EMBEDDED_SPEC_NAMES:
                entries = [n for n in names if n.endswith("/extra/" + wanted)]
                if entries:
                    source = f"embedded {wanted}"
                    try:
                        raw = json.loads(zf.read(entries[0]).decode("utf-8-sig"))
                    except (json.JSONDecodeError, UnicodeDecodeError) as exc:
                        raise TorchScriptModelError(f"{source} is not valid JSON: {exc}") from exc
                    embedded = parse_input_spec(raw, source)
                    break
    except zipfile.BadZipFile as exc:
        raise TorchScriptModelError(NOT_TORCHSCRIPT) from exc

    spec = {"channels": 3, "layout": "NCHW", "dtype": "float32", **embedded}
    if "height" not in spec or "width" not in spec:
        raise TorchScriptModelError(INPUT_SIZE_UNKNOWN)
    if spec["channels"] not in (1, 3):
        raise TorchScriptModelError(f"input must have 1 or 3 channels, declared {spec['channels']}")
    spec["source"] = source
    return spec
