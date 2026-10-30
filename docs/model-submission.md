# Submitting a model to TANUH

You upload:

1. **A model file** in one of three formats: ONNX, TorchScript or Hugging Face.
2. **`adaptor.py`**, which turns your model's raw outputs into the dataset's `predictions.csv`.
3. **Optionally, `preprocessing.py`** (ONNX and TorchScript only), which turns one dataset file into your model's input. See [preprocessing.py](#preprocessingpy).

The platform runs your model exactly as you uploaded it. **It does not convert, retrain or fix your model.**
Without a `preprocessing.py`, it does only these steps to every image before scoring:

1. Decode the image to RGB. For DICOM, it takes the middle frame, inverts MONOCHROME1, min-max scales to 0–255 and copies the grey channel into 3 channels.
2. Resize to your model's input size.
3. Scale to 0–1 (float inputs) or keep 0–255 (uint8 inputs).
4. Arrange the channels and layout your model declares.

**Your model file, or your `preprocessing.py`, must include every other preprocessing step your training used**, for example ImageNet mean/std, colour balancing or BGR order.
If a step is missing, nothing fails: your model simply scores lower. Test locally with `tools/check_model.py` before you upload.

## Choosing a format

| Your preprocessing | Use |
|---|---|
| Only the standard steps a Hugging Face image processor supports: resize, rescale, mean/std normalise, centre crop | Hugging Face, ONNX or TorchScript |
| Anything custom, such as colour balancing, BGR order or CLAHE | ONNX or TorchScript, with the steps built into the model graph |

## Hugging Face

Upload **one `.zip`** of the folder that `save_pretrained()` writes.

The platform loads it with:

```python
AutoImageProcessor.from_pretrained(dir, local_files_only=True, trust_remote_code=False)
AutoModelForImageClassification.from_pretrained(dir, local_files_only=True, use_safetensors=True,
                                                trust_remote_code=False, torch_dtype=torch.float32)
```

It runs offline with `transformers` 4.46.3 and `torch` 2.5.1. Your model gets **no internet access and runs none of your Python code**.

### How to create the zip

```python
model.save_pretrained("my-model", safe_serialization=True)   # config.json + model.safetensors
processor.save_pretrained("my-model")                         # preprocessor_config.json
```

Then zip the folder. You can put the files at the root of the zip, or inside one top-level folder (`my-model/config.json`).

### Required in the zip

| File | What it must be |
|---|---|
| `config.json` | Valid JSON with a `model_type` that `transformers` 4.46.3 supports for image classification, such as `vit`, `convnext`, `resnet`, `mobilevitv2`, `swin`, `efficientnet`, `deit` or `beit`. |
| `*.safetensors` | The weights: `model.safetensors`, or shards plus `model.safetensors.index.json`. |
| `preprocessor_config.json` | Your image processor's settings: `size`, `do_resize`, `do_rescale`, `do_normalize`, `image_mean`, `image_std`, `resample`, `crop_size`. **This is your preprocessing.** The processor receives RGB `uint8` images straight from step 1 above; the platform does no resizing or scaling. |

### Allowed

- Other JSON and text files, for example `model.safetensors.index.json`, `generation_config.json`, `README.md` or `.gitattributes`. They are ignored.

### Rejected

The job fails with `ModelFormatError` (error code 2) if the zip contains any of these:

| What | Why |
|---|---|
| Any file ending in `.py`, `.pyc`, `.bin`, `.pt`, `.pth`, `.pkl`, `.pickle`, `.ckpt`, `.h5` or `.joblib`, for example `pytorch_model.bin`, `training_args.bin`, `optimizer.pt`, `rng_state.pth` or `tf_model.h5` | These formats can run code when loaded. Do not zip a Trainer `checkpoint-XXXX/` folder; it contains these files. Zip the output of `save_pretrained()`. |
| `auto_map` in `config.json`, or a custom architecture | Custom model code is not run. The architecture must be built into `transformers`. |
| A missing `config.json`, `preprocessor_config.json` or `*.safetensors` | The model cannot load without them. |
| Symlinks, or paths that escape the folder (`../`) | Unsafe. |

### Size limits

- The zip must be at most 4 GiB.
- Unzipped, it must be at most 8 GiB and at most 10,000 files.

### Output

The platform saves the model's `logits` as `[N, num_labels]`, under the name `logits`.
The platform does not use `id2label` in `config.json`. Your adaptor decides how each output column maps to the dataset's `class_names`.

## ONNX

- **Upload:** `model.onnx`, plus the optional external-data file (`.onnx.data`) in the weights slot.
- **Input:** one image input with a fixed height and width, in NCHW or NHWC layout, with 1 or 3 channels, as float32, float16 or uint8. The batch size can be dynamic or fixed.
- **Resize method:** set metadata key `tanuh.resize` to `bilinear` (the default) or `bicubic`.
- **Your preprocessing:** add it as graph nodes in front of your network.

## TorchScript

- **Upload:** a `.pt` saved with `torch.jit.save`. Files saved with `torch.save` (state dicts or pickles) are rejected.
- **Input spec:** embed it in the file.

  ```python
  torch.jit.save(scripted, "model.pt", _extra_files={"tanuh.json": json.dumps(
      {"input_shape": [3, 256, 256], "layout": "NCHW", "dtype": "float32", "resize": "bicubic"})})
  ```

- **Your preprocessing:** put it inside the scripted module's `forward`.
- **Loading:** the model is loaded with `torch` 2.5.1.

## preprocessing.py

Optional, for ONNX and TorchScript models. Start from the BODH `preprocessing_template.py`.
When you upload it, the platform does **not** resize or scale your inputs: it calls your
`preprocess(x)` once per dataset file and feeds the result to your model.

- **What `x` is** depends on `INPUT["source"]` in your script:

  | `INPUT["source"]` | `x` |
  |---|---|
  | `"array"`, or no `INPUT` | the file decoded to `uint8` `(H, W, 3)` RGB (DICOM: middle frame, MONOCHROME1 inverted, min-max to 0–255) |
  | `"path"` or `"auto"` | the file's path, as a `str`, so you can read it yourself (e.g. DICOM headers) |
  | `"bytes"` | the file's raw bytes |

- **What it returns:** one input for your model **without** the batch axis, e.g. `float32` `[3, 256, 256]` for an NCHW model. It must match the model's declared channels, height and width (where they are fixed) and contain no NaN or infinity. It is cast to the model's input type; a `uint8` model needs integer values.
- **Libraries:** numpy, OpenCV, Pillow, pydicom and torch are installed. Other imports are installed from PyPI before the job runs, except imports inside a `try` block that catches `ImportError`, which are treated as optional.
- **Failures:** a script that fails to import, raises, or returns the wrong shape fails the job with error code 1 (`PreprocessingError`).
- **Hugging Face models** cannot use it: `preprocessor_config.json` is their preprocessing.

## adaptor.py

It runs as:

```
python3 adaptor.py --raw-dir raw/ --spec dataset_spec.json --output-dir predictions/
```

It reads these inputs:
- **`raw/raw_outputs.npz`:** `ids` holds the file names, and there is one array per model output.
- **`dataset_spec.json`:** holds `task_type`, `class_names` and `num_classes`.

It must write `predictions/predictions.csv`:

| Problem type | Columns |
|---|---|
| `binary_classification` | `file,label,score`, where `score` is the probability of class 1 |
| `multiclass_classification` | `file,label,prob_0,…,prob_{C-1}`, where the probabilities sum to 1 |

Include every file exactly once. For templates, see `tools/reference/adaptor_sigmoid_binary.py` and `adaptor_softmax_multiclass.py`.
