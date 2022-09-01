# Model Deployment
1. convert the torch model to onnx
```shell
python convert_to_onnx.py --run --agents 32
```
2. simplify the onnx model
```shell
python convert_to_onnx.py --simplify
```
3. inference
   1. using onnxruntime-gpu:
   ```shell
   python convert_to_onnx.py --run --agents 32
   ```
   2. using pytorch:
   ```shell
   python convert_to_onnx.py --run_torch
   ```