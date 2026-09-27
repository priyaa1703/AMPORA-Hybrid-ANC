# Model Checkpoints

The trained PyTorch checkpoint files are intentionally not included in this GitHub repository.

The checkpoints are stored separately in Google Drive:

- `AMPORA_HIGHBAND_DCCRN_BEST.pt`
- `AMPORA_HIGHBAND_DCCRN_LAST.pt`

The repository instead contains the exported deployment models:

- `ampora_lite_dccrn_fp32.onnx`
- `ampora_lite_dccrn_int8.onnx`

The **BEST** checkpoint corresponds to the model selected using the lowest validation loss during training.

For reproducibility, the training notebook and training history are included in this repository.
