# har-edge-accelerator

**A human-activity-recognition model taken all the way from training to a hand-written FPGA design, with every hardware cost predicted before it is measured.**

> **Status: Phase 1 complete.** The PyTorch model is trained and measured (see [phase1_model/](phase1_model/README.md)); Phases 2 to 4 are not started. The results table below is filled in as each phase lands. See [docs/plan.md](docs/plan.md) for the task list.

## The question

Can you tell how much FPGA a neural network will cost *before* you write the RTL?

This repo answers it for one small, complete example. At each step the predicted LUTs, DSPs, block RAM and latency are written down first, then the design is built and the real numbers are recorded next to the prediction. The gap between the two is the result.

## What it does

A phone or wearable produces accelerometer and gyroscope readings. The model classifies each 2.56-second window as one of six activities: walking, walking upstairs, walking downstairs, sitting, standing, lying down.

The same model is implemented four ways:

| Phase | Folder | What gets built | Runs on |
|---|---|---|---|
| 1 | `phase1_model/` | Small 1D CNN trained in PyTorch on the UCI HAR dataset | Laptop CPU |
| 2 | `phase2_cuda/` | The same inference written by hand in CUDA | NVIDIA GPU (lab machine or Colab) |
| 3 | `phase3_quant/` | 8-bit integer version, plus a bit-exact Python reference model | Laptop CPU |
| 4 | `phase4_rtl/` | Hand-written Verilog accelerator, verified against the Phase 3 reference | Simulation, then Tang Nano 20K |

A fifth, optional step generates the same model with hls4ml and compares it with the hand-written design.

## Results

Filled in as phases complete. Empty cells mean "not measured yet", not zero.

| Implementation | Accuracy | Latency per window | Energy per window | LUTs | DSPs | BRAM |
|---|---|---|---|---|---|---|
| PyTorch float32 (CPU) | 90.33% | 0.034 ms (batch 1, 1 thread, Apple M4) | | n/a | n/a | n/a |
| CUDA float32 (GPU) | | | | n/a | n/a | n/a |
| int8 reference (CPU) | | | | n/a | n/a | n/a |
| Hand-written RTL, **predicted** | | | | | | |
| Hand-written RTL, **measured** | | | | | | |
| hls4ml (optional) | | | | | | |

## How the hardware is checked

- The Phase 3 Python model is the reference: it does the exact integer arithmetic the hardware is supposed to do.
- [cocotb](https://www.cocotb.org/) testbenches feed the same input windows to the Verilog and to the reference, and require identical outputs.
- Simulation uses Verilator; synthesis uses the open-source Gowin flow (Yosys, nextpnr, Apicula), driven through [Edalize](https://github.com/olofk/edalize).
- Simulation and synthesis run in GitHub Actions, so the numbers in the table can be reproduced without owning a board.

## Repository layout

```
phase1_model/   training, evaluation, exported weights
phase2_cuda/    hand-written CUDA inference and benchmarks
phase3_quant/   quantisation and the bit-exact integer reference
phase4_rtl/     Verilog, cocotb tests, build scripts
docs/           plan, prediction log, design notes
```

## Dataset

[Human Activity Recognition Using Smartphones](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) (UCI): 30 volunteers, 6 activities, 50 Hz, windows of 128 samples across 9 sensor channels. The data is downloaded by a script and is not committed to this repo.

## Author

Rithika Sridharbabu, Computer Engineering, San José State University.
