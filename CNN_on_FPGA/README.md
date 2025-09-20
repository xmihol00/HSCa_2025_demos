# CNN Accelerated on FPGA with High-Level Synthesis
The source files in this directory contains just the minimal code to show the CNN model and C++ code that can be synthesized to FPGA using High-Level Synthesis (HLS) tools. The full implementation is available at https://github.com/xmihol00/PYNQ-Z2_image_classification/tree/main

## Inference performance
The following table summarizes the performance of the different implementations of the CNN model achieved on a desktop CPU and the PYNQ-Z2 board for sample by sample inference:
| Implementation               | Hardware                                                      | Average FPS | 
|------------------------------|---------------------------------------------------------------|-------------|
| FPGA accelerated             | Dual ARM® Cortex™-A9 CPU @ 650 MHz, Artix™ 7 FPGA @ 83.33 MHz | **22.3**    |
| Tensorflow                   | Intel(R) Core(TM) i5-4670K CPU @ 3.40 GHz                     | 20.9        |
| SW integer-based arithmetics | Intel(R) Core(TM) i5-4670K CPU @ 3.40 GHz                     | 18.1        |