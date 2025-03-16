#ifndef TWODIM_DFT_WO_OPT_H
#define TWODIM_DFT_WO_OPT_H

#include "include/gemmini_params.h"

#define const_seq_len 128
#define const_d_model 512

#define const_seq_len_x2 256
#define const_d_model_x2 1024

#define const_d_inner 2048

// A
static const elem_t _input[const_seq_len][const_d_model];

// Extended
static elem_t _input_extended[const_seq_len_x2][const_d_model];
static elem_t _input_transposed[const_seq_len][const_d_model_x2];

// Weights
static const elem_t _Wseqlen[const_seq_len_x2][const_seq_len_x2];
static const elem_t _Wdmodel[const_d_model_x2][const_d_model_x2];

// Outputs
static elem_t _output_1st_dft[const_seq_len_x2][const_d_model];
static elem_t _output_2nd_dft[const_d_model_x2][const_seq_len];

#endif TWODIM_DFT_H