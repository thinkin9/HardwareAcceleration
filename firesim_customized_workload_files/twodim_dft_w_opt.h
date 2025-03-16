#ifndef TWODIM_DFT_H
#define TWODIM_DFT_H

#include "include/gemmini_params.h"

#define const_seq_len 128
#define const_d_model 512

#define const_seq_len_x2 256
#define const_d_model_x2 1024

#define const_d_inner 2048

// A
static const elem_t _input[const_seq_len][const_d_model];

// Weights
static const elem_t _W_d_model_R[const_d_model][const_d_model];
static const elem_t _W_d_model_I[const_d_model][const_d_model];

static const elem_t _W_seq_len_R[const_seq_len][const_seq_len];
static const elem_t _W_seq_len_I[const_seq_len][const_seq_len];

// Outputs
static elem_t _output_1st_dft[const_seq_len_x2][const_d_model];
static elem_t _output_2nd_dft[const_seq_len_x2][const_d_model];

static elem_t _output_fourier[const_seq_len][const_d_model];

#endif TWODIM_DFT_H