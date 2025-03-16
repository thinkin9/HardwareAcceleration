#ifndef IBERT_H
#define IBERT_H

#include <include/gemmini_params.h>

#define const_seq_len 128
#define const_d_model 512
#define const_d_inner 2048
#define const_n_head 4

static const elem_t _input[const_seq_len][const_d_model];
static const elem_t _enc_out[const_seq_len][const_d_model];
static elem_t _output[const_seq_len][const_d_model];

static const elem_t _Wqkvo[4][const_d_model][const_d_model];
static const elem_t _Wff1[const_d_model][const_d_inner];
static const elem_t _Wff2[const_d_inner][const_d_model];
static const elem_t _bff1[const_d_inner];
static const elem_t _bff2[const_d_model];

static elem_t _QKV_buf[3][const_seq_len][const_d_model];
static elem_t _attn1_buf[const_n_head][const_seq_len][const_seq_len];
static elem_t _attn2_buf[const_seq_len][const_d_model];
static elem_t _Z_buf[const_seq_len][const_d_model];

static elem_t _ff1_buf[const_d_model][const_d_inner];
static elem_t _ff2_buf[const_d_inner][const_d_model];

static elem_t _resadd1_out[const_seq_len][const_d_model];
static elem_t _resadd2_out[const_seq_len][const_d_model];

#endif