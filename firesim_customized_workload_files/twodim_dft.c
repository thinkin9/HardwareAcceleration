#include <stdio.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "twodim_dft_wo_opt.h"

/* 
Operation Lists
1. DFT on d_model direction
2. DFT on seq_len direction
*/

int seq_len = const_seq_len,
    d_model = const_d_model,
    d_inner = const_d_inner;

int seq_len_x2 = const_seq_len_x2,
    d_model_x2 = const_d_model_x2;

uint64_t start = 0,
    end = 0;

uint64_t dft_1st_cycles = 0,
    dft_2nd_cycles = 0;

int main(int argc, char* argv[]){
    gemmini_flush(0);
    
    elem_t* input_extended = _input_extended;
    elem_t* input_transposed = _input_transposed;
    const elem_t* Wseqlen = _Wseqlen;
    const elem_t* Wdmodel = _Wdmodel;

    elem_t* output_1st_dft = _output_1st_dft;
    elem_t* output_2nd_dft = _output_2nd_dft;

    // 1. DFT on seq_len direction
    //      W_seq_len * input_extend
    //      (seq_len_x2, seq_len_x2) x (seq_len_x2, d_model) = (seq_len_x2, d_model)

    printf("1ST_DFT(seq_len)\n");
    start = read_cycles();

    tiled_matmul_auto(seq_len_x2, d_model, seq_len_x2,
        Wseqlen, input_extended, NULL, output_1st_dft,
        seq_len_x2, d_model, 0, d_model,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ false, 
        /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
        /*tiled_matmul_type*/ WS);

    end = read_cycles();
    dft_1st_cycles += end - start;
    gemmini_fence();

    // 2. DFT on d_model direction
    //      W_d_model * (input_transposed)T
    //      (d_model_x2, d_model_x2) x (d_model_x2, seq_len) = (d_model_x2, seq_len) => we need to transpose it to use (seq_len, d_model)
    //      (W_d_model * (input_transposed)T )T == input_transposed * (W_d_model)T
    //      W_d_model is not symmetric

    printf("2ND_DFT(d_model)\n");
    start = read_cycles();

    tiled_matmul_auto(seq_len, d_model_x2, d_model_x2,
        input_transposed, Wdmodel, NULL, output_2nd_dft,
        d_model_x2, d_model_x2, 0, seq_len,
         MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        /*repeating_bias*/ false, /*transpose_A*/ true, /*transpose_B*/ false, 
        /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
        /*tiled_matmul_type*/ WS);

    end = read_cycles();
    dft_2nd_cycles += end - start;
    gemmini_fence();

    printf("dft_1st_cycles: %llu\n", dft_1st_cycles);
    printf("dft_2nd_cycles: %llu\n", dft_2nd_cycles);
}

