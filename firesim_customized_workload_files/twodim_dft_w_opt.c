#include <stdio.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "twodim_dft.h"

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

    elem_t* input = _input;
    const elem_t* W_d_model_R = _W_d_model_R;
    const elem_t* W_d_model_I = _W_d_model_I;
    const elem_t* W_seq_len_R = _W_seq_len_R;
    const elem_t* W_seq_len_I = _W_seq_len_I;

    elem_t* output_1st_dft = _output_1st_dft;
    elem_t* output_2nd_dft = _output_1st_dft;
    elem_t* output_fourier = _output_fourier;

    printf("1ST_DFT(seq_len)\n");
    start = read_cycles();

    // W_seq_len_R * A_R
    // (seq_len, seq_len) x (seq_len, d_model) = (seq_len, d_model)
    tiled_matmul_auto(seq_len, d_model, seq_len,
        W_seq_len_R, input, NULL, output_1st_dft,
        seq_len, d_model, 0, d_model,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ false, 
        /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
        /*tiled_matmul_type*/ WS);

    // W_seq_len_I * A_R
    // (seq_len, seq_len) x (seq_len, d_model) = (seq_len, d_model)
    tiled_matmul_auto(seq_len, d_model, seq_len,
        W_seq_len_I, input, NULL, output_1st_dft + seq_len * d_model,
        seq_len, d_model, 0, d_model,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ false, 
        /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
        /*tiled_matmul_type*/ WS);

    end = read_cycles();
    dft_1st_cycles += end - start;
    gemmini_fence();
    
    printf("2ND_DFT(d_model)\n");
    start = read_cycles();
    
    // W_d_model_R * (W_seq_len_R * A_R)T == (d_model, seq_len) => we need to transpose it to make (seq_len, d_model)
    // ( W_d_model_R * (W_seq_len_R * A_R)T )T == (W_seq_len_R * A_R) * (W_d_model_R)T
    // W_d_model_R is symmetric matrix
    // (W_seq_len_R * A_R) * (W_d_model_R)
    // (seq_len, d_model) x (d_model, d_model) = (seq_len, d_model)
    tiled_matmul_auto(seq_len, d_model, d_model,
        output_1st_dft, W_d_model_R, 0, output_2nd_dft,
        d_model, d_model, 0, d_model,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ false, 
        /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
        /*tiled_matmul_type*/ WS);

    // W_d_model_I * (W_seq_len_I * A_R)T == (d_model, seq_len) => we need to transpose it
    // ( W_d_model_I * (W_seq_len_I * A_R)T )T == (W_seq_len_I * A_R) * (W_d_model_I)T
    // W_d_model_I is symmetric matrix
    // (W_seq_len_I * A_R) * (W_d_model_I)
    // (seq_len, d_model) x (d_model, d_model) = (seq_len, d_model)
    tiled_matmul_auto(seq_len, seq_len, d_model,
        output_1st_dft + seq_len * d_model, W_d_model_I, 0, output_2nd_dft + seq_len * d_model,
        d_model, d_model, 0, d_model,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ false, 
        /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
        /*tiled_matmul_type*/ WS);

    // RESADD
    tiled_resadd_auto(seq_len, d_model,
        MVIN_SCALE_IDENTITY,
        MVIN_SCALE_IDENTITY,
        ACC_SCALE_IDENTITY,
        output_2nd_dft,
        output_2nd_dft + seq_len * d_model,
        output_fourier,
        /*relu=*/ false,
        WS);

    end = read_cycles();
    dft_2nd_cycles += end - start;
    gemmini_fence();

    printf("dft_1st_cycles: %llu\n", dft_1st_cycles);
    printf("dft_2nd_cycles: %llu\n", dft_2nd_cycles);

}

