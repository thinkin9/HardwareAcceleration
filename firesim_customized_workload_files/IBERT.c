#include <stdio.h>
#include <string.h>
#include <stdbool.h>

#include "include/gemmini.h"
#include "include/gemmini_nn.h"

#include "transformer_params.h"

uint64_t start = 0,
        end = 0;

uint64_t linear_wqkv_cycles = 0,
        matmul_qk_cycles = 0,
        matmul_v_cycles = 0,
        linear_wo_cycles = 0,
        resadd_cycles = 0,
        linear_ff1_cycles = 0,
        linear_ff2_cycles = 0;

int seq_len = 128;
int d_model = 512;
int d_inner = 2048;
int n_head = 4;
int n_layer = 1;

int main(int argc, char* argv[]){
    gemmini_flush(0);    

    enum tiled_matmul_type_t stationary_type = WS;

    const elem_t *Wq = _Wqkvo[0];
    const elem_t *Wk = _Wqkvo[1];
    const elem_t *Wv = _Wqkvo[2];
    const elem_t *Wo = _Wqkvo[3];

    const elem_t* input = _input;
    const elem_t* enc_out = _enc_out;

    elem_t *Q_buf = _QKV_buf[0];
    elem_t *K_buf = _QKV_buf[1];
    elem_t *V_buf = _QKV_buf[2];

    elem_t *attn1_buf = _attn1_buf;
    elem_t *attn2_buf = _attn2_buf;
    elem_t *Z_buf = _Z_buf;


    // QKV with WQKV where Q is scaled to root of dk
    const int qkv_matmuls_n = 3;
    const elem_t* qkv_weights[] = {Wq, Wq, Wv};
    const elem_t* qkv_inputs[] = {input, enc_out, enc_out};
    elem_t* qkv_outputs[] = {Q_buf, K_buf, V_buf};
    printf("QKV\n");
    start = read_cycles();
    for (int i = 0; i < qkv_matmuls_n; i++){
        const elem_t* qkv_w = qkv_weights[i];
        const elem_t* qkv_i = qkv_inputs[i];
        elem_t* qkv_o = qkv_outputs[i];
        int sqrt_dk = 8;
        tiled_matmul_auto(seq_len, d_model, d_model,
            qkv_i, qkv_w, NULL, qkv_o,
            d_model, d_model, 0, d_model,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, 1/sqrt_dk ? i == 0 : ACC_SCALE_IDENTITY, 0,
            /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ false, 
            /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
            /*tiled_matmul_type*/ WS);
    }
    end = read_cycles();
    linear_wqkv_cycles += end - start;

    gemmini_fence();

    // attn1 = Q * KT
    // attn1 = softmax(attn)
    int d_head = d_model / n_head;
    printf("QK\n");
    start = read_cycles();
    for (int i = 0; i < n_head; i++) {
        const elem_t * A = Q_buf + i * d_head;
        const elem_t * B = K_buf + i * d_head;
        elem_t * C = attn1_buf + i * seq_len * seq_len;

        tiled_matmul_auto(seq_len, seq_len, d_head,
            A, B, NULL, C,
            d_model, d_model, 0, seq_len,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            SOFTMAX, ACC_SCALE_IDENTITY, /*bert_scale@tiled_matmul_ws_softmax*/ 0.05,
            /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ true, 
            /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
            /*tiled_matmul_type*/ WS);
    }
    end = read_cycles();
    matmul_qk_cycles += end - start;

    gemmini_fence();

    printf("V\n");
    // attn2 = attn1 * V
    start = read_cycles();
    for (int i = 0; i < n_head; i++) {
        const elem_t * A = attn1_buf + i * seq_len * seq_len;
        const elem_t * B = V_buf + i * d_head;
        elem_t * C = attn2_buf + i * d_head;

        tiled_matmul_auto(seq_len, d_head, seq_len,
            A, B, NULL, C,
            seq_len, d_model, 0, d_model,
            MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
            NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
            /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ false, 
            /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
            /*tiled_matmul_type*/ WS);
    }
    end = read_cycles();
    matmul_v_cycles += end - start;

    gemmini_fence();

    printf("Wo\n");
    // Z = attn2 * Wo
    start = read_cycles();
    tiled_matmul_auto(seq_len, d_model, d_model,
        attn2_buf, Wo, NULL, Z_buf,
        d_model, d_model, 0, d_model,
        MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY, MVIN_SCALE_IDENTITY,
        NO_ACTIVATION, ACC_SCALE_IDENTITY, 0,
        /*repeating_bias*/ false, /*transpose_A*/ false, /*transpose_B*/ false, 
        /*full_C*/ false, /*low_D*/ false, /*weightA*/ false,
        /*tiled_matmul_type*/ WS);
    end = read_cycles();
    linear_wo_cycles += end - start;

    gemmini_fence();

    printf("linear_wqkv_cycles: %llu\n", linear_wqkv_cycles);
    printf("matmul_qk_cycles: %llu\n", matmul_qk_cycles);
    printf("matmul_v_cycles: %llu\n", matmul_v_cycles);
    printf("linear_wo_cycles: %llu\n", linear_wo_cycles);

}