/********************************************************************************************
* Abstract: benchmarking/testing KEM scheme
*********************************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "ds_benchmark.h"
#include "cpucycles.h"
#include "speed_print.h"
//#include "api.h"
//#include "Alg.h"

#define KEM_TEST_ITERATIONS 100000
#define NTESTS KEM_TEST_ITERATIONS
#define KEM_BENCH_SECONDS     1
#define FALSE 0
#define TRUE  1
#define KEM_PUBLICKEYBYTES 896
#define KEM_SECRETKEYBYTES 1152
#define KEM_BYTES 32
#define KEM_CIPHERTEXTBYTES 992
#define CRYPTO_PUBLICKEYBYTES KEM_PUBLICKEYBYTES
#define CRYPTO_SECRETKEYBYTES KEM_SECRETKEYBYTES
#define CRYPTO_BYTES KEM_BYTES
#define CRYPTO_CIPHERTEXTBYTES KEM_CIPHERTEXTBYTES
#define CRYPTO_ALGNAME "Aigis-enc"
#define crypto_kem_keypair(pk, sk) mkem_keygen(pk, sk)
#define crypto_kem_enc(ct, ss, pk) mkem_enc(pk, ss, ct)
#define crypto_kem_dec(ss, ct, sk) mkem_dec(sk, ct, ss)

int mkem_keygen(unsigned char *pk, unsigned char *sk);
int mkem_enc(const unsigned char *pk, unsigned char *ss, unsigned char *ct);
int mkem_dec(const unsigned char *sk, const unsigned char *ct, unsigned char *ss);

static int kem_test(const char *named_parameters, int iterations)
{
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t ss_encap[CRYPTO_BYTES], ss_decap[CRYPTO_BYTES];
    uint8_t ct[CRYPTO_CIPHERTEXTBYTES];

    printf("\n");
    printf("Testing correctness of %s, tests for %d iterations\n", named_parameters, iterations);

    for (int i = 0; i < iterations; i++) {
        crypto_kem_keypair(pk, sk);
        crypto_kem_enc(ct, ss_encap, pk);
        crypto_kem_dec(ss_decap, ct, sk);
        if (memcmp(ss_encap, ss_decap, CRYPTO_BYTES) != 0) {
            printf("\n ERROR!\n");
	        return FALSE;
        }
    }
    printf("Tests PASSED. All session keys matched.\n");
    printf("\n");

    return TRUE;
}


static void kem_bench(const int seconds)
{
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t ss_encap[CRYPTO_BYTES], ss_decap[CRYPTO_BYTES];
    uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
    uint64_t t[NTESTS];
    int i;
    
    TIME_OPERATION_SECONDS({ crypto_kem_keypair(pk, sk); }, "Key generation", seconds);

    //crypto_kem_keypair(pk, sk);
    TIME_OPERATION_SECONDS({ crypto_kem_enc(ct, ss_encap, pk);}, "KEM encapsulate", seconds);

    //crypto_kem_enc(ct, ss_encap, pk);
    TIME_OPERATION_SECONDS({ crypto_kem_dec(ss_decap, ct, sk);}, "KEM decapsulate", seconds);

    for(i=0;i<NTESTS;i++) {
      t[i] = cpucycles();
      crypto_kem_keypair(pk, sk);
    }
    print_results("kem_keypair: ", t, NTESTS);

    for(i=0;i<NTESTS;i++) {
      t[i] = cpucycles();
      crypto_kem_enc(ct, ss_encap, pk);
    }
    print_results("kem_enc: ", t, NTESTS);

    for(i=0;i<NTESTS;i++) {
      t[i] = cpucycles();
      crypto_kem_dec(ss_decap, ct, sk);
    }
    print_results("kem_dec: ", t, NTESTS);
}


int main()
{
    int OK = TRUE;

    printf("sk_len=%d\n", KEM_SECRETKEYBYTES);
    printf("pk_len=%d\n", KEM_PUBLICKEYBYTES);
    printf("ct_len=%d\n", KEM_CIPHERTEXTBYTES);
    printf("ss_len=%d\n", KEM_BYTES);
    OK = kem_test(CRYPTO_ALGNAME, KEM_TEST_ITERATIONS);
    if (OK != TRUE) {
        goto exit;
    }

    PRINT_TIMER_HEADER
    kem_bench(KEM_BENCH_SECONDS);

exit:
    return (OK == TRUE) ? EXIT_SUCCESS : EXIT_FAILURE;
}
