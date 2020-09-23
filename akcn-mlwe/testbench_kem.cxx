/********************************************************************************************
* Abstract: benchmarking/testing KEM scheme
*********************************************************************************************/

#include <string.h>
#include "ds_benchmark.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "api.h"

#define KEM_TEST_ITERATIONS 100
#define KEM_BENCH_SECONDS     1
#define FALSE 0
#define TRUE  1
#define NTESTS 10000

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

    crypto_kem_keypair(pk, sk);
    TIME_OPERATION_SECONDS({ crypto_kem_enc(ct, ss_encap, pk); }, "KEM encapsulate", seconds);

    crypto_kem_enc(ct, ss_encap, pk);
    TIME_OPERATION_SECONDS({ crypto_kem_dec(ss_decap, ct, sk); }, "KEM decapsulate", seconds);

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

    OK = kem_test(CRYPTO_ALGNAME, KEM_TEST_ITERATIONS);
    if (OK != TRUE) {
        goto exit;
    }

    PRINT_TIMER_HEADER
    kem_bench(KEM_BENCH_SECONDS);

exit:
    return (OK == TRUE) ? EXIT_SUCCESS : EXIT_FAILURE;
}
