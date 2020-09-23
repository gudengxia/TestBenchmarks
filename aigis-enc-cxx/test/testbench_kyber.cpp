/********************************************************************************************
* Abstract: benchmarking/testing KEM scheme
*********************************************************************************************/

#include <string.h>
#include <stdio.h>
#include <stdint.h>
#include "ds_benchmark.h"
#include <oqs/oqs.h>

#define KEM_TEST_ITERATIONS 100
#define KEM_BENCH_SECONDS     1
#define FALSE 0
#define TRUE  1
#define CRYPTO_PUBLICKEYBYTES 1088
#define CRYPTO_SECRETKEYBYTES 2400
#define CRYPTO_BYTES 32
#define CRYPTO_CIPHERTEXTBYTES 1184


static int kem_test(const char *named_parameters, int iterations)
{
    uint8_t pk[CRYPTO_PUBLICKEYBYTES];
    uint8_t sk[CRYPTO_SECRETKEYBYTES];
    uint8_t ss_encap[CRYPTO_BYTES], ss_decap[CRYPTO_BYTES];
    uint8_t ct[CRYPTO_CIPHERTEXTBYTES];
    OQS_KEM *oqs;
    printf("\n");
    printf("Testing correctness of %s, tests for %d iterations\n", named_parameters, iterations);
    oqs = OQS_KEM_new("kyber768");

    printf("sk_len=%ld\n", oqs->length_secret_key);
    printf("pk_len=%ld\n", oqs->length_public_key);
    printf("ct_len=%ld\n", oqs->length_ciphertext);
    printf("ss_len=%ld\n", oqs->length_shared_secret);

    for (int i = 0; i < iterations; i++) {
        OQS_KEM_keypair(oqs, pk, sk);
        OQS_KEM_encaps(oqs, ct, ss_encap, pk);
        OQS_KEM_decaps(oqs, ss_decap, ct, sk);
        if (memcmp(ss_encap, ss_decap, CRYPTO_BYTES) != 0) {
            printf("\n ERROR!\n");
	        return FALSE;
        }
    }
    OQS_KEM_free(oqs);
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
    OQS_KEM *oqs;
    oqs = OQS_KEM_new("kyber768");
    TIME_OPERATION_SECONDS({ OQS_KEM_keypair(oqs, pk, sk); }, "Key generation", seconds);

    OQS_KEM_keypair(oqs, pk, sk);
    TIME_OPERATION_SECONDS({ OQS_KEM_encaps(oqs, ct, ss_encap, pk);}, "KEM encapsulate", seconds);

    OQS_KEM_encaps(oqs, pk, ss_encap, ct);
    TIME_OPERATION_SECONDS({ OQS_KEM_decaps(oqs, ss_decap, ct, sk);}, "KEM decapsulate", seconds);
    OQS_KEM_free(oqs);
}


int main()
{
    int OK = TRUE;

    OK = kem_test("kyber768", KEM_TEST_ITERATIONS);
    if (OK != TRUE) {
        goto exit;
    }

    PRINT_TIMER_HEADER
    kem_bench(KEM_BENCH_SECONDS);

exit:
    return (OK == TRUE) ? EXIT_SUCCESS : EXIT_FAILURE;
}
