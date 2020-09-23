#include <stdint.h>
#include "api.h"
#include "Alg.h"
#include "cpucycles.h"
#include "speed_print.h"
#include "ds_benchmark.h"
#define NTESTS 10000
#define MLEN 59
uint64_t t[NTESTS];

int main(void)
{
  unsigned int i;
  uint32_t ret;

  unsigned char m[MLEN];
  unsigned char sm[AIGIS_SIG_BYTES];
  unsigned char pk[AIGIS_SIG_PUBLICKEYBYTES];
  unsigned char sk[AIGIS_SIG_SECRETKEYBYTES];

  unsigned long long pk_bytes, sk_bytes, sm_bytes;	      
  const int seconds = 1;
  aigis_randombytes(m, MLEN);
  PRINT_TIMER_HEADER
  TIME_OPERATION_SECONDS({ aigis_sig_keygen(pk, &pk_bytes, sk, &sk_bytes); }, "Key generation", seconds);
  TIME_OPERATION_SECONDS({ aigis_sig_sign(sk, sk_bytes, m, MLEN, sm, &sm_bytes);}, "Signature", seconds);
  TIME_OPERATION_SECONDS({ aigis_sig_verf(pk, pk_bytes, sm, sm_bytes, m, MLEN);}, "KEM decapsulate", seconds);
  
  for(i = 0; i < NTESTS; ++i) {
    t[i] = cpucycles();
    aigis_sig_keygen(pk, &pk_bytes, sk, &sk_bytes);
  }
  print_results("Keypair:", t, NTESTS);

  for(i = 0; i < NTESTS; ++i) {
    t[i] = cpucycles();
    aigis_sig_sign(sk, sk_bytes, m, MLEN, sm, &sm_bytes);
  }
  print_results("Sign:", t, NTESTS);

  for(i = 0; i < NTESTS; ++i) {
    t[i] = cpucycles();
    ret = aigis_sig_verf(pk, pk_bytes, sm, sm_bytes, m, MLEN);
    if(ret != 1)
	  printf("error!");
  }
  print_results("Verify:", t, NTESTS);

  return 0;
}
