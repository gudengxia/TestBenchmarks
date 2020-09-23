#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include "oqs/oqs.h"
#include "cpucycles.h"
#include "speed_print.h"

#define NTESTS 100

uint64_t t[NTESTS];
//uint8_t seed[KYBER_SYMBYTES] = {0};
#define CRYPTO_PUBLICKEYBYTES 897
#define CRYPTO_SECRETKEYBYTES 1281
#define CRYPTO_SIGNATUREBYTES 690
#define CRYPTO_BYTES 500
int main()
{
  int ret;
  unsigned int i;
  unsigned char pk[CRYPTO_PUBLICKEYBYTES] = {0};
  unsigned char sk[CRYPTO_SECRETKEYBYTES] = {0};
  unsigned char sig[NTESTS][CRYPTO_SIGNATUREBYTES] = {0};
  size_t sig_len[NTESTS], msg_len = CRYPTO_BYTES;
  unsigned char m[CRYPTO_BYTES] = {0};

  printf("SK_SIZE=%d\n", CRYPTO_SECRETKEYBYTES);
  printf("PK_SIZE=%d\n", CRYPTO_PUBLICKEYBYTES );
  printf("CT_SIZE=%d\n", CRYPTO_SIGNATUREBYTES );
 
  for(i = 0; i < CRYPTO_BYTES; i++)
	m[i] = i;

  OQS_SIG *oqs = OQS_SIG_new("falcon-512");
  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    ret = OQS_SIG_keypair(oqs, pk, sk);
    /*if(ret != OQS_SUCCESS)
      printf("keypair err.\n");*/
  }
  print_results("keypair: ", t, NTESTS);

  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    ret = OQS_SIG_sign(oqs, (uint8_t*)sig[i], &sig_len[i], m, msg_len, sk);
    /*if(ret != OQS_SUCCESS)
    {
	    printf("sig err.\n");
	    }*/
  }
  print_results("sig: ", t, NTESTS);

  for(i=0;i<NTESTS;i++) {
    t[i] = cpucycles();
    ret = OQS_SIG_verify(oqs, m, msg_len, sig[i], sig_len[i], pk);
    /*if(ret != OQS_SUCCESS)
    {
	    printf("vrfy err\n");
	    }*/
  }
  print_results("verify: ", t, NTESTS);

  OQS_SIG_free(oqs);
  return 0;
}
