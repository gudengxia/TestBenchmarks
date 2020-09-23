#ifndef API_H
#define API_H


#include "params.h"
#include "fips202.h"
#include "../keccak4x/sha.h"
#if SEED_BYTES == 32
//#define Hash sha3_256
//#define Hash2 sha3_512
#define Hash(OUT, IN, INBYTES) SHA256(IN, INBYTES, OUT)
#define Hash2(OUT, IN, INBYTES) SHA512(IN, INBYTES, OUT)
#elif SEED_BYTES == 64
//#define Hash sha3_512
#define Hash(OUT, IN, INBYTES) SHA512(IN, INBYTES, OUT)
#define Hash2 sha3_1024
#else
#error "kem.c only supports SEED_BYTES in {32,64}"
#endif

#define KEM_SECRETKEYBYTES  SK_BYTES
#define KEM_PUBLICKEYBYTES  PK_BYTES
#define KEM_BYTES           SEED_BYTES
#define KEM_CIPHERTEXTBYTES CT_BYTES
#define KEM_ALGNAME "Aigis-enc"

//#define PKC_ALG_API __attribute__((visibility("default")))
int mkem_keygen( unsigned char *pk, unsigned char *sk);
int mkem_enc(const unsigned char *pk, unsigned char *ss, unsigned char *ct);
int mdkem_enc(unsigned char *pk, unsigned char *rnd, unsigned char *ss, unsigned char *ct);
int mkem_dec(const unsigned char *sk, const unsigned char *ct, unsigned char *ss);
#endif
