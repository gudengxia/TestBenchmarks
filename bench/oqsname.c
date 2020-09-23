#include "oqs/oqs.h"
#include <stdio.h>
using namespace std;

/*int main()
{
  int i, cnt;
  const char * alg_name;
  uint8_t ss[32];
  uint8_t pk[1184], sk[2400], ct[1088];
  
  OQS_KEM *alg;
  OQS_STATUS ret;
  cnt = OQS_KEM_alg_count();
  for(i = 0; i < cnt; i++)
  {
      alg_name = OQS_KEM_alg_identifier(i);
      cout<< i << " "
	  << alg_name <<" "
	  << OQS_KEM_alg_is_enabled(alg_name) << endl;
  }

  alg = OQS_KEM_new("kyber768-90s");
  if(alg == NULL)
  {
      cout<<"ERROR"<<endl;
      exit(-1);
  }
  ret = OQS_KEM_keypair(alg, pk, sk);
  cout<<"keypair_ret="<<ret<<endl;

  ret = OQS_KEM_encaps(alg, (uint8_t*)ct, ss, pk);
  cout<<"encaps_ret="<<ret<<endl;

  //sig[0] = 0;
  ret = OQS_KEM_decaps(alg, ss, ct, sk);
  cout<<"verify_ret="<<ret<<endl;
  
  OQS_KEM_free(alg);

  return 0;
  }*/

int main()
{
  int i, cnt;
  const char * alg_name;
  const uint8_t msg[10000] = "HelloWorld";
  uint8_t ss[];
  uint8_t pk[10000], sk[10000];
  size_t mlen, siglen;
  OQS_SIG *alg;
  OQS_STATUS ret;
  cnt = OQS_KEM_alg_count();
  for(i = 0; i < cnt; i++)
  {
      alg_name = OQS_KEM_alg_identifier(i);
      printf("%d %s %d\n", i, alg_name, OQS_KEM_alg_is_enable(alg_name));
  }

  alg = OQS_SIG_new("aigis");
  if(alg == NULL)
  {
      printf("err\n");
      exit(-1);
  }
  ret = OQS_KEM_keypair(alg, pk, sk);
  printf("keypair_ret=%d\n", ret);

  ret = OQS_KEM_sign(alg, (uint8_t*)&sig, &siglen, msg, mlen, sk);
  printf("sign_ret=%d\n", ret);

  //sig[0] = 0;
  ret = OQS_SIG_verify(alg, msg, mlen, sig, siglen, pk);
  printf("vrfy_ret=%d\n");
  OQS_SIG_free(alg);

  return 0;
}
