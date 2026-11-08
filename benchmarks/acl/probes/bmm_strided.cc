// Can aclnnBatchMatMul consume a permuted (strided) view directly, and is it
// as fast as materialising the permute first?
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_batch_matmul.h>
#include <aclnnop/aclnn_permute.h>
#include <chrono>
#include <cstdio>
#include <vector>
static aclrtStream stream; static void* wsp=nullptr; static uint64_t wsps=0;
static double now_us(){using namespace std::chrono;
  return duration<double,std::micro>(steady_clock::now().time_since_epoch()).count();}
static void* ws(uint64_t n){ if(n>wsps){ if(wsp) aclrtFree(wsp);
  aclrtMalloc(&wsp,n,ACL_MEM_MALLOC_HUGE_FIRST); wsps=n;} return n?wsp:nullptr; }
#define CK(e) do{auto r=(e); if(r!=0){printf("FAIL %s -> %d\n",#e,(int)r); return 1;}}while(0)

int main(){
  CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&stream));
  // attention: q,k are [B,T,H,hd] in memory, logically [B*H, T, hd]
  const int64_t B=8,T=256,H=8,hd=64;
  const int64_t bat=B*H;
  int64_t nq=B*T*H*hd;
  void *qd,*kd,*od,*qcd;
  CK(aclrtMalloc(&qd,nq*4,ACL_MEM_MALLOC_HUGE_FIRST));
  CK(aclrtMalloc(&kd,nq*4,ACL_MEM_MALLOC_HUGE_FIRST));
  CK(aclrtMalloc(&qcd,nq*4,ACL_MEM_MALLOC_HUGE_FIRST));
  CK(aclrtMalloc(&od,bat*T*T*4,ACL_MEM_MALLOC_HUGE_FIRST));
  // contiguous [B,H,T,hd] -> flat [B*H,T,hd]
  int64_t c3[3]={bat,T,hd}, c3s[3]={T*hd,hd,1};
  int64_t k3t[3]={bat,hd,T}, k3ts[3]={T*hd,1,hd};   // transposed last two, still contiguous storage
  int64_t o3[3]={bat,T,T},  o3s[3]={T*T,T,1};
  // strided view of a [B,T,H,hd] buffer as [B*H,T,hd]:
  // element (b,h,t,d) lives at ((b*T + t)*H + h)*hd + d
  // as [B,H,T,hd] the strides are (T*H*hd, hd, H*hd, 1); flattening B,H needs
  // a 4-D descriptor, so keep it 4-D and let BMM batch over the leading two.
  int64_t v4[4]={B,H,T,hd}, v4s[4]={T*H*hd, hd, H*hd, 1};
  int64_t k4[4]={B,H,hd,T}, k4s[4]={T*H*hd, hd, 1, H*hd};
  int64_t o4[4]={B,H,T,T},  o4s[4]={H*T*T, T*T, T, 1};
  int64_t flat[1]={nq};
  aclTensor* Qc = aclCreateTensor(c3,3,ACL_FLOAT,c3s,0,ACL_FORMAT_ND,c3,3,qcd);
  aclTensor* Kc = aclCreateTensor(k3t,3,ACL_FLOAT,k3ts,0,ACL_FORMAT_ND,c3,3,kd);
  aclTensor* Oc = aclCreateTensor(o3,3,ACL_FLOAT,o3s,0,ACL_FORMAT_ND,o3,3,od);
  aclTensor* Qv = aclCreateTensor(v4,4,ACL_FLOAT,v4s,0,ACL_FORMAT_ND,flat,1,qd);
  aclTensor* Kv = aclCreateTensor(k4,4,ACL_FLOAT,k4s,0,ACL_FORMAT_ND,flat,1,kd);
  aclTensor* Ov = aclCreateTensor(o4,4,ACL_FLOAT,o4s,0,ACL_FORMAT_ND,o4,4,od);
  auto bench=[&](const char* tag, aclTensor* A, aclTensor* Bt, aclTensor* O){
    uint64_t s=0; aclOpExecutor* ex=nullptr;
    auto st=aclnnBatchMatMulGetWorkspaceSize(A,Bt,O,1,&s,&ex);
    if(st){printf("  %-34s workspace FAILED %d\n",tag,(int)st); return;}
    for(int rep=0;rep<2;rep++){
      int it=rep?100:10; aclrtSynchronizeStream(stream); double t=now_us();
      for(int i=0;i<it;i++){
        uint64_t s2=0; aclOpExecutor* e2=nullptr;
        auto r=aclnnBatchMatMulGetWorkspaceSize(A,Bt,O,1,&s2,&e2);
        if(r){printf("  %-34s ws %d\n",tag,(int)r);return;}
        void* w=ws(s2); r=aclnnBatchMatMul(w,s2,e2,stream);
        if(r){printf("  %-34s launch FAILED %d\n",tag,(int)r);return;}
      }
      aclrtSynchronizeStream(stream);
      if(rep) printf("  %-34s %8.2f us\n",tag,(now_us()-t)/it);
    }
  };
  printf("== BMM [%ld,%ld,%ld] x [%ld,%ld,%ld] ==\n",bat,T,hd,bat,hd,T);
  bench("contiguous 3-D operands",Qc,Kc,Oc);
  bench("permuted strided 4-D views",Qv,Kv,Ov);
  return 0;
}
