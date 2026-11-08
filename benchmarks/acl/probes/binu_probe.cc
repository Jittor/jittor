// aclnnMul vs aclnnAbs at the same size, distinct output buffers per iteration
// so neither gets an unfair cache advantage.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_add.h>
#include <aclnnop/aclnn_abs.h>
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
  const int64_t n = 4*1024*1024/4;   // 4 MB
  const int K = 32;                  // distinct outputs, like the framework benches
  int64_t shp[1]={n}, st[1]={1};
  void *xd,*yd; CK(aclrtMalloc(&xd,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
  CK(aclrtMalloc(&yd,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
  std::vector<void*> outs(K); std::vector<aclTensor*> OT(K);
  for(int i=0;i<K;i++){ CK(aclrtMalloc(&outs[i],n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    OT[i]=aclCreateTensor(shp,1,ACL_FLOAT,st,0,ACL_FORMAT_ND,shp,1,outs[i]); }
  aclTensor* X=aclCreateTensor(shp,1,ACL_FLOAT,st,0,ACL_FORMAT_ND,shp,1,xd);
  aclTensor* Y=aclCreateTensor(shp,1,ACL_FLOAT,st,0,ACL_FORMAT_ND,shp,1,yd);
  aclScalar* one=aclCreateScalar((void*)&(*(new float(1.0f))), ACL_FLOAT);
  auto bench=[&](const char* tag,int kind){
    for(int rep=0;rep<2;rep++){
      int it=rep?K*4:K; aclrtSynchronizeStream(stream); double t=now_us();
      for(int i=0;i<it;i++){
        uint64_t s=0; aclOpExecutor* ex=nullptr; aclnnStatus r;
        aclTensor* O=OT[i%K];
        if(kind==0) r=aclnnAbsGetWorkspaceSize(X,O,&s,&ex);
        else if(kind==1) r=aclnnMulGetWorkspaceSize(X,Y,O,&s,&ex);
        else r=aclnnAddGetWorkspaceSize(X,Y,one,O,&s,&ex);
        if(r){printf("  %-24s ws %d\n",tag,(int)r);return;}
        void* w=ws(s);
        if(kind==0) r=aclnnAbs(w,s,ex,stream);
        else if(kind==1) r=aclnnMul(w,s,ex,stream);
        else r=aclnnAdd(w,s,ex,stream);
        if(r){printf("  %-24s launch %d\n",tag,(int)r);return;}
      }
      aclrtSynchronizeStream(stream);
      if(rep) printf("  %-24s %8.2f us  %6.0f GB/s\n",tag,(now_us()-t)/it,
        (kind?3.0:2.0)*n*4/((now_us()-t)/it)*1e-3);
    }
  };
  printf("== 4 MB, %d distinct output buffers ==\n",K);
  bench("aclnnAbs (unary)",0);
  bench("aclnnMul (x,y)",1);
  bench("aclnnAdd (x,y,alpha=1)",2);
  return 0;
}
