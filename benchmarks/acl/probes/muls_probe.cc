// Does aclnnMuls (aclScalar) beat aclnnMul with a stride-0 broadcast tensor?
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_mul.h>
#include <aclnnop/aclnn_div.h>
#include <chrono>
#include <cstdio>
#include <vector>

static aclrtStream stream;
static void* wsp = nullptr; static uint64_t wsp_size = 0;
static double now_us(){using namespace std::chrono;
  return duration<double,std::micro>(steady_clock::now().time_since_epoch()).count();}
#define CK(e) do{auto r=(e); if(r!=0){printf("FAIL %s -> %d\n", #e, (int)r); return 1;}}while(0)

static void* ws(uint64_t n){ if(n>wsp_size){ if(wsp) aclrtFree(wsp);
  aclrtMalloc(&wsp,n,ACL_MEM_MALLOC_HUGE_FIRST); wsp_size=n;} return n?wsp:nullptr; }

int main(int argc,char**argv){
  CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&stream));
  for (int64_t MEL : {int64_t(262144), int64_t(4194304)}) {
    int64_t n = MEL; double mb = n*4/1e6;
    void *xd,*od,*sd;
    CK(aclrtMalloc(&xd,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&od,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&sd,32,ACL_MEM_MALLOC_HUGE_FIRST));
    float one=2.0f; CK(aclrtMemcpy(sd,4,&one,4,ACL_MEMCPY_HOST_TO_DEVICE));
    int64_t shp[1]={n}, st1[1]={1}, st0[1]={0}, sshp[1]={1};
    aclTensor* X = aclCreateTensor(shp,1,ACL_FLOAT,st1,0,ACL_FORMAT_ND,shp,1,xd);
    aclTensor* O = aclCreateTensor(shp,1,ACL_FLOAT,st1,0,ACL_FORMAT_ND,shp,1,od);
    // stride-0 broadcast of a single element, exactly what jittor hands aclnnMul
    aclTensor* B = aclCreateTensor(shp,1,ACL_FLOAT,st0,0,ACL_FORMAT_ND,sshp,1,sd);
    aclScalar* S = aclCreateScalar(&one, ACL_FLOAT);
    // the same one-element buffer described by its real shape, letting CANN broadcast
    aclTensor* B1 = aclCreateTensor(sshp,1,ACL_FLOAT,st1,0,ACL_FORMAT_ND,sshp,1,sd);
    auto bench=[&](const char*tag,int kind){
      for(int rep=0;rep<2;rep++){
        int iters = rep? 100: 10;
        aclrtSynchronizeStream(stream);
        double t=now_us();
        for(int i=0;i<iters;i++){
          uint64_t s=0; aclOpExecutor* ex=nullptr; aclnnStatus r;
          if(kind==4) r=aclnnMulGetWorkspaceSize(X,B1,O,&s,&ex);
          else if(kind==5) r=aclnnDivGetWorkspaceSize(X,B1,O,&s,&ex);
          else if(kind==0) r=aclnnMulGetWorkspaceSize(X,B,O,&s,&ex);
          else if(kind==1) r=aclnnMulsGetWorkspaceSize(X,S,O,&s,&ex);
          else if(kind==2) r=aclnnDivGetWorkspaceSize(X,B,O,&s,&ex);
          else r=aclnnDivsGetWorkspaceSize(X,S,O,&s,&ex);
          if(r!=0){printf("  %s workspace failed %d\n",tag,(int)r);return;}
          void* w=ws(s);
          if(kind==4) r=aclnnMul(w,s,ex,stream);
          else if(kind==5) r=aclnnDiv(w,s,ex,stream);
          else if(kind==0) r=aclnnMul(w,s,ex,stream);
          else if(kind==1) r=aclnnMuls(w,s,ex,stream);
          else if(kind==2) r=aclnnDiv(w,s,ex,stream);
          else r=aclnnDivs(w,s,ex,stream);
          if(r!=0){printf("  %s launch failed %d\n",tag,(int)r);return;}
        }
        aclrtSynchronizeStream(stream);
        double us=(now_us()-t)/iters;
        if(rep) printf("  %-34s %8.2f us  %6.0f GB/s\n",tag,us,2*mb/us*1e3);
      }
    };
    printf("== %.1f MB ==\n", mb);
    bench("aclnnMul  (stride-0 tensor)",0);
    bench("aclnnMuls (aclScalar)      ",1);
    bench("aclnnDiv  (stride-0 tensor)",2);
    bench("aclnnDivs (aclScalar)      ",3);
    bench("aclnnMul  (shape-[1] tensor)",4);
    bench("aclnnDiv  (shape-[1] tensor)",5);
    aclDestroyTensor(X);aclDestroyTensor(O);aclDestroyTensor(B);aclDestroyTensor(B1);aclDestroyScalar(S);
    aclrtFree(xd);aclrtFree(od);aclrtFree(sd);
  }
  return 0;
}
