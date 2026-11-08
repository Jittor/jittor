// aclnnPermute vs a strided->contiguous copy for the same permutation.
#include <acl/acl.h>
#include <aclnn/acl_meta.h>
#include <aclnnop/aclnn_permute.h>
#include <aclnnop/aclnn_copy.h>
#include <chrono>
#include <cstdio>
#include <vector>
static aclrtStream stream;
static void* wsp=nullptr; static uint64_t wsps=0;
static double now_us(){using namespace std::chrono;
  return duration<double,std::micro>(steady_clock::now().time_since_epoch()).count();}
static void* ws(uint64_t n){ if(n>wsps){ if(wsp) aclrtFree(wsp);
  aclrtMalloc(&wsp,n,ACL_MEM_MALLOC_HUGE_FIRST); wsps=n;} return n?wsp:nullptr; }
#define CK(e) do{auto r=(e); if(r!=0){printf("FAIL %s -> %d\n",#e,(int)r); return 1;}}while(0)

struct Case { const char* tag; std::vector<int64_t> shape; std::vector<int64_t> perm; };

int main(){
  CK(aclInit(nullptr)); CK(aclrtSetDevice(0)); CK(aclrtCreateStream(&stream));
  std::vector<Case> cases = {
    {"5D [8,256,3,8,64] (2,0,3,1,4)", {8,256,3,8,64}, {2,0,3,1,4}},
    {"5D [8,256,3,8,64] (2,0,1,3,4)", {8,256,3,8,64}, {2,0,1,3,4}},
    {"5D [3,8,8,256,64] (1,3,2,0,4)", {3,8,8,256,64}, {1,3,2,0,4}},
    {"4D [8,8,256,64]   (0,2,1,3)  ", {8,8,256,64},   {0,2,1,3}},
    {"4D [8,256,8,64]   (0,2,1,3)  ", {8,256,8,64},   {0,2,1,3}},
    {"4D [8,8,256,256]  (0,1,3,2)  ", {8,8,256,256},  {0,1,3,2}},
    {"4D [64,64,64,64]  (3,2,1,0)  ", {64,64,64,64},  {3,2,1,0}},
    {"3D [512,512,64]   (1,0,2)    ", {512,512,64},   {1,0,2}},
    {"3D [64,512,512]   (2,1,0)    ", {64,512,512},   {2,1,0}},
    {"2D [2048,2048]    (1,0)      ", {2048,2048},    {1,0}},
    {"2D [256,8192]     (1,0)      ", {256,8192},     {1,0}},
  };
  for (auto& c : cases) {
    int r = c.shape.size();
    int64_t n=1; for(auto s:c.shape) n*=s;
    double mb=n*4/1e6;
    std::vector<int64_t> src_stride(r), out_shape(r), out_stride(r), view_stride(r);
    src_stride[r-1]=1; for(int i=r-2;i>=0;i--) src_stride[i]=c.shape[i+1]*src_stride[i+1];
    for(int i=0;i<r;i++){ out_shape[i]=c.shape[c.perm[i]]; view_stride[i]=src_stride[c.perm[i]]; }
    out_stride[r-1]=1; for(int i=r-2;i>=0;i--) out_stride[i]=out_shape[i+1]*out_stride[i+1];
    void *xd,*od; CK(aclrtMalloc(&xd,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    CK(aclrtMalloc(&od,n*4,ACL_MEM_MALLOC_HUGE_FIRST));
    int64_t flat[1]={n};
    aclTensor* X   = aclCreateTensor(c.shape.data(),r,ACL_FLOAT,src_stride.data(),0,ACL_FORMAT_ND,c.shape.data(),r,xd);
    // the permuted *view* of the same buffer, plus a contiguous destination
    aclTensor* VIEW= aclCreateTensor(out_shape.data(),r,ACL_FLOAT,view_stride.data(),0,ACL_FORMAT_ND,flat,1,xd);
    aclTensor* OUT = aclCreateTensor(out_shape.data(),r,ACL_FLOAT,out_stride.data(),0,ACL_FORMAT_ND,out_shape.data(),r,od);
    auto bench=[&](const char* tag,int kind){
      for(int rep=0;rep<2;rep++){
        int it = rep?100:10;
        aclrtSynchronizeStream(stream); double t=now_us();
        for(int i=0;i<it;i++){
          uint64_t s=0; aclOpExecutor* ex=nullptr; aclnnStatus st;
          if(kind==0){ aclIntArray* d=aclCreateIntArray(c.perm.data(),r);
            st=aclnnPermuteGetWorkspaceSize(X,d,OUT,&s,&ex);
            if(st){printf("  %-26s workspace %d\n",tag,(int)st); aclDestroyIntArray(d); return;}
            void* w=ws(s); st=aclnnPermute(w,s,ex,stream); aclDestroyIntArray(d);
          } else {
            st=aclnnInplaceCopyGetWorkspaceSize(OUT,VIEW,&s,&ex);
            if(st){printf("  %-26s workspace %d\n",tag,(int)st); return;}
            void* w=ws(s); st=aclnnInplaceCopy(w,s,ex,stream);
          }
          if(st){printf("  %-26s launch %d\n",tag,(int)st); return;}
        }
        aclrtSynchronizeStream(stream);
        double us=(now_us()-t)/it;
        if(rep) printf("  %-26s %8.2f us  %6.0f GB/s\n",tag,us,2*mb/us*1e3);
      }
    };
    printf("== %s  %.1f MB ==\n", c.tag, mb);
    bench("aclnnPermute",0);
    bench("aclnnInplaceCopy(view->out)",1);
    aclDestroyTensor(X);aclDestroyTensor(VIEW);aclDestroyTensor(OUT);
    aclrtFree(xd);aclrtFree(od);
  }
  return 0;
}
