#include <vector>
#include <functional>

namespace jittor {

    namespace torch {
        struct Tensor;
    }

    namespace at {
        template<typename T>
        class ArrayRef {
            public:
                std::vector<std::reference_wrapper<T> > containers;
                ArrayRef() {containers.clear();}
                ArrayRef(T& item) { 
                    containers.clear();
                    containers.push_back(item);
                }
                ArrayRef(const std::initializer_list<T> &Vec) {
                    containers.clear();
                    for(T item : Vec)
                        containers.push_back(item);
                }
                ArrayRef(std::vector<T> &Vec) {
                    containers.clear();
                    for(T& item : Vec)
                        containers.push_back(item);
                }
                ~ArrayRef() { containers.clear(); }
                typedef typename std::vector<std::reference_wrapper<T> >::iterator it; 
                it begin() { return containers.begin(); }
                it end() { return containers.end(); }
        };

        namespace cuda {
            bool check_device(std::vector<torch::Tensor>);
        }
    }
}