#pragma once
#include "core/common.h"
#include "aclnn/aclnn.h"
#include <acl/acl.h>
#include <functional>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace jittor {

// Only grouped unary/binary runners query through the registry. Other runners
// already own their typed SDK queries and use the registry for availability.
// No per-operator function slots, signature casts or SDK-dependent overload set.
struct AclWorkspaceArguments {
    aclTensor* x = nullptr;
    aclTensor* y = nullptr;
    aclTensor* output = nullptr;
    aclScalar* alpha = nullptr;
    aclDataType dtype = ACL_FLOAT;
};

class AclOpFunctions {
public:
    enum class QueryKind { Direct, Unary, Cast, Binary, Add };
    using Execute = aclnnStatus (*)(void*, uint64_t, aclOpExecutor*, aclrtStream);
    using Query = std::function<aclnnStatus(const AclWorkspaceArguments&,
                                          uint64_t*, aclOpExecutor**)>;

    template<class Function>
    static AclOpFunctions unary(Function query, Execute execute) {
        return AclOpFunctions([query](const AclWorkspaceArguments& args,
                                    uint64_t* size, aclOpExecutor** executor) {
            return query(args.x, args.output, size, executor);
        }, execute, QueryKind::Unary);
    }

    template<class Function>
    static AclOpFunctions cast(Function query, Execute execute) {
        return AclOpFunctions([query](const AclWorkspaceArguments& args,
                                    uint64_t* size, aclOpExecutor** executor) {
            return query(args.x, args.dtype, args.output, size, executor);
        }, execute, QueryKind::Cast);
    }

    template<class Function>
    static AclOpFunctions binary(Function query, Execute execute) {
        return AclOpFunctions([query](const AclWorkspaceArguments& args,
                                    uint64_t* size, aclOpExecutor** executor) {
            return query(args.x, args.y, args.output, size, executor);
        }, execute, QueryKind::Binary);
    }

    template<class Function>
    static AclOpFunctions add(Function query, Execute execute) {
        return AclOpFunctions([query](const AclWorkspaceArguments& args,
                                    uint64_t* size, aclOpExecutor** executor) {
            return query(args.x, args.y, args.alpha, args.output, size, executor);
        }, execute, QueryKind::Add);
    }

    static AclOpFunctions direct(Execute execute) {
        return AclOpFunctions({}, execute, QueryKind::Direct);
    }

    aclnnStatus workspace(const AclWorkspaceArguments& args, uint64_t* size,
                          aclOpExecutor** executor) const {
        if (!query_)
            throw std::logic_error("ACL runner owns its workspace query; no grouped query registered");
        return query_(args, size, executor);
    }

    Execute launcher() const { return execute_; }
    bool has_grouped_query() const { return bool(query_); }
    bool supports(QueryKind kind) const { return kind_ == kind; }

private:
    AclOpFunctions(Query query, Execute execute, QueryKind kind)
        : query_(std::move(query)), execute_(execute), kind_(kind) {
        if (!execute_) throw std::invalid_argument("ACL launcher must not be null");
    }

    Query query_;
    Execute execute_;
    QueryKind kind_;
};

using AclOpRegistry = std::unordered_map<std::string, AclOpFunctions>;
// One immutable, lazily initialized table in acl_jittor.cc, shared by all TUs.
EXTERN_LIB const AclOpRegistry& acl_op_registry();

} // namespace jittor
