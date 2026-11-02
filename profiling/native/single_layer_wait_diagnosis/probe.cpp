// Diagnostic-only NVTX annotations around the unchanged GGML action body.
#define NOMINMAX
#if defined(NVTX_DISABLE) || defined(NVTX_NO_IMPL)
#error "This diagnostic requires enabled header-only NVTX implementation"
#endif
#include <nvtx3/nvToolsExt.h>
#include <nlohmann/json.hpp>
#include <cstddef>
#include <cstdint>
#include <string>

namespace wait_diag {
class Range {
public:
    Range(const char *message,uint32_t category) {
        nvtxEventAttributes_t event{};event.version=NVTX_VERSION;event.size=NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        event.category=category;event.messageType=NVTX_MESSAGE_TYPE_ASCII;event.message.ascii=message;
        nvtxRangePushEx(&event);
    }
    ~Range(){nvtxRangePop();}
    Range(const Range &)=delete;Range &operator=(const Range &)=delete;
};
class ActionRange {
    static const char *message(const nlohmann::json &action) {
        const auto &kind=action.at("kind").get_ref<const std::string&>();
        if(kind=="COMPUTE"||kind=="COPY")return action.at("target_id").get_ref<const std::string&>().c_str();
        return kind.c_str();
    }
    static uint32_t category(const nlohmann::json &action) {
        const auto &kind=action.at("kind").get_ref<const std::string&>();
        if(kind=="COMPUTE")return action.at("device").get_ref<const std::string&>()=="gpu"?11:10;
        if(kind=="COPY")return 20;
        return kind=="EVICT"?31:30;
    }
    Range range;
public:
    explicit ActionRange(const nlohmann::json &action):range(message(action),category(action)){}
};
class BlockRange {
    static const char *name(size_t index) {
        static constexpr const char *names[]={"pair_block_0","pair_block_1","pair_block_2","pair_block_3"};
        return names[index]; // Pinned pair executor always uses exactly four blocks.
    }
    Range range;
public:
    explicit BlockRange(size_t index):range(name(index),1){}
};
}

#include "pair_generated.inc"

int main(int argc,char **argv){try {
    require(argc==9,"usage: wait-diagnosis-probe PROGRAM.json context_vs_isolated|instrumentation TARGET_ACTION_INDEX WARMUP SAMPLES default|0 ABBA|BAAB OUTPUT.json");
    const std::string pair_kind=argv[2];require(pair_kind=="context_vs_isolated"||pair_kind=="instrumentation","unsupported pair kind");
    const int target=single_layer::integer(argv[3],0,10000,"target"),warmup=single_layer::integer(argv[4],0,10000,"warmup"),samples=single_layer::integer(argv[5],3,10000,"samples");
    const std::string poll=argv[6],order=argv[7];require(poll=="default"||poll=="0","pool poll must be explicit default or 0");
    require(order=="ABBA"||order=="BAAB","order must be explicit ABBA or BAAB");
    std::ifstream existing(argv[8],std::ios::binary);require(!existing.good(),"output already exists");
#if defined(_WIN32)
    _putenv_s("NVIDIA_TF32_OVERRIDE","0");_putenv_s("GGML_CUDA_DISABLE_GRAPHS","1");
#else
    setenv("NVIDIA_TF32_OVERRIDE","0",1);setenv("GGML_CUDA_DISABLE_GRAPHS","1",1);
#endif
    std::ifstream input(argv[1]);require(bool(input),"cannot read program");json program;input>>program;Executor executor(std::move(program));
    const auto pool=single_layer::configure_pool(executor,poll);auto result=context_pair::run(executor,pair_kind,static_cast<size_t>(target),warmup,samples,order,pool);
    result["nvtx_diagnostic"]=json{{"format_version","single-layer-wait-nvtx-v1"},{"enabled",true},{"header_only",true},
        {"block_range_names",json::array({"pair_block_0","pair_block_1","pair_block_2","pair_block_3"})},
        {"categories",json{{"block",1},{"cpu_compute",10},{"gpu_compute",11},{"copy",20},{"advance",30},{"evict",31}}},
        {"compute_message","borrowed_exact_action.target_id_string; no copied label or graph substitute"},
        {"target_range_count_per_block",1+warmup+samples},
        {"target_validation_range_count_per_block",1},{"target_warmup_range_count_per_block",warmup},
        {"target_timed_range_count_per_block",samples},
        {"target_selection_rule","Within each pair_block_N on the main thread, require exactly 1+warmup+samples GPU-compute ranges with the target operation_id; select the last samples ranges as timed. Reject missing, extra or dropped ranges."},
        {"readback_scope","port verification CUDA reads can occur inside block ranges but outside action ranges; exclude them from action attribution"},
        {"overlap_rule","inclusive CUDA API waits and GPU work overlap; do not sum them as disjoint costs"},
        {"runtime_loader_resolution_proven",false},{"usable_for_parameter_fit",false},{"usable_for_heldout_acceptance",false},
        {"limitation","NVTX and profiler change execution overhead; correlation attributes intervals, not a complete causal explanation"}};
    std::ofstream output(argv[8],std::ios::binary);require(bool(output),"cannot create output");output<<result.dump(2)<<'\n';output.close();require(bool(output),"output write failed");
    std::cout<<json{{"status","ok"},{"process_id",result.at("process_id")},{"pair_kind",pair_kind},{"order",order},{"blocks",4},{"diagnostic_only",true},{"nvtx_enabled",true}}.dump()<<'\n';return 0;
}catch(const std::exception &error){std::cerr<<"wait-diagnosis-probe: "<<error.what()<<'\n';return 2;}}
