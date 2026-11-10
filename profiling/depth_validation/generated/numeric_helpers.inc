static json compare_ports(const json &expected,const json &actual) {
    require(expected.size()==actual.size(),"port identity changed");double maximum_abs=0,maximum_relative=0;size_t values=0;
    for(const auto &item:expected.items()) {
        require(actual.contains(item.key())&&actual.at(item.key()).size()==item.value().size(),"port layout changed");
        for(size_t i=0;i<item.value().size();++i) {
            const double a=item.value().at(i),b=actual.at(item.key()).at(i);require(std::isfinite(a)&&std::isfinite(b),"non-finite output");
            const double difference=std::abs(a-b);maximum_abs=std::max(maximum_abs,difference);maximum_relative=std::max(maximum_relative,difference/std::max(std::abs(a),1e-30));
            require(difference<=5e-5+5e-5*std::abs(a),"timed result differs from identical untimed replay");++values;
        }
    }
    return json{{"passed",true},{"checked_values",values},{"maximum_absolute_error",maximum_abs},{"maximum_relative_error",maximum_relative},{"atol",5e-5},{"rtol",5e-5},
        {"reference_scope","same_native_executor_untimed_full_replay; independent_scalar_or_NumPy_oracle_required_externally"}};
}
struct InputSnapshot {std::string id,location;std::vector<uint8_t> bytes;};
static std::vector<InputSnapshot> inputs(Executor &executor,const json &action) {
    const auto &op=executor.operations.at(str(action,"target_id"));const std::string location=action.at("device")=="cpu"?"dram":"vram";
    std::set<std::string> outputs={str(op,"output")};for(const auto &id:op.value("extra_outputs",json::array()))outputs.insert(id);
    std::vector<InputSnapshot> result;
    for(const auto &id:op.at("inputs")) {
        require(!outputs.count(id.get<std::string>()),"isolated action aliases an input/output");
        auto *tensor=executor.tensor(id,location);InputSnapshot value{id,location,std::vector<uint8_t>(ggml_nbytes(tensor))};
        ggml_backend_tensor_get(tensor,value.bytes.data(),0,value.bytes.size());result.push_back(std::move(value));
    }
    return result;
}
static json verify_inputs(Executor &executor,const std::vector<InputSnapshot> &snapshots) {
    json result=json::array();for(const auto &value:snapshots) {
        std::vector<uint8_t> observed(value.bytes.size());ggml_backend_tensor_get(executor.tensor(value.id,value.location),observed.data(),0,observed.size());
        require(observed==value.bytes,"target input mutated during timed diagnostic");
        result.push_back(json{{"tensor_id",value.id},{"location",value.location},{"storage_bytes",value.bytes.size()},{"exact_bytes_unchanged",true}});
    }return result;
}
