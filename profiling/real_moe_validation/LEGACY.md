# The engine-timed probes are a record, not the instrument

`native/`, `native_attn/`, `tap_native/` and `tap_native_attn/` hold the four probe
sources that include `llama.h` and call `llama_decode`. They are the only files in this
repository that depend on a runtime other than ggml. They are kept because they produced
the published tables and dropping the sources would leave those numbers with no
reconstruction path.

They are no longer the model column and are not on the default path of
`tools/run_accuracy_campaign.py`: the column is now `model-ggml`
(`profiling/ir_ggml/model_column.py`), which lowers the plan itself and executes it on
ggml against the model file's own weight bytes — no engine between the plan and the
kernel. The default `--columns` no longer names `model`, and `model-ggml` needs none of
this directory.

What that changes about a reading: the old column asked "does the simulator predict this
runtime", and the answer depended on that runtime's scheduler. The new column asks "does
the frozen price of this action describe what the action costs", on a graph whose
placement is the thing being priced. The two are different questions and their numbers
are not interchangeable.
