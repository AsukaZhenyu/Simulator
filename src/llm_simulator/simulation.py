"""A small public facade over the existing compiler and common runtime."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import shutil
import tempfile

from llm_infer_model.tensor.environment import EnvironmentEvent
from llm_infer_model.tensor.spec import InvalidInput, Unsupported
from tensor_mapping.artifacts import write_json
from tensor_mapping.python_frontend import ModelPlan, Scheduling

from .analytical import AnalyticalCosts
from .config import Hardware, Model, Workload


def _empty_directory(path):
    path = Path(path).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise InvalidInput(f"directory must be new or empty: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


class Simulator:
    """Describe one finite LLM workload and compile different strategies.

    No hardware is auto-detected and no cost source is selected implicitly.
    This simulates dependencies and resource use, not tensor numeric values or
    language generation. ModelPlan remains available for externally supplied
    exact costs and all existing advanced execution transforms.
    """
    def __init__(self, *, model: Model, hardware: Hardware, workload: Workload,
                 cost_model: AnalyticalCosts):
        for value, expected, label in (
            (model, Model, "model"), (hardware, Hardware, "hardware"),
            (workload, Workload, "workload"),
        ):
            if not isinstance(value, expected):
                raise InvalidInput(f"{label} must be {expected.__name__}")
        if not isinstance(cost_model, AnalyticalCosts):
            raise Unsupported("choose AnalyticalCosts explicitly; for measured/exact costs use ModelPlan")
        self.model, self.hardware = model, hardware
        self.workload, self.cost_model = workload, cost_model

    def compile(self, *, scheduling: Scheduling, configure=None, directory=None,
                environment_events=(), max_decisions=100000,
                visibility_mode="data_dependencies"):
        """Generate exact inputs and freeze a strategy through the real compiler.

        configure(plan) may call module/place/transform/prepare/runtime_policy.
        A persistent directory is recommended when saving checkpoints. With no
        directory, the returned object owns its temporary input files until
        close(). No generated input depends on repository fixtures/profiles.
        """
        if not isinstance(scheduling, Scheduling):
            raise InvalidInput("scheduling must be Scheduling")
        if configure is not None and not callable(configure):
            raise InvalidInput("configure must be callable or None")
        if type(max_decisions) is not int or max_decisions < 1:
            raise InvalidInput("max_decisions must be a positive integer")
        if not isinstance(environment_events, tuple) or any(
                not isinstance(event, EnvironmentEvent) for event in environment_events):
            raise InvalidInput("environment_events must be a tuple of EnvironmentEvent")
        if scheduling.reconfiguration is not None:
            raise Unsupported("resource reclaim control needs explicit control costs; use ModelPlan with environment-trace-0.2")
        temporary = tempfile.TemporaryDirectory(prefix="llm-simulator-") if directory is None else None
        root = Path(temporary.name) if temporary else _empty_directory(directory)
        try:
            inputs = self.cost_model.build_inputs(root, self.model, self.hardware, self.workload)
            write_json(Path(inputs["environment_file"]), dict(
                format_version="environment-trace-0.1",
                events=[{key: value for key, value in asdict(event).items() if value is not None}
                        for event in environment_events],
            ))
            plan = ModelPlan(id="library-simulation", model_kind=inputs["model_kind"],
                model_file=inputs["model_file"], environment_file=inputs["environment_file"],
                base_dir=root, scheduling=scheduling, max_decisions=max_decisions,
                visibility_mode=visibility_mode)
            if configure is not None:
                configure(plan)
            compiled = plan.compile()
            metadata = dict(inputs.get("metadata", {}))
            metadata.update(cost_model=inputs.get("cost_model", "analytical_roofline"),
                            hardware_measurement=False, calibrated=False,
                            input_directory=str(root.resolve()))
            write_json(root / "library-inputs.json", metadata)
            return CompiledSimulation(compiled, root, metadata, temporary)
        except BaseException:
            if temporary is not None:
                temporary.cleanup()
            raise

    def run(self, *, scheduling: Scheduling, **compile_options):
        """Compile and run; the result retains the compiled input lifetime."""
        return self.compile(scheduling=scheduling, **compile_options).run()


class CompiledSimulation:
    """Own generated inputs and the existing immutable CompiledPlan."""
    def __init__(self, plan, directory, metadata, temporary):
        self.plan = plan
        self.directory = Path(directory)
        self.metadata = metadata
        self._temporary = temporary
        self._closed = False

    def _check_open(self):
        if self._closed:
            raise InvalidInput("compiled simulation is closed; generate or load a new project")

    def run(self, *, checkpoint=None, max_decisions=None):
        self._check_open()
        raw = self.plan.run(checkpoint=checkpoint, max_decisions=max_decisions)
        return SimulationResult(self, raw)

    def ir_document(self):
        self._check_open()
        return self.plan.ir_document()

    def close(self):
        if not self._closed and self._temporary is not None:
            self._temporary.cleanup()
        self._closed = True

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *args):
        self.close()


class SimulationResult:
    """Convenient metrics plus the complete causal execution for inspection."""
    def __init__(self, compiled, raw):
        self.compiled = compiled
        self.raw = raw

    @property
    def evaluation(self):
        return self.raw[0].evaluation

    @property
    def status(self):
        return self.evaluation.status

    @property
    def makespan_ns(self):
        return self.evaluation.makespan_ns

    @property
    def makespan_ms(self):
        return None if self.makespan_ns is None else self.makespan_ns / 1_000_000

    @property
    def events(self):
        return self.evaluation.events

    @property
    def success(self):
        return self.status == "valid" and self.raw[0].termination_reason == "goal_reached"

    def summary(self):
        evaluation, run = self.evaluation, self.raw[0]
        metadata = self.compiled.metadata
        return dict(status=self.status, success=self.success,
                    termination_reason=run.termination_reason,
                    compilation_error=run.compilation_error,
                    error_code=evaluation.error_code,
                    makespan_ns=self.makespan_ns, makespan_ms=self.makespan_ms,
                    peak_dram_bytes=evaluation.peak_dram_bytes,
                    peak_vram_bytes=evaluation.peak_vram_bytes,
                    h2d_bytes=evaluation.h2d_bytes, d2h_bytes=evaluation.d2h_bytes,
                    ssd_read_bytes=evaluation.ssd_read_bytes,
                    ssd_write_bytes=evaluation.ssd_write_bytes,
                    decision_count=len(run.decisions),
                    cost_model=metadata["cost_model"], calibrated=False,
                    hardware_measurement=False,
                    provenance=metadata.get("provenance", "synthetic_layer"),
                    macro_contract=metadata.get("macro_contract"),
                    numerical_llm_execution=False,
                    input_directory=str(self.compiled.directory.resolve()))

    def checkpoint(self):
        """Same-source replay; keep the project and use this compiled object."""
        self.compiled._check_open()
        return self.compiled.plan.checkpoint(self.raw)

    def save(self, output):
        """Write detailed IR/events/state/results without overwriting files.

        A copy of generated input JSON is included for auditing. Checkpoint
        fingerprints still refer to the original input paths: keep the
        persistent compile(directory=...) project to resume. This is not an
        executable export of arbitrary Python RuntimePolicy classes.
        """
        self.compiled._check_open()
        output = Path(output).resolve()
        source = self.compiled.directory.resolve()
        if output == source or source in output.parents or output in source.parents:
            raise InvalidInput("result directory must be separate from the input project")
        self.compiled.plan.write_result(output, self.raw)
        write_json(output / "library-summary.json", self.summary())
        shutil.copytree(source, output / "input-snapshot")
        return output
