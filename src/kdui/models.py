
from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, JsonValue

API_PREFIX = "/api/v1"

REQUEST_KEY_HEADER = "X-KD-Key"


class ApiModel(BaseModel):

    model_config = ConfigDict(extra="forbid", use_attribute_docstrings=True)







class JobKind(StrEnum):

    FIT = "fit"
    VERIFY = "verify"
    COMPARE = "compare"


class JobQueue(StrEnum):

    FIT = "fit"
    LIGHT = "light"


QUEUE_OF_KIND: dict[JobKind, JobQueue] = {
    JobKind.FIT: JobQueue.FIT,
    JobKind.VERIFY: JobQueue.LIGHT,
    JobKind.COMPARE: JobQueue.LIGHT,
}


class JobState(StrEnum):

    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    BUDGET_STOPPED = "budget_stopped"
    CANCELLED = "cancelled"
    FAILED = "failed"
    INTERRUPTED = "interrupted"


TERMINAL_STATES: frozenset[JobState] = frozenset(
    {
        JobState.COMPLETED,
        JobState.BUDGET_STOPPED,
        JobState.CANCELLED,
        JobState.FAILED,
        JobState.INTERRUPTED,
    }
)


class JobPhase(StrEnum):

    STARTING = "starting"
    PREPARING = "preparing"
    SEARCHING = "searching"
    REPORTING = "reporting"
    WORKING = "working"


class StopReason(StrEnum):

    ITERATIONS_EXHAUSTED = "iterations_exhausted"
    ALGORITHM_DONE = "algorithm_done"
    EARLY_STOPPING = "early_stopping"
    USER_STOP = "user_stop"
    WALL_CLOCK_BUDGET = "wall_clock_budget"
    WALL_CLOCK_BUDGET_BEFORE_SEARCH = "wall_clock_budget_before_search"
    EVALUATION_BUDGET = "evaluation_budget"
    FINISHED = "finished"
    CANCELLED_WHILE_QUEUED = "cancelled_while_queued"
    KILLED = "killed"
    ERROR = "error"
    HARD_TIMEOUT = "hard_timeout"
    MEMORY_LIMIT = "memory_limit"
    SYSTEM_KILLED = "system_killed"
    TERMINATED = "terminated"
    PROCESS_LOST = "process_lost"
    MACHINE_RESTARTED = "machine_restarted"


class MemoryCapKind(StrEnum):

    CGROUP = "cgroup"
    JOB_OBJECT = "job_object"
    SAMPLED = "sampled"


class MemoryMeasure(StrEnum):

    RSS = "rss"
    COMMIT = "commit"


class ErrorCode(StrEnum):

    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    NOT_FOUND = "not_found"
    CONFLICT = "conflict"
    NOT_READY = "not_ready"
    INVALID_REQUEST = "invalid_request"
    KD_ERROR = "kd_error"
    KD_UNAVAILABLE = "kd_unavailable"
    PAYLOAD_TOO_LARGE = "payload_too_large"


class WaitingCode(StrEnum):

    SLOT = "slot"
    AHEAD = "ahead"
    MEMORY = "memory"


class RunStatus(StrEnum):

    COMPLETED = "completed"
    RAISED = "raised"
    NO_RECORD = "no_record"
    UNSEALED = "unsealed"


class RunOrigin(StrEnum):

    UI = "ui"
    IMPORTED = "imported"


class ArtifactKind(StrEnum):

    REPORT = "report"
    FIGURE = "figure"
    DATA = "data"
    RECORD = "record"
    LOG = "log"


class VerificationCheck(StrEnum):

    FORWARD_ERROR = "forward_error"
    ABLATION = "ablation"
    BLOCK_DISPERSION = "block_dispersion"


ScoreDirection = Literal["min", "max"]






class ErrorResponse(ApiModel):

    code: ErrorCode
    message: str
    field: str | None = None


class NativeScore(ApiModel):

    value: float | None
    kind: str
    direction: ScoreDirection


class EquationTerm(ApiModel):

    latex: str
    coefficient: float


class Equation(ApiModel):

    latex: str
    lhs_latex: str
    terms: list[EquationTerm]


class PriorKnowledge(ApiModel):

    sketch: bool







class SessionRequest(ApiModel):

    token: str


class SessionInfo(ApiModel):

    workspace: str
    version: str


class SessionOpened(SessionInfo):

    request_key: str


class MemoryBudget(ApiModel):

    total_bytes: int
    available_bytes: int
    reserve_bytes: int
    supervisor_allowance_bytes: int
    running_cap_bytes: int
    admissible_bytes: int
    default_fit_cap_bytes: int
    default_light_cap_bytes: int


class QueueStatus(ApiModel):

    queue: JobQueue
    slots: int
    running: int
    queued: int


class SystemInfo(ApiModel):

    hostname: str
    platform: str
    wsl: bool
    cpu_count: int
    versions: dict[str, str]
    workspace: str
    memory: MemoryBudget
    memory_cap_kind: MemoryCapKind | None
    queues: list[QueueStatus]
    warnings: list[str]







class ParameterField(ApiModel):

    name: str
    json_kind: str | None
    default: JsonValue
    literal_values: list[JsonValue] | None
    nullable: bool
    required: bool
    settable_from: str
    facade_param: str | None
    resume_tier: str
    knob: bool
    description: str | None
    object_fields: list[ParameterField] | None


class FacadeParameter(ApiModel):

    name: str
    kind: str
    default: JsonValue
    literal_values: list[JsonValue] | None
    config_field: str | None
    effect: str


class ModeInfo(ApiModel):

    name: str
    description: str
    forms: list[str]
    topologies: list[str]
    lhs_orders: list[int]
    provider_kind: str | None
    derivative_providers: list[str]
    selected_by: dict[str, JsonValue]
    required_params: list[str]
    refused_fields: dict[str, str]
    default_overrides: dict[str, JsonValue]
    sketch: dict[str, str]
    selectable_from_data: bool


class Segmentation(ApiModel):

    archive: str
    unit: str
    reseed: bool


class AlgorithmInfo(ApiModel):

    algorithm: str
    summary: str
    cost_class: str
    score_kind: str
    score_direction: ScoreDirection
    one_shot: bool
    segmentation: Segmentation
    modes: list[ModeInfo]
    fields: list[ParameterField]
    facade_params: list[FacadeParameter]
    warnings: list[str]


class GpuInfo(ApiModel):

    index: int
    name: str
    memory_bytes: int


class EnvironmentInfo(ApiModel):

    cuda_available: bool
    gpus: list[GpuInfo]
    memory_cap_kind: MemoryCapKind
    memory_cap_reason: str | None


class CapabilitySnapshot(ApiModel):

    refreshed_at: AwareDatetime
    kd_version: str
    schema_digest: str
    algorithms: list[AlgorithmInfo]
    environment: EnvironmentInfo


class Capabilities(ApiModel):

    snapshot: CapabilitySnapshot | None







class CatalogDataset(ApiModel):

    catalog_id: str
    topology: str
    axes: list[str]
    lhs: str | None
    file_format: str
    tier: str
    tags: list[str]


class ArrayInfo(ApiModel):

    key: str
    shape: list[int]
    dtype: str
    min: float | None
    max: float | None
    nan_count: int
    inf_count: int


class SourceInspection(ApiModel):

    container: str
    arrays: list[ArrayInfo]
    layouts: list[str]


class Upload(ApiModel):

    upload_id: str
    filename: str
    size_bytes: int
    sha256: str | None
    path: str | None
    created_at: AwareDatetime


class PathImport(ApiModel):

    path: str


class LoadRecipe(ApiModel):

    kind: Literal["pde", "tabular", "trajectory"] | None = None
    coords: dict[str, str] | None = None
    fields: dict[str, str] | None = None
    field_axes: list[str] | None = None
    select: dict[str, int] | None = None
    layout: str | None = None
    lhs: str | None = None
    periodic: list[str] | None = None
    trajectory_id: str | None = None
    params: dict[str, str] | None = None
    target_state: str | None = None
    trim: int | None = None


class CatalogSource(ApiModel):

    origin: Literal["catalog"] = "catalog"
    catalog_id: str


class UploadSource(ApiModel):

    origin: Literal["upload"] = "upload"
    upload_id: str


DatasetSource = Annotated[CatalogSource | UploadSource, Field(discriminator="origin")]


class DatasetRegistration(ApiModel):

    source: DatasetSource
    recipe: LoadRecipe = LoadRecipe()
    label: str | None = None


class ReferenceEquation(ApiModel):

    latex: str


class ReferenceToggle(ApiModel):

    shown: bool


class AxisReport(ApiModel):

    name: str
    n: int
    min: float | None
    max: float | None
    spacing: str
    periodic: bool
    step_first: float | None
    step_mean: float | None
    step_min: float | None
    step_max: float | None


class FieldReport(ApiModel):

    name: str
    dtype: str
    shape: list[int]
    min: float | None
    max: float | None
    mean: float | None
    nan_count: int
    inf_count: int
    noise_estimate: dict[str, float | None]
    first_derivative_gap: dict[str, float | None]
    second_derivative_gap: dict[str, float | None]
    constant_axes: list[str] | None = None
    noise_limited_axes: list[str] | None = None


class DatasetReport(ApiModel):

    name: str
    topology: str
    axes: list[AxisReport] | None
    fields: list[FieldReport] | None
    lhs_field: str | None
    lhs_axis: str | None
    lhs_label: str | None
    lhs_order: int
    warnings: list[str]


class ModeApplicability(ApiModel):

    algorithm: str
    mode: str
    accepts: bool
    reason: str | None
    params: dict[str, JsonValue]


class SliceSpec(ApiModel):

    field: str
    axes: list[str] = Field(min_length=1, max_length=2)
    fixed: dict[str, int] = {}


class SamplingInfo(ApiModel):

    method: Literal["none", "stride", "random"]
    n_total: int
    n_shown: int
    stride: dict[str, int] | None


class FieldSlice(ApiModel):

    dataset_id: str
    fingerprint: str
    field: str
    quantity_latex: str
    topology: str
    axes: list[str]
    coordinates: dict[str, list[float]]
    values: list[float | None] | list[list[float | None]]
    fixed: dict[str, int]
    full_shape: list[int]
    sampling: SamplingInfo
    missing_count: int


class DatasetPreview(ApiModel):

    created_at: AwareDatetime
    fingerprint: str
    report: DatasetReport
    applicability: list[ModeApplicability]
    slices: list[SliceSpec]


class DatasetSummary(ApiModel):

    dataset_id: str
    label: str
    source: DatasetSource
    created_at: AwareDatetime
    fingerprint: str | None
    reference_available: bool
    reference_shown: bool


class DatasetDetail(DatasetSummary):

    recipe: LoadRecipe
    reference_equation: ReferenceEquation | None


class DatasetList(ApiModel):

    catalog: list[CatalogDataset]
    registered: list[DatasetSummary]







class LowpassSmoothing(ApiModel):

    method: Literal["lowpass"] = "lowpass"
    modes: int | dict[str, int]
    axes: list[str] | None = None
    time_window: int = 0
    polyorder: int = 3


class NetworkSmoothing(ApiModel):

    method: Literal["network"] = "network"
    seed: int = 0
    recipe: dict[str, JsonValue] | None = None


Smoothing = Annotated[
    LowpassSmoothing | NetworkSmoothing, Field(discriminator="method")
]


class Preprocessing(ApiModel):

    smoothing: Smoothing | None = None


class FitRequest(ApiModel):

    kind: Literal["fit"] = "fit"
    dataset_id: str
    algorithm: str
    mode: str | None = None
    params: dict[str, JsonValue] = {}
    seeds: list[int] = Field(min_length=1)
    device: str = "cpu"
    preprocessing: Preprocessing = Preprocessing()
    sketch: dict[str, JsonValue] | None = None
    max_seconds: float | None = None
    hard_seconds: float | None = Field(default=None, gt=0)
    memory_cap_bytes: int | None = None
    checkpoint_every: int | None = None
    schema_digest: str
    note: str | None = None


class PreprocessPreviewRequest(ApiModel):

    dataset_id: str
    preprocessing: Preprocessing = Preprocessing()
    savgol_window: int = 0
    derivatives: str = "finite_diff"
    region: SliceSpec


class PreflightRequest(ApiModel):

    dataset_id: str
    algorithm: str
    mode: str | None = None
    params: dict[str, JsonValue] = {}
    device: str = "cpu"
    preprocessing: Preprocessing = Preprocessing()
    sketch: dict[str, JsonValue] | None = None
    schema_digest: str


class VerifyRequest(ApiModel):

    kind: Literal["verify"] = "verify"
    run_id: str
    checks: list[VerificationCheck] = Field(min_length=1)
    max_seconds: float | None = None


class CompareRequest(ApiModel):

    kind: Literal["compare"] = "compare"
    run_ids: list[str] = Field(min_length=2)


class HistoryImportRequest(ApiModel):

    path: str


JobRequest = Annotated[
    FitRequest | VerifyRequest | CompareRequest,
    Field(discriminator="kind"),
]


class ResumeRequest(ApiModel):

    iteration: int | None = None
    params: dict[str, JsonValue] = {}
    reseed: bool = False
    seed: int | None = None
    max_seconds: float | None = None
    hard_seconds: float | None = Field(default=None, gt=0)
    memory_cap_bytes: int | None = None







class WaitingReason(ApiModel):

    code: WaitingCode
    message: str
    needed_bytes: int | None = None
    admissible_bytes: int | None = None


class JobProgress(ApiModel):

    iteration: int | None
    best_score: NativeScore | None
    best_equation_latex: str | None
    elapsed_seconds: float | None


class Lineage(ApiModel):

    resumed_from_run_id: str | None = None
    resume_iteration: int | None = None
    reseed: bool = False
    cloned_from_run_id: str | None = None


class JobLimits(ApiModel):

    memory_cap_bytes: int
    memory_cap_kind: MemoryCapKind | None
    max_seconds: float | None
    hard_seconds: float | None


class JobSummary(ApiModel):

    job_id: str
    kind: JobKind
    queue: JobQueue
    state: JobState
    phase: JobPhase | None
    stop_reason: StopReason | None
    created_at: AwareDatetime
    started_at: AwareDatetime | None
    ended_at: AwareDatetime | None
    waiting: WaitingReason | None
    dataset_id: str | None
    algorithm: str | None
    seed: int | None
    run_id: str | None
    progress: JobProgress | None
    limits: JobLimits
    memory_bytes: int | None
    hostname: str | None
    warnings: list[str]


class ProcessUsage(ApiModel):

    returncode: int | None
    wallclock_seconds: float
    peak_memory_bytes: int
    memory_measure: MemoryMeasure


class JobError(ApiModel):

    error_type: str
    message: str


class JobDetail(JobSummary):

    request: JobRequest
    lineage: Lineage | None
    prior_knowledge: PriorKnowledge | None
    process: ProcessUsage | None
    error: JobError | None
    stop_requested_at: AwareDatetime | None
    kill_requested_at: AwareDatetime | None
    result_available: bool


class JobList(ApiModel):

    jobs: list[JobSummary]
    next_cursor: str | None


class JobAccepted(ApiModel):

    jobs: list[JobSummary]


class PhaseChange(ApiModel):

    event: Literal["phase"] = "phase"
    at: AwareDatetime
    phase: JobPhase


class IterationUpdate(ApiModel):

    event: Literal["iteration"] = "iteration"
    iteration: int
    elapsed_seconds: float
    n_candidates: int
    n_invalid: int
    n_unique_candidates: int | None
    mean_complexity: float | None
    best_score: NativeScore
    best_equation_latex: str | None


ProgressEvent = Annotated[PhaseChange | IterationUpdate, Field(discriminator="event")]


class EventsPage(ApiModel):

    events: list[ProgressEvent]
    next_cursor: str
    has_more: bool
    reset: bool
    skipped: int


class LogPage(ApiModel):

    text: str
    next_cursor: str
    has_more: bool
    reset: bool







class FitResult(ApiModel):

    kind: Literal["fit"] = "fit"
    run_id: str


class SpectrumCurve(ApiModel):

    axis: str
    missed: list[float]


class PreprocessPreview(ApiModel):

    dataset_id: str
    field: str
    derivatives: str
    noise_estimate: dict[str, float | None]
    noise_estimate_after: dict[str, float | None] | None
    spectrum: list[SpectrumCurve]
    field_before: FieldSlice
    field_after: FieldSlice | None
    lhs_before: FieldSlice | None
    lhs_after: FieldSlice | None
    warnings: list[str]


class PreflightIssue(ApiModel):

    message: str
    field: str | None


class PreflightResult(ApiModel):

    ok: bool
    refusals: list[PreflightIssue]
    warnings: list[str]
    resolved_mode: str | None


class ForwardErrorResult(ApiModel):

    value: float | None
    definition: str
    reason: str | None
    warning: str | None
    seconds: float
    figure_artifact_id: str | None


class AblationRow(ApiModel):

    dropped_latex: str
    mse: float
    nmse: float
    delta_mse: float
    relative_rise: float | None


class AblationResult(ApiModel):

    baseline_mse: float
    baseline_nmse: float
    rows: list[AblationRow]


class BlockDispersionResult(ApiModel):

    value: float
    n_blocks: int
    axis: str | None


class VerifyResult(ApiModel):

    kind: Literal["verify"] = "verify"
    run_id: str
    forward_error: ForwardErrorResult | None
    ablation: AblationResult | None
    block_dispersion: BlockDispersionResult | None
    unavailable: dict[VerificationCheck, str]


class ComparisonRow(ApiModel):

    run_id: str
    algorithm: str
    seed: int | None
    equation: Equation | None
    native_score: NativeScore | None
    nmse: float | None
    complexity: int | None
    prior_knowledge: PriorKnowledge | None
    preprocessing: Preprocessing | None
    savgol_window: int | None
    error: str | None


class CompareResult(ApiModel):

    kind: Literal["compare"] = "compare"
    dataset_fingerprint: str
    frame: str
    rows: list[ComparisonRow]
    figure_artifact_id: str | None


class HistorySource(ApiModel):

    source_id: str
    path: str
    imported_at: AwareDatetime
    n_runs: int
    n_unsealed: int


JobResult = Annotated[
    FitResult | VerifyResult | CompareResult,
    Field(discriminator="kind"),
]






class RunMetrics(ApiModel):

    nmse: float | None
    mse: float | None
    r2: float | None
    complexity: int | None


class RunCost(ApiModel):

    wallclock_seconds: float | None
    search_seconds: float | None
    preprocessing_seconds: float | None
    surrogate_train_seconds: float | None
    cpu_seconds: float | None
    tokens_in: int | None
    tokens_out: int | None
    peak_rss_bytes: int | None
    device: str | None
    iterations: int | None
    max_iterations: int | None


class Convergence(ApiModel):

    score_kind: str
    score_direction: ScoreDirection
    iterations: list[int]
    best_score: list[float | None]


class Candidate(ApiModel):

    rank: int
    equation_latex: str
    score: float | None
    complexity: int | None


class CandidateList(ApiModel):

    form: Literal["top_k", "pareto"]
    score_kind: str
    score_direction: ScoreDirection
    iteration: int | None
    iterations: list[int]
    rows: list[Candidate]


class SketchClause(ApiModel):

    clause: str
    level: str
    satisfied: bool | None
    detail: str | None


class SketchOutcome(ApiModel):

    solution: Equation | None
    best_candidate: Equation | None
    clauses: list[SketchClause]
    failure: str | None


class Provenance(ApiModel):

    kd_version: str | None
    config_hash: str | None
    record_hash: str | None
    hostname: str | None
    run_dir: str
    environment: dict[str, str] | None


class RunAvailability(ApiModel):

    record: bool
    convergence: bool
    candidates: bool
    report: bool
    result_file: bool
    checkpoints: bool


class RunSummary(ApiModel):

    run_id: str
    origin: RunOrigin
    job_id: str | None
    source_id: str | None
    kd_run_id: str | None
    status: RunStatus
    created_at: AwareDatetime | None
    algorithm: str | None
    seed: int | None
    dataset_id: str | None
    dataset_name: str | None
    dataset_fingerprint: str | None
    loaded_fingerprint: str | None
    preprocessing: Preprocessing | None
    savgol_window: int | None
    equation_latex: str | None
    native_score: NativeScore | None
    nmse: float | None
    stop_reason: str | None
    prior_knowledge: PriorKnowledge | None
    lineage: Lineage | None


class RunDetail(RunSummary):

    equation: Equation | None
    metrics: RunMetrics | None
    cost: RunCost | None
    config: dict[str, JsonValue] | None
    convergence: Convergence | None
    candidates: CandidateList | None
    sketch_outcome: SketchOutcome | None
    provenance: Provenance
    availability: RunAvailability
    request: FitRequest | None


class RunList(ApiModel):

    runs: list[RunSummary]
    next_cursor: str | None
    warnings: list[str] = []


class CheckpointInfo(ApiModel):

    iteration: int
    size_bytes: int | None


class CheckpointList(ApiModel):

    run_id: str
    resumable: bool
    reason: str | None
    unit: str | None
    checkpoints: list[CheckpointInfo]
    resume_safe_params: list[str]


class VerificationList(ApiModel):

    verifications: list[VerifyResult]


class HistorySourceList(ApiModel):

    sources: list[HistorySource]







class Artifact(ApiModel):

    artifact_id: str
    owner: Literal["job", "run"]
    owner_id: str
    name: str
    kind: ArtifactKind
    media_type: str
    size_bytes: int


class ArtifactList(ApiModel):

    artifacts: list[Artifact]











class CatalogRequest(ApiModel):
    pass


class CatalogListing(ApiModel):

    datasets: list[CatalogDataset]


class ReferenceRequest(ApiModel):

    catalog_id: str


class InspectionRequest(ApiModel):

    path: str


class DatasetPreviewRequest(ApiModel):

    workspace: str
    dataset_id: str


class SliceRequest(ApiModel):

    workspace: str
    dataset_id: str
    spec: SliceSpec
    max_points: int
    max_cells: int


class CapabilitiesRequest(ApiModel):
    pass


class PreflightCall(ApiModel):

    workspace: str
    request: PreflightRequest


class PreprocessPreviewCall(ApiModel):

    workspace: str
    request: PreprocessPreviewRequest
    max_points: int
    max_cells: int


class RunProjectionRequest(ApiModel):

    run_id: str
    origin: RunOrigin
    run_dir: str
    job_dir: str | None
    source_id: str | None


class CandidatesRequest(ApiModel):

    run_dir: str
    iteration: int | None


class CheckpointsRequest(ApiModel):

    run_id: str
    run_dir: str


class HistoryScanRequest(ApiModel):

    source_id: str
    path: str
    imported_at: AwareDatetime


class RunListRequest(ApiModel):

    workspace: str
    sources: list[HistorySource]







class StoredJob(ApiModel):

    scheme: Literal["kdui-job-v1"] = "kdui-job-v1"
    job_id: str
    kind: JobKind
    queue: JobQueue
    created_at: AwareDatetime
    request: JobRequest
    memory_cap_bytes: int
    hard_seconds: float | None
    lineage: Lineage | None = None
    prior_knowledge: PriorKnowledge | None = None
    reference_in_report: bool = False
    warnings: list[str] = []


class Marker(ApiModel):

    at: AwareDatetime


class LaunchRecord(ApiModel):

    at: AwareDatetime
    hostname: str
    supervisor_allowance_bytes: int
    warnings: list[str] = []


class ProcessIdentity(ApiModel):

    pid: int
    create_time: float


class SupervisorStarted(ApiModel):

    record: Literal["supervisor_started"] = "supervisor_started"
    at: AwareDatetime
    process: ProcessIdentity
    boot_id: str
    boot_time: float
    hostname: str


class WorkerStarted(ApiModel):

    record: Literal["worker_started"] = "worker_started"
    at: AwareDatetime
    process: ProcessIdentity
    memory_cap_kind: MemoryCapKind


class WorkerEnded(ApiModel):

    record: Literal["worker_ended"] = "worker_ended"
    at: AwareDatetime
    returncode: int | None
    ended_by: Literal["exit", "stop", "kill"]
    cause: Literal["timeout", "requested", "memory_limit", "system_kill"] | None
    wallclock_seconds: float
    peak_memory_bytes: int
    memory_measure: MemoryMeasure


StateRecord = Annotated[
    SupervisorStarted | WorkerStarted | WorkerEnded,
    Field(discriminator="record"),
]


class JobOutcome(ApiModel):

    at: AwareDatetime
    status: Literal["completed", "raised"]
    kd_stop_reason: str | None = None
    kd_run_id: str | None = None
    error: JobError | None = None


class DatasetRecord(ApiModel):

    scheme: Literal["kdui-dataset-v1"] = "kdui-dataset-v1"
    dataset_id: str
    label: str
    source: DatasetSource
    recipe: LoadRecipe
    name: str
    created_at: AwareDatetime
    reference_available: bool


class UploadRecord(ApiModel):

    scheme: Literal["kdui-upload-v1"] = "kdui-upload-v1"
    upload_id: str
    filename: str
    suffix: str
    size_bytes: int
    sha256: str | None
    path: str | None = None
    created_at: AwareDatetime
