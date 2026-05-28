# TEPA Dataset Generation Plan: Condition Fan-Out

This document describes how to generate the puck-world training data efficiently while testing a central TEPA claim: the same intended condition can be represented many ways, and the model should learn that those representations point to the same conditioned outcome.

The dataset should be built around canonical semantics first, then rendered into many condition surfaces.

```text
scene specification
  -> canonical intervention event
  -> simulated conditioned outcome
  -> many condition renderings that express the same event
```

The important rule is that templates do not define the physics. Templates only express an already-defined intervention.

## Goals

The generation system should:

- Simulate each physical event once and fan it out into many equivalent condition surfaces.
- Store canonical meaning separately from surface form.
- Support natural language, structured text, semi-structured text, visual, and demonstration-style conditions.
- Make train/test splits that can hold out templates, surface families, layouts, magnitudes, and object references.
- Keep data generation deterministic, reproducible, and cheap enough to regenerate often.
- Preserve enough metadata to debug failures by semantic event, surface type, template, and simulator seed.

## Canonical Records

Use Pydantic models for serialized boundary records. Convert them to NumPy arrays or tensors before the training hot path.

### SceneSpec

The scene is the unconditioned world state.

```python
class SceneSpec(BaseModel):
    scene_id: int
    scene_hash: str
    seed: int
    image_size: int
    horizon: int
    pucks: list[PuckSpec]
    walls: WallSpec
    obstacles: list[ObstacleSpec]
    goal_zones: list[GoalZoneSpec]
    friction: float
    restitution: float
```

### InterventionEvent

The event is the canonical condition. Every condition surface for this event should mean the same thing.

```python
class InterventionEvent(BaseModel):
    event_id: int
    scene_id: int
    event_hash: str
    target_object_id: int
    target_object_aliases: list[str]
    impulse: tuple[float, float]
    impulse_frame: int = 0
    horizon: int
    coordinate_frame: Literal["screen_xy", "world_xy"]
    magnitude_bucket: Literal["tiny", "small", "medium", "large"]
    direction_bucket: str
```

### OutcomeBundle

The target is generated from the scene plus event. It should be independent of how the condition is phrased.

```python
class OutcomeBundle(BaseModel):
    outcome_id: int
    event_id: int
    outcome_hash: str
    final_positions_path: str
    trajectory_path: str
    heatmap_path: str
    event_flags_path: str
    time_to_contact_path: str
    rollout_seed: int
    simulator_version: str
```

### ConditionRendering

The rendering is one expression of the event.

```python
class ConditionRendering(BaseModel):
    condition_id: int
    event_id: int
    rendering_hash: str
    family: str
    template_id: str
    renderer_version: str
    payload_type: Literal[
        "text",
        "structured_text",
        "image",
        "trace",
        "mixed"
    ]
    payload_path: str | None
    payload_inline: str | None
    metadata: dict[str, Any]
```

## Generation Pipeline

Use a four-stage pipeline.

### 1. Sample Scenes

Generate scene specs from seeded distributions:

```text
num_pucks
puck colors and aliases
puck positions and radii
obstacles
goal zones
friction
restitution
rendering style
```

Render each scene once into a context image and optionally store the compact state vector.

```text
scene_id -> context_image
scene_id -> context_state
```

### 2. Sample Canonical Events

For each scene, sample many interventions:

```text
target object
impulse direction
impulse magnitude
horizon
coordinate frame
```

Example event:

```json
{
  "event_id": 830214,
  "scene_id": 1042,
  "target_object_id": 2,
  "target_object_aliases": ["blue puck", "blue", "puck 2"],
  "impulse": [-1.75, -0.85],
  "horizon": 48,
  "coordinate_frame": "screen_xy",
  "magnitude_bucket": "medium",
  "direction_bucket": "upper-left"
}
```

### 3. Simulate Once Per Event

Run the deterministic simulator once for each `(scene_id, event_id)` pair.

```text
scene + canonical event -> rollout -> outcome bundle
```

The outcome bundle is reused for every condition rendering of that event. Do not rerun the simulator for every natural-language phrasing or structured format.

### 4. Fan Out Condition Renderings

For each event, sample condition renderings from a large template bank.

```text
event -> text prompt A
event -> text prompt B
event -> JSON
event -> YAML
event -> TOML
event -> Markdown list
event -> HTML list
event -> CSV row
event -> key-colon-value text
event -> arrow mask
event -> demo trace
```

Training can either materialize these renderings ahead of time or generate some of them lazily from `(event_id, template_id)` during data loading.

## Identity, Cache Keys, and Dedupe

Add cache keys from the start. This is not just an optimization; it protects the validation set from accidental duplicate states produced by different generation runs.

Use multiple hashes because different layers have different identity rules:

```text
scene_hash      hash(canonical SceneSpec without ids, seeds, paths, or timestamps)
event_hash      hash(scene_hash + canonical InterventionEvent without ids or paths)
outcome_hash    hash(event_hash + simulator_version + target_config_hash)
rendering_hash  hash(event_hash + renderer_version + template_id + rendered payload)
```

The seed should be stored for reproducibility, but it should not be the identity. Two seeds can produce the same state, and one seed can produce different data after a sampler change. The hash should come from a canonical, sorted, quantized representation of the actual scene or event.

Recommended canonicalization:

```text
sort objects by stable object id
sort obstacles and goal zones by stable id
round or quantize floats to the simulator precision
serialize with sorted keys and no whitespace dependence
exclude generated ids, local paths, creation time, and run id
include simulator config only where it changes the artifact
```

For example, scene identity should include puck positions, radii, colors, walls, obstacles, goals, friction, and restitution. It should not include `scene_id`. Event identity should include the selected object, impulse, horizon, coordinate frame, and `scene_hash`. It should not include `event_id`.

Maintain a lightweight dedupe index while generating:

```text
dedupe_index.sqlite
  scenes(scene_hash unique, scene_id, split)
  events(event_hash unique, event_id, scene_hash)
  outcomes(outcome_hash unique, outcome_id, event_hash)
  renderings(rendering_hash unique, condition_id, event_hash, family, template_id)
```

SQLite is a good first choice because unique constraints make accidental duplicates obvious. A JSONL or Parquet index is fine later, but SQLite keeps the first implementation simple and safe.

Generation behavior:

```text
if scene_hash exists:
  reuse scene_id and context arrays

if event_hash exists:
  reuse event_id

if outcome_hash exists:
  reuse rollout targets

if rendering_hash exists:
  skip duplicate condition row
```

This allows repeated generation runs to add genuinely new examples without silently duplicating old ones.

## Split Assignment

Assign train/validation/test splits from stable hashes, not from row order or generation run.

For scene-level splits:

```text
split = hash(scene_hash + split_salt) % split_buckets
```

All events and condition renderings for a held-out scene should remain held out. This prevents the model from seeing the same physical state during training and validation under different phrasing.

For template and surface-form tests, the scene can be shared only when that is the point of the test:

```text
held-out template split:
  same event_hash may appear in train and test
  train and test use disjoint template groups
  metric isolates condition-surface generalization

held-out scene split:
  scene_hash never crosses train/test
  metric isolates physical generalization
```

Keep these split types separate in reports. A shared-event held-out-template test is useful, but it should never be mistaken for a held-out-world validation score.

## Condition Surface Families

### Natural Language

Natural language templates should be numerous and deliberately varied. They should cover:

- Imperatives: "Push the blue puck up-left with medium force."
- Questions: "Where will the blue puck end up after a medium push toward the upper-left?"
- Prediction requests: "Predict the scene 48 frames after nudging the blue puck northwest."
- Concise commands: "Blue puck, medium northwest impulse, 48 frames."
- Verbose descriptions: "Apply a moderate impulse to the blue puck, aimed toward the upper-left corner, then forecast the state after 48 frames."
- Relative descriptions: "Move the blue puck away from the lower-right wall with medium strength."
- Coordinate descriptions: "Apply impulse dx=-1.75, dy=-0.85 to puck 2."
- Object-reference variants: color, index, spatial relation, alias, or label.
- Horizon variants: frames, steps, ticks, or short phrases such as "after the rollout."

A natural-language template should be compiled from smaller controlled pieces:

```text
verb phrase      push, nudge, strike, tap, launch, apply an impulse to
object phrase    the blue puck, puck 2, the puck near the bottom wall
direction        up-left, northwest, toward the upper-left corner
magnitude        tiny, slight, medium, strong, magnitude 1.95
horizon          48 frames, 48 simulation steps, after the rollout
task framing     predict, forecast, estimate, determine
```

This gives a large surface set while keeping the underlying semantics exact.

### Structured Text

Structured renderings should preserve the same event in formats that an LLM-like encoder might reasonably interpret as context.

JSON:

```json
{"object":"blue puck","impulse":{"dx":-1.75,"dy":-0.85},"horizon_frames":48}
```

YAML:

```yaml
object: blue puck
impulse:
  dx: -1.75
  dy: -0.85
horizon_frames: 48
```

TOML:

```toml
object = "blue puck"
horizon_frames = 48

[impulse]
dx = -1.75
dy = -0.85
```

CSV:

```csv
object,dx,dy,horizon_frames
blue puck,-1.75,-0.85,48
```

Key-colon-value:

```text
object: blue puck
dx: -1.75
dy: -0.85
horizon: 48 frames
```

Markdown list:

```markdown
- object: blue puck
- direction: upper-left
- strength: medium
- horizon: 48 frames
```

HTML list:

```html
<ul>
  <li>object: blue puck</li>
  <li>direction: upper-left</li>
  <li>strength: medium</li>
  <li>horizon: 48 frames</li>
</ul>
```

Other useful structured surfaces:

```text
URL query string: object=blue&dx=-1.75&dy=-0.85&horizon=48
function call: predict_push(object="blue", dx=-1.75, dy=-0.85, horizon=48)
CLI flags: --object blue --dx -1.75 --dy -0.85 --horizon 48
Markdown table
plain English plus a small parameter table
```

### Visual Conditions

Visual surfaces should express the same canonical event without text:

- Arrow mask from the selected puck in the impulse direction.
- Overlay image with selected puck highlighted and arrow drawn.
- Direction-only arrow plus separate strength indicator.
- Before/after ghost of the first few simulated frames.
- Small trajectory prefix showing only the immediate effect of the impulse.

Store these as arrays in the condition shard, or render them lazily when cheap.

### Demonstration Conditions

Demo traces can condition the model by showing a short prefix trajectory.

```text
trace positions for first 3 to 8 frames
selected object id or object mask
optional velocity deltas
optional noisy demonstration variant
```

The target remains the full rollout. The demo trace is just another way to express the event.

### Mixed Conditions

Mixed renderings combine formats:

```text
natural language + JSON block
Markdown list + arrow image
HTML list + short trace
CSV row + text task framing
```

Mixed conditions are useful later, but the first benchmark should keep them in a separate split or low sampling rate so they do not blur early results.

## Template Bank

Store templates as versioned data, not as scattered string literals.

```text
experiments/puck_world/templates/
  natural_language.yaml
  structured_text.yaml
  visual.yaml
  mixed.yaml
```

Each template should declare what fields it uses and what split it belongs to.

```yaml
template_id: nl_prediction_042
family: natural_language
split_group: train
required_fields:
  - object_alias
  - direction_phrase
  - magnitude_phrase
  - horizon_phrase
template: "Forecast the puck world {horizon_phrase} after a {magnitude_phrase} push sends {object_alias} {direction_phrase}."
```

Use held-out template groups for evaluation:

```text
train templates
validation templates
held-out paraphrase templates
held-out structured formats
held-out object-reference styles
```

This prevents the model from merely memorizing a small phrase set.

## Sampling Strategy

For each scene:

```text
sample 8 to 64 canonical events
simulate each event once
sample 2 to 8 condition renderings per event for training
sample more renderings per event for evaluation
```

Use weighted sampling by family:

```text
natural_language    40%
structured_text     30%
visual              20%
demo_trace           8%
mixed                2%
```

The exact percentages should be configurable. Early experiments can start with structured text plus natural language, then add visual and demo surfaces once the simulator and storage are stable.

## Storage Layout

Avoid duplicating context images and outcome bundles for every condition rendering.

Recommended layout:

```text
data/processed/puck-v0.1/
  dataset_config.yaml
  templates/
    natural_language.yaml
    structured_text.yaml
  manifests/
    scenes.jsonl
    events.jsonl
    outcomes.jsonl
    conditions_train.jsonl
    conditions_val.jsonl
    conditions_test.jsonl
  arrays/
    scenes/
      shard_00000.npz
    outcomes/
      shard_00000.npz
    conditions/
      shard_00000.npz
```

Scene shard arrays:

```text
scene_id          int64   [N]
scene_hash        utf8    [N] or manifest field
context_image     uint8   [N, H, W, 3]
context_state     float32 [N, state_dim]
```

Outcome shard arrays:

```text
event_id          int64   [N]
event_hash        utf8    [N] or manifest field
outcome_hash      utf8    [N] or manifest field
target_final_pos  float32 [N, num_pucks, 2]
target_traj       float32 [N, horizon, num_pucks, 2]
target_heatmap    float32 [N, num_pucks, H, W]
target_events     float32 [N, event_dim]
target_ttc        float32 [N]
```

Condition shard arrays:

```text
condition_id      int64   [N]
event_id          int64   [N]
rendering_hash    utf8    [N] or manifest field
family_id         int64   [N]
template_id       int64   [N]
text_bytes        object or external utf8 file
arrow_image       uint8   optional [N, H, W, 1]
demo_trace        float32 optional [N, T, trace_dim]
condition_params  float32 optional [N, param_dim]
```

For the first implementation, it is acceptable to use `.npz` shards plus JSONL manifests. If text arrays become awkward, store text payloads in JSONL and keep numeric payloads in `.npz`.

## Dataset Loader

The PyTorch dataset should join by ids:

```text
condition_id -> event_id
event_id -> scene_id
event_id -> outcome
scene_id -> context
```

At batch time:

```text
context = scene_store[scene_id]
condition = condition_store[condition_id]
target = outcome_store[event_id]
```

This keeps the dataset compact and makes multi-query evaluation natural.

## Split Design

The splits should test generalization at several levels.

Train:

```text
seen scenes
seen simulator distributions
seen template groups
seen condition families
seen magnitude buckets
```

Validation:

```text
same distribution as train
different seeds
different scene_hash values
```

Held-out template split:

```text
same scenes and event distribution
new natural-language templates
new structured text layouts
```

Held-out surface split:

```text
train on JSON/YAML/text
test on TOML/CSV/HTML/key-value
```

Held-out semantics split:

```text
new magnitudes
new horizons
new directions
new object counts
new obstacle layouts
```

Multi-query split:

```text
same scene
many events
many condition renderings per event
small semantic changes between nearby events
```

This last split is especially important for TEPA because it directly tests whether the context vector can be reused while the condition vector shifts the prediction. It should also drive an amortized runtime benchmark. For each selected scene, evaluate batches with `K = 1, 4, 16, 64, 256` different conditions. TEPA should encode the scene once and reuse `z_context`; the fused context-stuffing baseline should re-encode the combined context-condition input for each condition. Report both quality and systems metrics:

```text
final-position MSE
trajectory MSE
target-latent MSE
condition-shuffle degradation
wall-contact F1
total latency per scene
latency per condition
peak memory
```

This benchmark records the more precise current TEPA premise: comparable prediction quality may still be valuable if the separated architecture is more reusable, inspectable, and efficient when many counterfactual perspectives are evaluated over the same world state.

The first generated version of this split is JSON-only so it can evaluate the current JSON-trained predictors without adding a surface-family generalization confound:

```text
configs/counterfactual_eval_json.yaml
data/processed/puck-v0.1-counterfactual-eval-json/
```

It has 256 scenes, 64 events per scene, 4 equivalent JSON renderings per event, and 65,536 total condition rows. The event mix is 50% nearby force/direction grid, 25% object-binding swaps, and 25% random filler. Surface generalization should be tested separately with a mixed-format counterfactual split after the JSON-only benchmark is understood.

## Quality Checks

Run automated checks after generation:

- No `scene_hash` appears in both train and held-out-scene validation/test splits.
- No duplicate `event_hash` appears inside a split unless intentionally materialized for condition fan-out.
- No duplicate `rendering_hash` appears anywhere in the dataset version.
- Every condition rendering points to an existing event.
- Every event points to one scene and one outcome.
- Equivalent condition renderings for the same event share the same target.
- Near-neighbor events with different impulses usually produce different targets.
- Held-out templates do not appear in the train split.
- Structured renderings parse successfully when they are meant to be parseable.
- Rendered arrows and demo traces agree with the canonical impulse direction.
- Dataset statistics are balanced across object colors, directions, magnitudes, horizons, and surface families.

## Recommended First Milestone

Start with a small deterministic dataset:

```text
1,000 scenes
16 events per scene
4 condition renderings per event
64,000 condition examples
64x64 context images
40-frame horizon
1 to 2 pucks
natural language + JSON + YAML + key-value
```

Then add:

```text
arrow masks
demo traces
held-out templates
held-out structured formats
larger scene layouts
```

The first useful result does not require a huge dataset. It requires a clean semantic hierarchy: scenes, canonical events, outcome bundles, and many condition renderings that faithfully express the same event.
