Yes—this is a much better rescue than returning to fixed head labels. Your own results suggest that physical implementation fragments across prompts and architectures, while the residual-level relational geometry may remain conserved. That makes the plausible object of study:

Residual ISA: a reusable set of latent-state transformations that are dynamically composed for each prompt and implemented by different heads/MLPs across models.

But do not claim you have recovered it from one RSA result yet. A cross-model RSA of ρ = 0.78 with a permutation p ≈ 10⁻⁶ is strong evidence of shared geometry, but it does not by itself prove that the geometry represents computational instructions, is causal, is complete, or generalizes beyond the eight categories. Your earlier pipeline already showed how compelling correlations can disappear under causal testing.

First: one statistical warning

Your 28 category pairs are not 28 independent samples. Each category participates in seven distances, so the entries of the representational dissimilarity matrix are coupled. Do not compute ordinary correlation significance or statistical power as though N = 28 independent observations.

Use:

a Mantel-style permutation test, permuting category labels as whole units;
bootstrap over prompts within categories;
bootstrap over category subsets;
preferably several independently generated dataset replicas;
confidence intervals over models, prompts and templates.

A permutation p-value can be valid if you permuted category identities correctly. The “88.8% power from 28 pairs” claim needs rechecking under this dependence structure.

What “complete Residual ISA” should mean

You cannot prove that you found “every function” in natural language; the possible task space is open-ended. You can instead establish a finite basis relative to a declared task universe.

A convincing Residual ISA needs to satisfy six properties:

Separability: different operations occupy distinguishable residual trajectories.
Invariance: paraphrases and token substitutions preserve the operation.
Cross-architecture alignment: relational structure transfers between models.
Causality: manipulating an operation representation selectively changes that operation.
Compositionality: complex tasks can be predicted from combinations of primitives.
Coverage: adding new task families eventually stops producing new primitives.

That last one is how you approach “nothing left out”: demonstrate a saturation curve, not an absolute claim.

The full research program
1. Build a computational ontology, not just eight task labels

Your current categories mix multiple levels:

Fact recall is a task.
NIAH is a retrieval setting.
Arithmetic is a domain containing many operations.
Copy is closer to a primitive.

You need a hierarchical ontology.

A. Memory and routing
exact token retrieval
semantic retrieval
associative lookup
positional lookup
recent-context lookup
long-range lookup
key–value binding
variable binding
dereferencing
pointer following
one-hop retrieval
multi-hop retrieval
source selection
distractor rejection
working-memory maintenance
working-memory update
memory overwrite
memory gating
B. Matching and comparison
equality
inequality
identity matching
semantic matching
greater-than / less-than
ordinal comparison
set membership
subset/superset
thresholding
nearest match
anomaly detection
contradiction detection
entailment discrimination
C. Transformation
copy
substitute
translate
normalize
inflect
reorder
reverse
rotate/permutate
map through a learned relation
encode
decode
format conversion
abstraction from instances
instantiation from a rule
D. Aggregation
count
sum
average
minimum
maximum
majority
conjunction
disjunction
set union/intersection
histogram/frequency
accumulation across positions
summarization/compression
E. Sequence operations
next-item prediction
induction
continuation
sorting
grouping
segmentation
boundary detection
bracket matching
sequence alignment
subsequence detection
repetition detection
temporal ordering
causal ordering
F. Symbolic and logical operations
negation
AND/OR/XOR
implication
modus ponens
transitive inference
quantifier handling
variable substitution
unification
constraint propagation
case splitting
consistency checking
proof-step selection
G. Linguistic operations
subject identification
predicate identification
syntactic attachment
agreement
coreference resolution
semantic-role assignment
relation extraction
entity typing
disambiguation
scope resolution
discourse tracking
pragmatic inference
H. Control-like operations
task recognition
instruction selection
branch/select
suppress
amplify
route
halt/commit
confidence sharpening
error correction
fallback
conflict resolution
output formatting

Do not assume these are all genuine primitives. They are the candidate instruction library to test.

2. Factor content away from computation

For every proposed operation, build prompts using interchangeable content.

For example, comparison should appear as:

numbers: 17 > 12
lengths: A is taller than B
dates: June occurs after April
arbitrary symbols: k precedes m
invented words: dax outranks wug

Likewise, retrieval should vary:

answer token identity
token frequency
needle position
context topic
query wording
delimiter style
language
output format

The goal is to estimate:

representation = operation + content + difficulty + format + position + confidence

Then isolate the operation component.

Use factorial datasets rather than a collection of hand-written examples:

operation × content domain × template × length × difficulty × answer format

Without this, RSA may simply detect shared vocabulary, prompt shape, sequence length or answer type.

3. Record trajectories, not single-layer vectors

The transformer residual stream is an additive communication channel through which heads and MLPs read and write, but different information can occupy different subspaces.

For every token and layer, collect:

pre-attention residual
post-attention residual
pre-MLP residual
post-MLP residual
normalized and unnormalized residual
attention update
MLP update
final answer-position trajectory
relevant source-token trajectories
pairwise token-to-token information transfer

Represent each computation as a trajectory:

T(x) = [r₀, r₁, …, rL]

Then compare:

trajectory direction
velocity: r(l+1) − r(l)
acceleration
curvature
path length
layer of maximum change
subspace transitions
attractor/fixed-point behavior
branch points
convergence to target
divergence under distractors

Recent work is already treating the layer sequence as a geometric trajectory, so your novelty cannot merely be “residuals have trajectories.” It must be the discovery and causal validation of a reusable operation basis.

4. Use stronger cross-model alignment than raw RSA

RSA is a good first test because it compares relational geometry rather than requiring neuron-by-neuron correspondence.

But run a battery:

RSA with cosine, correlation and Euclidean distance
centered kernel alignment
linear CKA
Procrustes alignment
canonical correlation analysis
singular-vector canonical correlation
optimal-transport alignment
representational topology/persistent homology
neighborhood preservation
cross-model k-NN category transfer
train-on-Qwen, decode-on-Llama after linear alignment
bidirectional transfer, not only one direction

The killer experiment is not merely correlated matrices. It is:

Learn an alignment on some operations, then correctly locate entirely unseen operations from the other model.

For example:

align Qwen and Llama using six operations;
hold out sorting and comparison;
test whether their relative positions are predicted in the aligned space.

That demonstrates out-of-sample structural correspondence.

5. Discover primitives without giving the model your labels

Supervised categories can force geometry into your ontology. Add unsupervised discovery:

cluster residual trajectory segments;
change-point detection over layers;
hidden Markov models or switching linear dynamical systems;
sparse dictionary learning;
residual-stream SAEs;
non-negative matrix factorization;
independent component analysis;
tensor decomposition across model × task × layer × token;
contrastive learning that holds operation constant and changes content;
minimum-description-length selection of primitive count.

Sparse autoencoders and dictionary learning are specifically used to decompose superposed model activations into learned features, although they do not automatically guarantee causal or complete explanations.

Ask whether unsupervised motifs line up with your human labels. A strong result is:

The model independently forms recurring trajectory motifs corresponding to operations such as retrieve, compare and copy—even when trained without operation labels.

6. Define an instruction mathematically

A residual instruction should not merely be a cluster label. Give it an operator definition.

Candidate forms:

Translation operator

r_out ≈ r_in + v_k

The computation adds a stable direction.

Linear operator

r_out ≈ A_k r_in + b_k

Low-rank operator

r_out ≈ r_in + U_k V_kᵀ r_in

Conditional operator

r_out ≈ Σ_k g_k(r_in, prompt) F_k(r_in)

where g_k dynamically selects or mixes instructions.

Token-coupled operator

r_t,out ≈ F_k(r_t,in, {r_i,in : i in source set})

This captures attention-mediated movement across token positions.

The most likely formulation is a conditional mixture of low-rank operators, not one fixed vector per task.

Your paper should test which family gives the best:

held-out reconstruction;
causal predictive power;
cross-model transfer;
compactness;
compositionality.
7. Separate state, operand, opcode and implementation

Borrow the ISA analogy carefully:

ISA concept	Transformer analogue
Opcode	latent operation/motif
Operand	entities, numbers, tokens, relations
Register/state	residual representation at token positions
Memory access	attention-mediated token transfer
ALU-like transform	MLP/head residual update
Instruction scheduler	input-conditioned gating/activation
Program	trajectory of operation mixtures
Hardware implementation	specific heads, MLPs and layers

Then test whether you can disentangle:

operation representation ⟂ operand identity

For example, the same COMPARE operation should work on numbers, dates and arbitrary symbols, while the operands change independently.

Use probes carefully: decodability does not imply causal use. Your own report already demonstrated this distinction repeatedly.

8. Causally transplant instructions

This is the most important section.

Create matched prompts:

clean prompt performs operation A;
corrupted prompt differs only in the required operation or operand.

Then intervene on candidate residual ISA components:

Patch

Patch the candidate operation state from one prompt into another.

Remove

Project out the operation direction/subspace.

Replace

Replace COMPARE with COPY, while preserving operands.

Scale

Increase/decrease operation activation.

Swap

Swap operation vectors between prompts while keeping content fixed.

Cross-model transplant

Map an instruction state from Qwen to Llama using learned alignment and test whether it changes behavior as predicted.

A valid instruction should satisfy:

necessity: removal harms the operation;
sufficiency: insertion induces or restores it;
specificity: unrelated tasks remain intact;
dose response: scaling produces graded effects;
counterfactual control: replacing it produces the predicted alternate computation.

Activation patching has important design and interpretation pitfalls, so use clean/corrupted prompts, matched controls, multiple metrics and restoration baselines.

9. Prove compositionality

A real ISA must compose.

Construct primitive and composite tasks:

retrieve
compare
retrieve then compare
copy
transform
retrieve then transform then copy
count
compare
count then compare
bind
retrieve
bind then retrieve
sort
select
sort then select kth item

Test whether the trajectory for A then B can be predicted from primitive trajectories for A and B.

Possible models:

vector addition;
operator multiplication;
ordered low-rank composition;
finite-state transition;
mixture-of-operators sequence;
graph composition.

Evaluate:

reconstruction of composite trajectories;
prediction of unseen compositions;
transfer across architectures;
causal swap of one stage while preserving the others.

The strongest result imaginable is:

Operators learned only from primitive tasks predict the internal trajectory and output behavior of unseen compositions.

That is much closer to an ISA than category RSA.

10. Find the dynamic scheduler

Your Phase 12 result suggests heads are not permanent opcodes; operations are dynamically assigned. So investigate the scheduler:

Which early residual features predict the later operation?
At what layer does task identity become decodable?
Which components gate the operation?
Is routing discrete or continuous?
Does operation selection occur once or repeatedly?
Can the model switch operation mid-prompt?
Are there common “control” subspaces?
Does instruction wording alter scheduling but preserve execution geometry?

Train a lightweight predictor:

early residual state → future operation mixture

Then causally modify the predicted scheduler state and test whether later computation changes.

11. Map implementation separately

Once residual instructions are established, map which components implement each instance:

direct logit attribution
attribution patching
path patching
edge interventions
head/MLP output replacement
sparse feature circuit tracing
attribution graphs
token-source decomposition
Q/K/V/OV path analysis

Current circuit-tracing work explicitly models feature-to-feature interactions through residual paths, which is relevant—but your proposed contribution would be operation-level invariants and cross-model implementation equivalence.

Do not require the same hardware components across models. Instead test:

same residual operation → different implementation graph

That would support your hardware/software analogy.

12. Build a coverage and saturation test

This is how you address “every single function.”

Start with K operation families. Repeatedly add new, adversarially selected tasks and measure:

nearest existing primitive distance;
reconstruction error using existing operators;
whether a new primitive is needed;
held-out behavioral prediction;
description length;
marginal increase in explained variance.

Plot:

number of task families → number of required primitives

Three outcomes:

Plateau: evidence for a compact ISA.
Sublinear growth: reusable basis but not closed.
Linear growth: likely no compact ISA at this granularity.

Pre-register a novelty threshold: a new operation is added only if it improves held-out reconstruction and causal prediction beyond a fixed margin.

Essential controls

You need all of these:

same operation, different vocabulary;
same vocabulary, different operation;
same answer, different computation;
same computation, different answer;
same prompt length;
same target-token frequency;
same output format;
same difficulty;
same number of reasoning steps;
shuffled labels;
shuffled tokens;
random residual directions;
layer-matched random subspaces;
norm-matched interventions;
position-matched controls;
paraphrase holdout;
domain holdout;
language holdout;
composition holdout;
architecture holdout;
scale holdout;
instruction-tuned versus base-model holdout.

Also distinguish whether RSA aligns:

task difficulty;
confidence;
answer entropy;
sequence length;
output-token class;
or genuinely the computation.

Regress those confounds out and rerun RSA.

Dataset scale

N = 224/112 is a good pilot, not a definitive universal-ISA dataset.

For a major paper, aim for:

at least 20–30 operation families;
multiple sub-operations per family;
hundreds of prompts per operation after templating;
multiple independent template generators;
three or more architecture families;
at least two scales within each family where feasible;
base and instruction variants;
strict discovery, validation and final untouched test sets.

The statistical unit should generally be the prompt or independently generated template family, not a correlated matrix entry.

The four papers hidden inside this

Trying to do everything in one paper could make it unfocused. The strongest first paper would be:

Paper 1: Cross-architecture residual computational geometry

Claims:

Task operations form reproducible residual trajectories.
Their relational geometry transfers across models.
Geometry generalizes to unseen templates, domains and operations.
It is not explained by lexical, positional, difficulty or output confounds.
Paper 2: Causal residual operators

Claims:

Operation-specific residual subspaces are necessary and sufficient.
They can be removed, inserted and swapped.
They compose to solve unseen composite tasks.
Paper 3: Dynamic hardware realization

Claims:

The same residual operation can be implemented by different heads/MLPs.
Physical circuits vary while operator-level geometry remains stable.
Runtime implementation can be predicted from the current residual state.
Paper 4: Residual ISA router

Claims:

Early residual state predicts the required operation mixture.
That operation predicts the minimal safe attention/compute policy.
Dynamic routing preserves PPL and retrieval while accelerating inference.
What would make this exceptional

The decisive demonstration would be:

Learn a compact set of residual operators from Qwen on primitive tasks. Align those operators to Llama using only a subset of operations. Predict the internal trajectories of unseen composite tasks in Llama. Then causally swap one operator—such as COMPARE for COPY—and cause the model to execute the corresponding counterfactual computation while preserving operands and unrelated behavior.

That simultaneously establishes:

abstraction;
cross-architecture invariance;
compositionality;
causal sufficiency;
functional specificity;
predictive understanding.

That is far more powerful than a taxonomy or an RSA correlation.

Your current ρ = 0.78 result is therefore not the end result. It is the first gate indicating that the Residual ISA hypothesis deserves this deeper program.