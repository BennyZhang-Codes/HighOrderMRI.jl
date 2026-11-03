# Field preprocessing and synchronization

Field coefficients must be aligned with ADC samples and use the encoding
operator's phase, coordinate and unit conventions. HighOrderMRI provides GIRF
prediction, time interpolation and model-based delay estimation. Field monitoring
provides measured coefficients; it does not remove the need to check these
conventions. [[6]](/references#ref-6)

## GIRF prediction

A gradient impulse response function models the gradient chain as a linear,
time-invariant system. A calibrated frequency response predicts field waveforms
from prescribed physical gradients; the approximation and calibration conditions
come from the measurement, not the container type. [[7]](/references#ref-7)

[`GIRFModel`](/reference/girf-model) stores a centred transfer matrix and
frequency metadata. [`apply_girf`](/reference/apply-girf) applies that matrix
to three DCS gradient channels. Extra dimensions, such as shots, are evaluated
independently. Its real output has the calibrated number of field channels.

The [identity-response example](/reference/girf-model#matched-grid-identity-example)
is a complete CPU configuration. A real acquisition requires a response matched
to the waveform dwell time and calibrated channel units. The model's frequency
vector is not used for automatic resampling. The current FFT operation has
periodic boundary behavior, and response expansion can omit a valid impulse
coefficient; see [current limits](/reference/apply-girf#current-limits).

To convert instantaneous frequency-field coefficients into accumulated phase,
use trapezoidal integration with [`grad2traj`](/reference/utilities). This helper
does not supply a gyromagnetic ratio or a $2\pi$ conversion. For example, after
converting a physical first-order gradient to Hz/m, integration in seconds gives
cycles/m. A zeroth-order field in Hz integrates to cycles. Conversion factors
must match each output channel's physical meaning.

## Interpolation to ADC times

[`InterpTrajTime`](/reference/interp-traj-time) accepts coefficient samples in
rows and field terms in columns. For shot $i$, field sample $j$ is placed at

$$
t_{j,i}=(j-1)\,dt-(\mathrm{StartTime}_i+\tau).
$$

Evaluating at the ADC time therefore samples the original field as
$k_i(t_\mathrm{ADC}+\mathrm{StartTime}_i+\tau)$. This fixes the sign convention:
a positive offset queries a later point in the original waveform. Do not
describe it as moving the input support to a later time.

Use the [linear-signal example](/reference/interp-traj-time#check-the-sign-on-a-linear-signal)
to check this sign. Transpose its two-dimensional result before passing it to
an explicit encoding constructor, which expects terms in rows. Multishot
interpolation uses one time row and one offset per shot. Zero extrapolation
outside the measured interval can distort the model; inspect the complete
shifted ADC interval before interpreting reconstructed results.

## Model-based delay estimation

[`FindDelay`](/reference/find-delay) and
[`FindDelay_multishot`](/reference/find-delay-multishot) follow the model-based
synchronization approach of Dubovan and Baron. [[8]](/references#ref-8)
The package uses density weights in each image reconstruction, then an
unweighted scalar delay update with JumpFact acceleration.

At a current delay $\tau$, reconstruct $x_\tau$ using $A_\tau$, and define

$$
r_\tau=y-A_\tau x_\tau,\qquad b_\tau=B_\tau x_\tau.
$$

For nonzero derivative information, the linearization
$r_\tau\approx\Delta\tau\,b_\tau$ gives the implemented update

$$
\Delta\tau=J\,\operatorname{Re}\!\left(
\frac{b_\tau^H r_\tau}{b_\tau^H b_\tau}\right).
$$

Here $J$ is the current JumpFact. Derivative-forward coefficients are formed
with the source's five-tap temporal filter and interpolated using the same
delay. For the positive phase convention,

$$
\frac{\partial}{\partial\tau}e^{\mathrm{i}2\pi\Phi_\tau}
=e^{\mathrm{i}2\pi\Phi_\tau}\,
\mathrm{i}2\pi\frac{\partial\Phi_\tau}{\partial\tau}.
$$

The implementation differentiates shifted field coefficients; it does not shift
the ADC times multiplying the static field map. After updating $\tau$, J is
reduced when the update changes sign. The stopping comparison uses the update
magnitude, and the scalar returned is
$\tau-\Delta\tau(J-1)/J$ with the final J. These implementation choices are
not an assertion of exact equivalence to every detail of the publication.

Use the CPU configuration on the [single-shot API page](/reference/find-delay#prepare-a-cpu-call):
`arrayType=Array`, a solver type such as `CGNR`, and a regularization object.
There are no `use_gpu` or `λ` keywords. Use seconds when the field map is in Hz,
and provide `Δτ_min` explicitly with the input floating-point type. The actual
default `0.005` means 5 ms in that convention; it is not 0.005 microseconds.
The log prints `τ/dt` and `Δτ/dt`, which are sample offsets despite their labels.

::: warning Current limits of the delay estimators
There is no outer-iteration cap or convergence-status result. `iter_max` only
limits an image reconstruction; the initial delay is fixed at zero. Check
identifiability, shifted interval coverage and an independent alignment
reference before interpreting the result.

For multishot calls, provide explicit ADC times. The convenience overload has
an incorrect time shape, and the per-shot residual operators currently reuse
the first shot's time vector. Different shot times with nonzero B₀ are not a
supported reliable case. Float64 is the conservative current multishot choice;
see its API page for the Float32 default and state-type limitations.
:::

The multishot method estimates one shared scalar, preserving the supplied shot
start offsets. Coil compression can reduce repeated encoding work; apply the
same fitted transform to both data and CSM as described in
[coil compression](/guide/coil-compression).

Other trajectory-correction methods, including spiral delay calibration,
RING and trajectory auto-correction, address related errors with different
models and are not interchangeable implementations of this estimator.
[[9]](/references#ref-9) [[10]](/references#ref-10) [[11]](/references#ref-11)

## Phase preparation checklist

Before constructing an encoding operator, check that:

1. coefficients have been integrated exactly once;
2. phase in radians has been converted to cycles;
3. spatial coefficient units match the grid coordinate unit;
4. coefficient ordering matches the [solid-harmonic basis](/theory/encoding-model#solid-harmonic-basis-order);
5. zeroth-order phase has not been corrected twice in data and coefficients;
6. density compensation and encoding use consistent trajectory and timing conventions;
7. data and coil maps use the same coil ordering and any whitening/compression transform.

Keep timing offsets, tolerances, units and model version with the acquisition's
configuration. Use independent field/timing information where available;
synthetic checks alone do not validate real-data synchronization or GIRF accuracy.
