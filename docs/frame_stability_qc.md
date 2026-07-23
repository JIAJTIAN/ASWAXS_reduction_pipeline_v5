# SAXS Frame Stability QC Guide

## Purpose

The Frame Stability QC tab evaluates whether repeated SAXS detector frames are
stable before they are interpreted as one measurement. For new reductions, QC
runs from the frame-resolved 1-D curves already in memory immediately before
the group average. The recommendation is advisory: it does not modify raw HDF5
files, replace the stored group average, or regenerate ASAXS, stitching, or
XAnoS outputs.

The analysis is intended to help answer three questions:

1. Are all repeated frames statistically and physically consistent?
2. Is there a consecutive stable prefix before drift begins?
3. Should the measurement be accepted, reviewed with fewer frames, or rejected?

## Opening the Tool

1. Open `Tools > HDF5 I-q Plot Viewer`.
2. Choose a task-level `*_analysis.h5`, a detector `*_analysis.h5`, or the task
   output folder.
3. Select the `Frame Stability QC` tab.
4. Choose a detector, energy, and group series.
5. A saved report opens immediately when averaging-time QC is present. For an
   older analysis file, click `Reduce Frames and Run QC` to use the legacy
   read-only fallback.

Pil300K is normally the primary detector for SAXS stability, low-q behavior,
Rg, and I(0). Eig1M can also be examined, but Guinier quantities are generally
not meaningful over a WAXS-only q range.

## Required Analysis History

Current analysis files store a completed QC report under the detector reduction
process. This report includes the frame-resolved plot arrays, calculated
metrics, labels, recommendation, q ranges, and a `qc_complete` marker.

For older files without a saved report, the selected analysis HDF5 must retain
enough provenance to locate:

- the reduction manifest or complete raw-file list;
- energy, group, and frame numbering;
- PONI and mask files;
- the detector image dataset path;
- the monitor-normalization key;
- q unit and number of radial bins.

Task-level combined files are supported. The viewer follows recorded detector
analysis references and also checks sibling `Pil300K` and `Eig1M` output
directories. Raw files and calibration paths are required only for the legacy
fallback.

## What Happens When QC Runs

During a new reduction, each raw detector image is integrated once. Frame QC is
calculated from those normalized 1-D curves before `average_groups` releases
them, and the report is stored beside the averaged data in `analysis.h5`.

When the viewer finds the saved `qc_complete` marker, it loads and plots that
report directly. It does not reopen raw HDF5 files or repeat pyFAI integration.

For a legacy file without saved QC, `Reduce Frames and Run QC` processes the
selected energy/group series in a background thread:

1. Each recorded raw HDF5 frame is opened read-only.
2. Energy, detector image, and monitor value are read.
3. The image is integrated independently with the recorded PONI, mask, q unit,
   and radial-bin count.
4. Intensity and uncertainty are divided by the monitor value.
5. Frame metrics and plots are calculated.

The integrated frame series is cached in memory for the current viewer session.
Changing q ranges or reference mode then recalculates quickly. Selecting a new
energy/group requires its own frame integration. Closing the viewer clears the
cache.

## Controls

### Series

Selects one `(detector, energy, group)` measurement. Every frame in the selected
series is examined independently.

### Auto q Range

Uses the overlapping finite q range shared by all frames. Disable it to enter a
custom q minimum and maximum. At least eight common q points are required.

### Auto Low-q

Uses approximately the first 10% of the selected q span. Disable it to define a
custom low-q window. Use a range appropriate for the scientific question and
free from beamstop-edge or detector artifacts.

### Reference Mode

- `Compare with first frame`: reduced chi^2 and CorMap compare every frame with
  frame 1.
- `Compare with previous frame`: reduced chi^2 and CorMap compare each frame
  with the immediately preceding frame.

Invariant-like and low-q ratios always remain normalized to frame 1 so that
long-term drift is visible.

### Show Recommended Average

Overlays the mean of the currently recommended initial stable frames. This is a
display-only calculation. It is not written to HDF5 and does not alter reduction
products.

## Plot 1: Frame-Resolved I(q)

Each colored line is one independently integrated frame. Both axes are
logarithmic.

Look for:

- separation between frames;
- changing low-q intensity;
- changing slope or peak shape;
- isolated spikes or detector artifacts;
- gradual movement with frame number.

Strong overlap is encouraging, but logarithmic scaling can hide several-percent
drift. Always inspect the ratio and metric plots.

## Plot 2: Relative-Intensity Heatmap

The heatmap displays:

`R_i(q) = I_i(q) / I_1(q)`

- White: ratio near 1.
- Red: intensity greater than frame 1.
- Blue: intensity lower than frame 1.
- Horizontal bands: one frame differs broadly across q.
- Gradual color change with frame number: time-dependent drift.
- Vertical stripes: localized q-dependent differences or noise in the
  reference frame.

The displayed color scale is centered on 1 and normally spans 0.95 to 1.05.
Values outside that range are color-clipped but remain present numerically.

## Plot 3: Invariant-Like Stability

For every frame:

`Q_i = integral[q^2 I_i(q) dq]`

The plot shows `Q_i / Q_1` versus frame number. The green region represents the
default +/-5% tolerance.

This metric summarizes overall scattering intensity over the selected q range.
It is called invariant-like because a finite detector q range is used. It is not
a complete physical Porod invariant unless the required full q range and
absolute corrections are available.

## Plot 4: Low-q Stability

The mean intensity in the selected low-q window is normalized to frame 1:

`L_i / L_1`

Low-q changes can indicate aggregation, fragmentation, sedimentation, sample
movement, concentration change, beam damage, or normalization problems. The
green region represents the default +/-5% tolerance.

An increase is often associated with growing large structures or aggregation.
A decrease can indicate depletion, fragmentation, sedimentation, movement out
of the beam, or an intensity-normalization change. The direction alone does not
identify the mechanism.

## Plot 5: Statistical Similarity

### Reduced chi^2

For frame `i` and reference frame `r`:

`chi_red^2 = (1 / (M - 1)) sum_q [(I_i - I_r)^2 / (sigma_i^2 + sigma_r^2)]`

Here `M` is the number of valid q points.

- Near 1: differences are broadly consistent with propagated uncertainty.
- Below 1: frames agree unusually well or uncertainties are overestimated.
- Between 1 and 3: measurable variation that may require review.
- Above 3: differences are substantially larger than expected uncertainty.

Frame 1 has chi^2 = 0 when compared with itself. A value of 10 does not mean a
10% intensity change; it means the error-weighted squared discrepancy is about
ten times the expected variance on average.

### CorMap p-value

CorMap examines the signs of consecutive differences across q and evaluates the
longest run with the same sign. It detects systematic residuals without using
sigma.

- `p >= 0.01`: no CorMap failure at the default significance level.
- `p < 0.01`: differences are statistically systematic.

CorMap can detect very small but coherent scale or shape changes. It is therefore
interpreted together with drift magnitude and chi^2 rather than used alone.

## Where sigma Comes From

For each raw frame, pixel counting variance is approximated as:

`variance_pixel = abs(pixel counts)`

This variance is passed to `pyFAI.integrate1d()` with the Poisson error model.
pyFAI propagates pixel variances through azimuthal integration to produce
`sigma_I(q)`. After monitor normalization:

`I_norm(q) = I(q) / monitor`

`sigma_norm(q) = sigma_I(q) / abs(monitor)`

This sigma is the propagated uncertainty of one raw frame. It is not the
standard deviation between repeated frames.

The current sigma does not include uncertainty from the monitor, PONI geometry,
mask, detector calibration, background subtraction, or other systematic
effects. Neighboring q bins can also be correlated by azimuthal integration.
These limitations are why chi^2 is not used by itself.

## Plot 6: Optional Structural Trends

### Rg and I(0)

The viewer fits `ln[I(q)]` versus `q^2` in the selected low-q window and reports
the corresponding Guinier Rg and I(0). These values are diagnostic only.

The current fit does not automatically enforce `q_max Rg < 1.3`. Rg and I(0)
must not be interpreted physically unless the selected range is a valid Guinier
region with positive intensity and adequate signal.

### Peak Position q* and FWHM

q* is the maximum intensity position within the selected q range. FWHM is
estimated relative to the local minimum baseline. A q* located at a range
boundary usually means no resolved internal peak exists and should not be
treated as a structural peak.

Rg, I(0), q*, and FWHM are displayed in the plot or table but do not currently
control the QC label.

## Default QC Classification

Frame 1 is labeled Good by definition. For later frames:

### Good

- invariant-like drift <=2%;
- low-q drift <=2%;
- reduced chi^2 <=1.5, when finite;
- no condition requiring a Bad label.

### Acceptable

The frame remains within the Bad limits but does not satisfy every Good limit.
This includes small statistically detectable differences whose overall
intensity drift remains limited.

### Bad

A frame is Bad when either condition is met:

1. invariant-like drift >5% or low-q drift >5%; or
2. CorMap `p < 0.01`, reduced chi^2 >3, and either invariant-like or low-q
   drift >2%.

The recommendation uses the consecutive initial frames before the first Bad
frame. Separately, the interface reports the start of three consecutive Bad
frames as a stronger instability-onset indicator.

## Interpreting Common Patterns

### Stable Measurement

- overlays coincide;
- ratio heatmap remains near white;
- invariant-like and low-q ratios remain near 1;
- chi^2 remains near 1;
- CorMap failures are absent or isolated;
- structural trends remain flat within uncertainty.

### Uniform Intensity Loss

- ratio heatmap becomes broadly blue;
- invariant-like and low-q ratios decrease together;
- q* may remain constant.

Possible causes include sample depletion, beam-induced movement, sedimentation,
transmission/monitor changes, or radiation-driven loss of scattering material.

### Aggregation-Like Change

- low-q intensity increases;
- Rg may increase;
- invariant-like intensity and chi^2 may rise;
- CorMap shows systematic failure.

### Isolated Bad Frame

- one horizontal heatmap band;
- one metric spike;
- neighboring frames recover.

This may be a detector, beam, or acquisition transient rather than monotonic
sample damage.

## Using QC in Reduction Decisions

The QC recommendation is deliberately advisory. A conservative workflow is:

1. perform the normal reduction, which now calculates QC before each average;
2. inspect the stored sample, solvent, and relevant background reports;
3. record the recommended stable prefix and any manual scientific decision;
4. compare synchronized detector behavior;
5. only then decide whether a future QC-selected re-average is warranted.

For simultaneous Pil300K and Eig1M measurements, a common temporal cutoff may
be appropriate when both detectors observe the same exposure sequence. The most
sensitive detector often determines the conservative cutoff.

## Performance

Analysis-HDF5 curve discovery runs in a worker thread. HDF5 rows are sliced
directly, and only one curve is plotted initially.

For current files, opening Frame QC reads the saved report and does not repeat
reduction. QC calculation adds metric processing during averaging, but no extra
raw HDF5 reads or pyFAI integrations.

Legacy files still require approximately `N` raw HDF5 reads and `N` pyFAI
integrations for `N` selected frames. This fallback runs in a background thread.

The plot itself is usually not the bottleneck. Network HDF5 access, image
decompression, and pyFAI integration dominate the first run.

## Troubleshooting

### No Series Found

- Open a detector-specific `*_analysis.h5` directly.
- Confirm the reduction manifest still exists.
- Confirm `/entry/data_reference/data_file` contains the complete raw history.
- Confirm the combined task output still has sibling `Pil300K` or `Eig1M`
  detector directories.

### Only One Frame Appears

The recorded history contains one raw image for that energy/group. A stability
trend requires multiple independently measured frames.

### Missing Raw, PONI, or Mask File in a Legacy Analysis

Older analyses store paths, not copies of raw detector images or calibration
files. Run legacy QC where those paths resolve. New analyses with stored QC do
not require these files merely to display the saved report.

### Viewer Appears Slow

Check the status line. `Stored averaging-time QC loaded` means no reduction is
running. A legacy report may still be slow because HDF5 discovery and frame
integration run in a worker thread. Reopening a legacy file clears its in-memory
cache and requires reintegration.

## References

The QC combines complementary methods rather than reproducing one paper's
workflow exactly:

- Frame-dependent quality metrics, Guinier quantities, and invariant-related
  measurements follow the objective SAXS-quality approach of Grant et al.
- CorMap-style sign-run testing follows Franke, Jeffries, and Svergun.
- The use of several metrics together, including integrated intensity, `Rg`,
  and `I(0)`, follows the recommendation of Hopkins and Thorne that no single
  metric detects every form of radiation damage.
- The default CorMap significance level of `p = 0.01` and the use of three
  consecutive failures to identify a persistent onset follow Brooks-Bartlett
  et al.

The `q*` and FWHM trends are FrameByFrame-ASWAXS extensions for materials
scattering. The 2-5% drift bands and reduced chi-square limits are configurable
software defaults, not universal limits prescribed by the cited papers. The
displayed invariant-like quantity is evaluated only over the selected measured
q range and is not a full Porod invariant from zero to infinite q.

1. Grant, T. D., Luft, J. R., Carter, L. G., Matsui, T., Weiss, T. M., Martel,
   A. & Snell, E. H. "The accurate assessment of small-angle X-ray scattering
   data." *Acta Crystallographica Section D* **71**, 45-56 (2015).
   [doi:10.1107/S1399004714010876](https://doi.org/10.1107/S1399004714010876)

2. Franke, D., Jeffries, C. M. & Svergun, D. I. "Correlation Map, a
   goodness-of-fit test for one-dimensional X-ray scattering spectra."
   *Nature Methods* **12**, 419-422 (2015).
   [doi:10.1038/nmeth.3358](https://doi.org/10.1038/nmeth.3358)

3. Hopkins, J. B. & Thorne, R. E. "Quantifying radiation damage in
   biomolecular small-angle X-ray scattering." *Journal of Applied
   Crystallography* **49**, 880-890 (2016).
   [doi:10.1107/S1600576716005136](https://doi.org/10.1107/S1600576716005136)

4. Brooks-Bartlett, J. C., Batters, R. A., Bury, C. S., Lowe, E. D., Ginn,
   H. M., Round, A. & Garman, E. F. "Development of tools to automate
   quantitative analysis of radiation damage in SAXS experiments."
   *Journal of Synchrotron Radiation* **24**, 63-72 (2017).
   [doi:10.1107/S1600577516015083](https://doi.org/10.1107/S1600577516015083)
