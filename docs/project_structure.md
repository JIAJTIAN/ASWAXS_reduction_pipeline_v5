# Project Structure

This project is intentionally not packaged for installation yet. The source is
organized in a standard `src/` layout, and the `scripts/` folder contains small
launchers that add `src/` to `sys.path`.

```text
ASWAXS_reduction_pipeline_v5/
  README.md
  .gitignore
  docs/
    project_structure.md
  scripts/
    run_gui.py
    run_pyfai_gui.py
    run_reducer.py
    run_viewer.py
    run_kafka_queue_bridge.py
    write_frame_done.py
    write_measurement_done.py
  src/
    aswaxs_live/
      __init__.py
      core/
        __init__.py
        analysis_h5.py
        reduce_aswaxs_sequence.py
        reduce_sequence.py
        reduction_pipeline.py
      preprocessing/
        gui.py
        io_utils.py
        processing.py
      gui.py
      bluesky_queue.py
      kafka_bridge.py
      reducer.py
      stitcher.py
      viewer.py
  outputs/              ignored, local generated data
```

Main modules:

- `aswaxs_live.reducer`: manifest replay, strict folder watcher, Bluesky/Kafka
  measurement_done queue mode, HDF5 analysis writing, resume/restart behavior.
- `aswaxs_live.bluesky_queue`: JSONL measurement_done reduction job queue helpers.
- `aswaxs_live.kafka_bridge`: optional Bluesky/Kafka bridge that normalizes
  beamline messages into local measurement_done reduction jobs.
- `aswaxs_live.core`: copied reduction science code from the previous pipeline,
  kept inside this repository so v3 can run by itself.
- `aswaxs_live.preprocessing`: HDF5-to-pyFAI calibration GUI, EDF bridge export,
  PONI loading, and mask authoring helpers.
- `aswaxs_live.viewer`: live curve viewer for single frames, group averages, and
  final ASAXS curves.
- `aswaxs_live.stitcher`: combines detector-named analysis records into one
  batch HDF5 and writes stitched averages.
- `aswaxs_live.gui`: three-window GUI launcher around the reducer and viewer.

For dual-detector runs, the user-facing batch file is
`outputs/<run>/<sample_name>_analysis.h5` with `/entry/Pil300K`,
`/entry/Eig1M`, and `/entry/stitched_averages` in the same HDF5 file.
