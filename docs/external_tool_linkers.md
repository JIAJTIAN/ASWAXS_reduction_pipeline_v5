# External Tool Linkers

FrameByFrame-ASWAXS links to XAnoS, XModFit, and sample-position planning tools
without requiring permanent edits to those external projects.

## Principle

The reduction platform owns the workflow; external tools keep their scientific
code and user interfaces. Integration code lives in FrameByFrame-ASWAXS as a
thin linker/adapter layer.

```text
FrameByFrame-ASWAXS
  prepares data, launches tools, applies helper defaults

external tool linker
  resolves paths, starts original scripts, optionally preloads data

original XAnoS / XModFit / Sample Position App
  remains source-compatible with upstream
```

## Current Linkers

- `aswaxs_live.tools.linkers.xanos`
  - finds `XAnoS_Components.py`;
  - opens the original XAnoS Components Qt widget;
  - optionally preloads FrameByFrame-ASWAXS XAnoS-format `.dat` files;
  - applies only runtime visual defaults and selection helpers.

- `aswaxs_live.tools.linkers.xmodfit`
  - finds `xmodfit.py`;
  - launches original XModFit with the current Python environment;
  - reserves a future `data_files` argument for an explicit import contract.

- `aswaxs_live.tools.linkers.sample_position`
  - launches the separate sample-position planning app;
  - keeps experiment-layout planning reusable by both reduction and measurement
    software.

## Environment Variables

Set these only when the external tools are not in the expected sibling folders:

```text
FRAMEBYFRAME_XANOS_COMPONENTS=/path/to/XAnoS_Components.py
FRAMEBYFRAME_XMODFIT_SCRIPT=/path/to/xmodfit.py
FRAMEBYFRAME_SAMPLE_POSITION_APP=/path/to/sample-position/main.py
```

The older `ASWAXS_XANOS_COMPONENTS` variable is still accepted for compatibility.

## Future Contract

The preferred long-term contract is file/folder based:

1. FrameByFrame-ASWAXS writes XAnoS-compatible ASAXS `.dat` files.
2. XAnoS Components reads those files and writes component outputs in a known
   folder.
3. FrameByFrame-ASWAXS detects or receives that component-output folder.
4. XModFit opens the SAXS, cross, and resonant component files.

Only this linker layer should automate GUI defaults such as log-log plots,
preselected data, theme styling, or helper buttons. The original scientific
calculation code should remain upstream-compatible whenever possible.
