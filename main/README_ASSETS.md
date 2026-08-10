# AURORA input data

This directory contains FITS catalogues and JSON configuration/index data.
Generated maps and caches do not belong here: final maps are stored in
`maps/`, regional products in `maps/regions/`, and caches in `maps/cache/`.

Expected shared inputs include:

- `aurora_gaia_catalog_900m.fits`
- `aurora_microlensing.fits`
- `rr_lyrae.fits`, `cepheids.fits`, `zz_ceti.fits`, `lbv.fits`,
  `cataclysmic_variables.fits`
- `index.json` (constellation names, stick-figure lines, HIP coordinates, and
  IAU boundary edges)

Large catalogues remain intentionally untracked by Git.

Constellation overlays are read from `index.json`. The renderer uses
`constellations[].lines` plus the embedded `hip_stars` coordinates to draw
stick figures directly on Hammer maps.
