# Dane wejściowe renderera wielopasmowego

Ten katalog jest domyślnym miejscem na duże mapy radio, mikrofalowe,
podczerwone, optyczne, UV, X-ray i gamma używane przez
`main/aurora_multiband_render.py`.

Pliki nie są częścią repozytorium. Ich nazwy i sposób odczytu definiuje
`assets/aurora_multiband_maps.json`; nie trzeba używać nazw z przykładowej
konfiguracji. Obsługiwane są FITS/WCS, FITS HEALPix, katalogi FITS/CSV/TSV/ECSV,
tablice NPY/NPZ i equirektangularne obrazy rastrowe.
