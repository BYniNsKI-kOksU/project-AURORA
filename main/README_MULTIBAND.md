# AURORA — uniwersalny renderer map wielopasmowych

`aurora_multiband_render.py` tworzy osobne mapy całego nieba w projekcji
Hammera dla dowolnego rodzaju sygnału: radio, mikrofale, podczerwień, światło
widzialne, UV, promieniowanie X, gamma albo własne dane liczbowe.

Program nie przypisuje fizyki do rozszerzenia pliku. Każda mapa jest warstwą
skalarną, której sposób odczytu, układ współrzędnych, paleta i rozciągnięcie
jasności są opisane w `assets/aurora_multiband_maps.json`.

## Szybki start

1. Umieść pliki wejściowe w `assets/multiband/` albo zmień ich ścieżki w
   konfiguracji.
2. Uruchom:

```bash
python main/aurora_multiband_render.py
```

Program wyświetli numerowaną listę. Można wpisać na przykład:

```text
1 3 5
```

albo zakres `2-4`, `a` dla wszystkich lub `q`, aby zakończyć. Każda wybrana
mapa trafia do osobnego pliku `maps/multiband/*_hammer.png`. Przy co najmniej
dwóch poprawnie utworzonych mapach program pyta o dodatkowy kolaż. Kolaż
układa obrazy jeden obok drugiego i nie miesza ich danych.

Przed renderowaniem program pyta też, czy nałożyć linie gwiazdozbiorów z
`assets/index.json`. Odpowiedź `t` dodaje je do każdej wybranej mapy.

Program pyta również o rozdzielczość mapy: `8`, `16`, `32` albo `64` K.
Enter wybiera wartość z konfiguracji, domyślnie `16K`. Ręczne `--width` i
`--height` nadal nadpisują ten wybór.

Tryb bez pytań, przydatny w skryptach:

```bash
python main/aurora_multiband_render.py \
  --select 1,3-5 \
  --collage yes \
  --constellations yes
```

Lista bez renderowania:

```bash
python main/aurora_multiband_render.py --list
```

Testowa mniejsza rozdzielczość:

```bash
python main/aurora_multiband_render.py \
  --select radio_408_mhz \
  --width 1024 \
  --height 512 \
  --collage no
```

## Obsługiwane dane

### Obraz FITS z WCS

`format: "auto"` rozpoznaje HDU obrazu i jego niebiański WCS. Obsługiwane są
układy galaktyczny, ICRS/FK5 oraz ekliptyczny. Dla kostki danych można wskazać
warstwę:

```json
{
  "id": "xray",
  "name": "X-ray",
  "path": "multiband/xray_cube.fits",
  "format": "fits_image",
  "hdu": 0,
  "plane": 2,
  "palette": "viridis"
}
```

### HEALPix FITS

Obsługiwane są pełne i częściowe tabele HEALPix, porządki `RING` i `NESTED`
oraz nagłówki `NSIDE`, `ORDERING`, `COORDSYS` i `INDXSCHM`.

```json
{
  "id": "planck",
  "name": "Planck",
  "path": "multiband/planck.fits",
  "format": "healpix",
  "hdu": 1,
  "field": "I_STOKES",
  "plane": 0,
  "palette": "inferno"
}
```

Gdy nagłówki są kompletne, wystarczą `path` i `format: "auto"`.

### Katalog punktowy FITS, CSV lub TSV

Renderer automatycznie szuka pary `l`/`b` albo `ra`/`dec`. Nazwy można podać
jawnie. `value_mode` przyjmuje `intensity`, `magnitude`, `log10` albo `count`.

```json
{
  "id": "gamma_catalog",
  "name": "Źródła gamma",
  "path": "multiband/gamma_sources.csv",
  "format": "catalog",
  "coordinates": "icrs",
  "longitude_column": "RAJ2000",
  "latitude_column": "DEJ2000",
  "value_column": "Flux",
  "value_mode": "intensity",
  "catalog_smoothing_sigma": 1.5,
  "ignore_zeros": true,
  "palette": "plasma"
}
```

Dla mapy gęstości bez kolumny sygnału należy ustawić
`"value_mode": "count"`.

### Raster equirektangularny

Obsługiwane są `.npy`, `.npz`, PNG, JPEG, TIFF, BMP i WebP. Domyślnie lewa
krawędź oznacza -180°, prawa +180°, góra +90°, dół -90°. Inny zapis można
opisać tak:

```json
{
  "id": "radio_array",
  "name": "Radio",
  "path": "multiband/radio.npy",
  "format": "equirectangular",
  "coordinates": "galactic",
  "longitude_range": [0.0, 360.0],
  "latitude_order": "south_to_north",
  "palette": "magma"
}
```

Dla pliku NPZ `field` wskazuje nazwę tablicy. Dla kolorowego rastra `channel`
może mieć wartość `red`, `green`, `blue`, numer kanału albo `luminance`.

## Sterowanie wyglądem

Najczęściej używane pola pojedynczej mapy:

- `percentiles: [low, high]` — odcięcie wartości odstających;
- `stretch` — `linear`, `asinh`, `log`, `sqrt` albo `power`;
- `stretch_strength` — siła `asinh`/`log`;
- `gamma` — wykładnik dla `power`;
- `palette` — między innymi `inferno`, `magma`, `plasma`, `viridis`, `hot`,
  `turbo`, `gray`; końcówka `_r` odwraca paletę;
- `color: "#RRGGBB"` — jednobarwny gradient zamiast palety;
- `scale_factor` i `offset` — kalibracja wartości przed tone mappingiem;
- `ignore_zeros` — pomijanie zer przy obliczaniu percentyli.

Ustawienia w sekcji `output` są domyślne dla wszystkich map. Ustawienie przy
konkretnej mapie ma pierwszeństwo.

## Wyniki i diagnostyka

Oprócz osobnych PNG oraz opcjonalnego
`aurora_multiband_collage.png` program zapisuje
`aurora_multiband_render.json`. Plik ten zawiera wejścia, typy danych,
rozdzielczość i rzeczywiste granice tone mappingu dla każdej poprawnie
utworzonej mapy.

Jeżeli jedna z wybranych pozycji jest błędna, pozostałe nadal są renderowane.
Program raportuje nieudaną pozycję i kończy się kodem `1`. Opcja `--debug`
dodaje pełny traceback.
