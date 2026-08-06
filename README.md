# AURORA — Astronomical Unified Rendering Of Relativistic Astrophysics

[Polski](#polski) · [English](#english)

AURORA to zestaw pipeline'ów Pythona, które pobierają wybrane dane Gaia DR3 i
przekształcają je w wysokorozdzielcze mapy oraz animacje astronomiczne.

---

<a id="polski"></a>

## Polski

AURORA pobiera i przygotowuje dane **Gaia DR3**, a następnie tworzy:

- 8-bitową mapę całego nieba w projekcji Hammera;
- prostokątną, pozbawioną projekcji mapę wybranego obszaru nieba;
- animację zdarzeń mikrosoczewkowania w modelu Paczyńskiego;
- interaktywny atlas wszystkich zdarzeń mikrosoczewkowania;
- stylizowaną animację pulsowania gwiazd zmiennych.

### Moduły

| Skrypt | Rola |
|---|---|
| `main/aurora_gaia_catalog.py` | Pobiera wznawialny katalog do 1 811 709 771 źródeł Gaia DR3 i łączy fragmenty w jeden plik FITS. |
| `main/aurora_sky_render.py` | Buduje z katalogu Gaia mapę całego nieba 16K w projekcji Hammera. |
| `main/aurora_sky_region_render.py` | Renderuje konfigurowalny prostokątny wycinek nieba bez projekcji kartograficznej. |
| `microlensing/aurora_microlensing_catalog.py` | Pobiera zdarzenia z `gaiadr3.vari_microlensing` wraz z pozycją i fotometrią źródła. |
| `microlensing/aurora_microlensing_render.py` | Nakłada zdarzenia mikrosoczewkowania na mapę Hammera i koduje wideo HEVC. |
| `microlensing/aurora_microlensing_map.py` | Uruchamia lokalny, interaktywny atlas wszystkich zdarzeń z płynnym zoomem i pełnymi kartami danych. |
| `variables/aurora_variable_catalog.py` | Pobiera i normalizuje katalogi gwiazd zmiennych Gaia DR3. |
| `variables/classify_hot_variables.py` | Dzieli szeroki katalog `BE|GCAS|SDOR|WR` na ostrożnie ocenione katalogi kandydatów i wylicza skorygowaną barwę. |
| `variables/aurora_variable_animation.py` | Nakłada podstawowe i pozostałe klasy gwiazd zmiennych na mapę Hammera i koduje animację HEVC. |
| `core/aurora_console.py` | Ujednolica nagłówki, statusy, błędy i paski postępu wszystkich programów w terminalu. |
| `core/aurora_render_core.py` | Udostępnia wspólne kolory gwiazd, projekcję Hammera, PSF, krzywe blasku, rasteryzację i sygnatury cache. |
| `core/aurora_paths.py` | Definiuje kanoniczne katalogi danych `assets/`, map `maps/`, cache `maps/cache/` i filmów `videos/`. |

### Przepływ danych

```text
MAPA CAŁEGO NIEBA
Gaia DR3
  └─ main/aurora_gaia_catalog.py
       └─ assets/aurora_gaia_catalog_900m.fits
            ├─ main/aurora_sky_render.py
            │    └─ maps/aurora_sky_map_hammer_16k.png
            └─ main/aurora_sky_region_render.py
                 └─ maps/regions/aurora_sky_region_rect_pic1_16k.png

MIKROSOCZEWKOWANIE
Gaia DR3
  └─ microlensing/aurora_microlensing_catalog.py
       └─ assets/aurora_microlensing.fits
            ┐
mapa 16K ───┴─ microlensing/aurora_microlensing_render.py
                 ├─ aurora_microlensing_animation.mp4
                 └─ aurora_microlensing_16k_hvc1.mp4
           └──── microlensing/aurora_microlensing_map.py
                  └─ interaktywny atlas lokalny

GWIAZDY ZMIENNE
Gaia DR3
  └─ variables/aurora_variable_catalog.py
       └─ assets/*.fits + mapa 16K
            └─ variables/aurora_variable_animation.py
                 ├─ aurora_variable_animation.mp4
                 └─ aurora_variable_16k_hvc1.mp4
```

### Wspólne zasoby

Katalog `assets/` zawiera wejściowe katalogi FITS oraz konfiguracje JSON.
Wynikowe mapy trafiają do `maps/`, mapy regionalne i ich layouty do
`maps/regions/`, cache do `maps/cache/`, a filmy MP4 do `videos/`.

```text
assets/aurora_gaia_catalog_900m.fits
assets/aurora_microlensing.fits
assets/rr_lyrae.fits
assets/cepheids.fits
assets/zz_ceti.fits
assets/lbv.fits
assets/be.fits, gcas.fits, sdor.fits, wr.fits
assets/be_gcas_sdor_wr_unknown.fits
assets/cataclysmic_variables.fits
assets/other_variables.fits          # opcjonalny
```

Programy rozwiązują te ścieżki względem katalogu projektu, niezależnie od
bieżącego katalogu roboczego.

## Instalacja

Wymagany jest **Python 3.12+**. Sam kod korzysta ze składni dostępnej w
starszych wersjach, ale przypięte w `requirements.txt` wersje NumPy i SciPy
wymagają co najmniej Pythona 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

W Windows aktywacja środowiska wygląda następująco:

```powershell
.venv\Scripts\activate
```

### Wyjście terminalowe

Wszystkie programy korzystają z `core/aurora_console.py`, dlatego nagłówki,
separatory, komunikaty diagnostyczne i paski postępu mają ten sam format:

```text
[AURORA] Nazwa etapu
────────────────────────────────────────────────────────────
  → wykonywana operacja lub informacja
  ✓ operacja zakończona powodzeniem
  ! ostrzeżenie, po którym program może kontynuować
  ✗ błąd zapisywany do stderr
  ? pytanie wymagające odpowiedzi użytkownika
  ↻ Etap:  50%|############            | 5/10 [00:01<00:01]
```

Paski postępu są domyślnie animowane tylko w interaktywnym terminalu, aby nie
zaśmiecać logów przekierowanych do pliku. Zmienna `AURORA_PROGRESS` przyjmuje
wartości `auto` (domyślna), `always` lub `never`. Formatowanie konsoli używa
wyłącznie biblioteki standardowej; `tqdm` z `requirements.txt` odpowiada za
paski postępu.

### FFmpeg

`microlensing/aurora_microlensing_render.py` i
`variables/aurora_variable_animation.py` wymagają programu **FFmpeg**
dostępnego w `PATH` oraz kodera `libx265`.

Przykładowa instalacja:

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update
sudo apt install ffmpeg
```

Sprawdzenie koderów:

```bash
ffmpeg -hide_banner -encoders | grep libx265
```

## Szybki start

Poniższe polecenia zakładają uruchomienie z katalogu projektu. Wejścia i
wyniki są rozwiązywane względem katalogu projektu: dane w `assets/`, mapy i
cache w `maps/`, a filmy w `videos/`.

### 1. Mapa całego nieba

```bash
python main/aurora_gaia_catalog.py
python main/aurora_sky_render.py
```

`main/aurora_gaia_catalog.py`:

- dzieli pobieranie na 61 deterministycznych zakresów `random_index`;
- pobiera do 30 000 000 rekordów na zakres, do skonfigurowanego limitu
  1 811 709 771 źródeł;
- zapisuje fragmenty jako `assets/gaia_chunks/gaia_chunk_NN.fits`;
- pomija już istniejące fragmenty, więc pobieranie można wznowić;
- wykonuje do 5 prób zapytania z limitem 1 godziny na próbę;
- składa FITS przez bezpośrednie kopiowanie binarnych danych tabel, bez
  materializowania wszystkich rekordów w RAM.

Pobranie dużego katalogu wymaga uwierzytelnionej sesji Gaia Archive. Dane można
podać interaktywnie albo przekazać przez środowisko:

```bash
export GAIA_USER="login"
export GAIA_PASSWORD="haslo"
python main/aurora_gaia_catalog.py
```

Downloader zapisuje finalny wynik bezpośrednio jako
`assets/aurora_gaia_catalog_900m.fits`. Jeżeli wynik już istnieje, pobieranie
kończy się bez nadpisywania. Historyczne `900m` w nazwie pozostaje dla
zgodności ścieżek; bieżący limit katalogu jest większy.

Przy starcie `main/aurora_sky_render.py` pyta o rozdzielczość `8`, `16`, `32`
lub `64` (Enter wybiera 16K). Profil ustala wymiary, `BINS_L/B` oraz nazwy
mapy, layoutu i cache, więc warianty nie kolidują ze sobą. W trybie bez
terminala można użyć `AURORA_RESOLUTION_K=8` (lub 16/32/64).

Renderer tworzy odpowiedni wariant, np. dla 16K:

- `maps/aurora_sky_map_hammer_16k.png` — 8-bitowy PNG RGB,
  16 384 × 8 192 px;
- opcjonalne warianty z niezależnie wybieranymi liniami konstelacji, subtelną
  siatką współrzędnych galaktycznych i dwiema granicami widoczności z Polski.
  Renderer zadaje trzy osobne pytania, a każda kombinacja otrzymuje osobny
  sufiks, np. `_constellations.png`, `_coordinates.png`, `_poland_limits.png`
  albo `_constellations_coordinates_poland_limits.png`. Siatka jest
  neutralnie szara, konstelacje jasnoniebieskie, a granice `δ=-38°` i
  `δ=-28°` odpowiednio złota i malinowa, z różnymi wzorami kresek;
- `maps/cache/full_sky/16k/aurora_rgb_projected_16k.npy` — mapowalny cache RGB `float16`;
- `maps/cache/full_sky/16k/aurora_sky_map_hammer_16k_layout.npz` — informację o projekcji i
  wymiarach;
- `debug_rgb.png` i pomniejszone pliki diagnostyczne NPZ, jeżeli
  `SAVE_DEBUG_FRAMES = True`.

W trybie bez interakcji każdy wybór ma osobną opcję:
`--constellations`, `--coordinate-grid` i `--poland-limits`, każda z wartością
`yes` albo `no`.

Przy kolejnym uruchomieniu istniejący PNG jest traktowany jako gotowy wynik.
Jeżeli PNG nie istnieje, ale istnieje poprawny cache `.npy`, skrypt odtwarza z
niego mapę bez ponownego histogramowania. Po zmianie rozdzielczości lub danych
wejściowych trzeba usunąć albo zmienić nazwy nieaktualnych plików cache.

Próbkowanie jasności odbywa się w 64 ciągłych oknach FITS, co ogranicza
koszt losowych odczytów bardzo dużej kolumny. Renderer raportuje też czas
próbkowania oraz pamięć przeznaczaną na trzy histogramy (strumień, temperatura
ważona strumieniem i liczność).

#### Prostokątny wycinek nieba

```bash
python main/aurora_sky_region_render.py
```

Ten wariant korzysta z tego samego katalogu Gaia, kolorów, PSF i tone
mappingu, ale mapuje długość i szerokość galaktyczną bezpośrednio na prostokąt,
bez projekcji Hammera. Obszar ustala się przez cztery stałe:

- `REGION_L_CENTER_DEG` — galaktyczna długość środka kadru;
- `REGION_B_CENTER_DEG` — galaktyczna szerokość środka kadru;
- `REGION_L_WIDTH_DEG` — poziomy zakres długości galaktycznej;
- `REGION_B_HEIGHT_DEG` — pionowy zakres szerokości galaktycznej.

Na obrazie długość `l` maleje od lewej do prawej, a szerokość `b` maleje od
góry do dołu. Zakres długości może przechodzić przez granicę 0°/360°.
Bieżące wartości domyślne w kodzie odpowiadają presetowi 2 poniżej.

Gotowe opcjonalne presety, które można wkleić do konfiguracji:

**Preset 1 — lewa część mapy**

```python
REGION_L_CENTER_DEG = 115.06
REGION_B_CENTER_DEG = -1.86
REGION_L_WIDTH_DEG = 125.38
REGION_B_HEIGHT_DEG = 35.47
```

Przybliżone granice: lewa krawędź `l = 177.75°`, prawa krawędź
`l = 52.37°`, góra `b = 15.87°`, dół `b = -19.60°`.

**Preset 2 — centralna część mapy i centrum Galaktyki (bieżące wartości
domyślne)**

```python
REGION_L_CENTER_DEG = 3.23
REGION_B_CENTER_DEG = 0.59
REGION_L_WIDTH_DEG = 112.70
REGION_B_HEIGHT_DEG = 56.38
```

Przybliżone granice: lewa krawędź `l = 59.58°`, prawa krawędź
`l = 306.88°`, góra `b = 28.78°`, dół `b = -27.60°`.

**Preset 3 — prawa część mapy**

```python
REGION_L_CENTER_DEG = 249.98
REGION_B_CENTER_DEG = -1.01
REGION_L_WIDTH_DEG = 140.13
REGION_B_HEIGHT_DEG = 34.37
```

Przybliżone granice: lewa krawędź `l = 320.04°`, prawa krawędź
`l = 179.92°`, góra `b = 16.18°`, dół `b = -18.19°`.

Wartości presetów wyznaczono przez dopasowanie kadrów do referencyjnej mapy
Hammera o rozdzielczości 16 384 × 8 192 px i zastosowanie odwrotnej
projekcji. Pełna mapa korzysta z projekcji Hammera, natomiast
`aurora_sky_region_render.py` tworzy prostokątną mapę bez projekcji
kartograficznej. Render regionalny nie będzie więc pikselowo identyczny z
wycinkiem mapy Hammera: Droga Mleczna zostanie wyprostowana, a zakrzywione
czarne narożniki występujące przy krawędziach mapy Hammera nie pojawią się w
renderze regionalnym. Preset centralny będzie wizualnie najbardziej zbliżony
do kadru referencyjnego.

Profil 16K tworzy regionalny obraz o rozdzielczości 16 384 × 8 192 px, czyli
w proporcjach 2:1. Jeżeli stosunek
`REGION_L_WIDTH_DEG / REGION_B_HEIGHT_DEG` różni się od 2:1, program wyświetli
ostrzeżenie i przeskaluje obraz do ustalonych wymiarów. Nie należy zmieniać
podanych parametrów tylko po to, aby usunąć ostrzeżenie — opisują one
dopasowane zakresy nieba.

Wynik i pasujący layout trafiają do `maps/regions/`, a cache roboczy do
`maps/cache/regions/<profil>/16k/`. Metadane przechowują położenie, rozmiar
pola i wymiary obrazu, więc po zmianie tych wartości niezgodny cache nie jest
ponownie używany. Warianty regionalne otrzymują tę samą paletę i niezależne
sufiksy co mapa Hammera; siatka i obie granice są przycinane do wybranego
pola, dlatego linia niewchodząca w region nie jest rysowana.

### 2. Mikrosoczewkowanie

```bash
python microlensing/aurora_microlensing_catalog.py
python microlensing/aurora_microlensing_render.py
```

Downloader wykonuje jedno połączenie ADQL tabel
`gaiadr3.vari_microlensing` i `gaiadr3.gaia_source`, stosując między innymi
filtr `0.03 < parallax < 21` mas. Istniejący
`assets/aurora_microlensing.fits` jest używany ponownie, jeżeli zawiera
aktualny zestaw wymaganych kolumn. Starszy lub niepełny plik jest zastępowany
atomowo katalogiem w bieżącym schemacie.
Podobnie jak downloader dużego katalogu Gaia, skrypt wymaga danych logowania
podanych interaktywnie lub przez `GAIA_USER` i `GAIA_PASSWORD`.

Renderer zawsze korzysta z profilu 16K, ponieważ parametry
mikrosoczewkowania i rozmiar bufora są związane z tym formatem. Pyta jednak,
czy użyć całego nieba, czy gotowego wycinka; dla całego nieba pyta dodatkowo
o czyste tło albo wariant z gwiazdozbiorami.

Renderer:

- używa modelu punktowe źródło–punktowa soczewka Paczyńskiego;
- generuje 1000 klatek przy 25 FPS, czyli 40 sekund;
- prowadzi oś czasu od najwcześniejszego do najpóźniejszego `tmax`, dodając po
  obu stronach margines równy 95. percentylowi `tE`;
- renderuje w 16 384 × 8 192 px;
- zapisuje rzadkie parametry nakładek w `maps/cache/videos/`;
- koduje 10-bitowe HEVC (`yuv420p10le`) przez `libx265`;
- tworzy zwykły plik MP4 i kopię ze znacznikiem `hvc1`;
- na końcu pyta, czy utworzyć wersję mobilną 8K;
- ponownie wykorzystuje jeden bufor klatki 16K i raportuje osobno czas
  rasteryzacji oraz oczekiwania na potok FFmpeg.

Wygenerowany katalog nie zawiera mas soczewek. Renderer przyjmuje więc
domyślnie masę równą `1`; potrafi również odczytać opcjonalną kolumnę
`paczynski0_mass`, `paczynski_mass`, `lens_mass` albo `mass`.

Najważniejsze pliki:

```text
maps/cache/videos/aurora_microlensing*_preview_*.png
maps/cache/videos/frames_micro_overlay*/
videos/aurora_microlensing*_animation.mp4
videos/aurora_microlensing*_hvc1.mp4
videos/aurora_microlensing*_mobile.mp4    # tylko po potwierdzeniu
```

Interaktywny atlas korzysta z tego samego katalogu FITS oraz obrazu Hammera
wygenerowanego przez `aurora_sky_render.py`:

```bash
python microlensing/aurora_microlensing_map.py
```

Skrypt automatycznie znajduje oba pliki, wybiera wolny port i otwiera atlas w
przeglądarce. Mapa umożliwia płynne powiększanie, przesuwanie, filtrowanie i
wyszukiwanie zdarzeń. Najechanie pokazuje skrócony podgląd, a kliknięcie otwiera
pełną kartę z parametrami modelu, danymi gwiazdy źródłowej, krzywą
Paczyńskiego i znormalizowanym widokiem geometrii soczewkowania.
Opcje `--catalog` i `--map` pozwalają wskazać własne wejścia, `--host` i
`--port` konfigurują serwer, a `--no-browser` wyłącza automatyczne otwieranie
przeglądarki. Opcja `--verbose` włącza logi żądań HTTP. Serwer domyślnie
nasłuchuje wyłącznie na `127.0.0.1`.

Gaia DR3 nie identyfikuje w tej tabeli obiektu soczewkującego ani jego masy.
Jeżeli katalog zawiera kolumnę `lens_mass`, `mass`, `paczynski_mass` lub
`paczynski0_mass`, atlas wyświetla wartość bezpośrednio. W przeciwnym razie
może oszacować masę z `tE` i paralaksy źródła przy jawnych założeniach:
soczewka leży w połowie odległości do źródła, a względna prędkość poprzeczna
wynosi 100 km/s. Na tej podstawie prezentuje niskiej pewności hipotezę typu
soczewki; jest to wskazówka wizualna, nie pomiar. Atlas pokazuje również
obwiednię scenariuszy dla prędkości 50–200 km/s i względnej odległości
soczewki 0,25–0,75 odległości do źródła. Nie jest to przedział ufności.

### 3. Gwiazdy zmienne

Podstawowe katalogi gwiazd zmiennych można pobrać bezpośrednio z Gaia DR3:

```bash
python variables/aurora_variable_catalog.py
```

Downloader zapisuje w `assets/` pliki `rr_lyrae.fits`, `cepheids.fits`,
`zz_ceti.fits`, `lbv.fits` i `cataclysmic_variables.fits`; opcja
`--include-other` dodaje `other_variables.fits`. RR Lyrae i cefeidy
pochodzą z dedykowanych tabel SOS. Pozostałe katalogi używają ogólnego
klasyfikatora Gaia. Zapytania można przejrzeć bez pobierania przez
`--dry-run`; `--catalog`, `--row-limit`, `--min-score` i opcje prób
kontrolują zakres pobierania. Wyniki są zapisywane atomowo, a istniejące pliki
są zachowywane bez `--overwrite`.

Gaia publikuje gorące kandydaty jako szeroką klasę `BE|GCAS|SDOR|WR`.
Po pobraniu `lbv.fits` należy przygotować katalogi podtypów:

```bash
python variables/classify_hot_variables.py assets/lbv.fits \
  --output-dir assets --overwrite
```

Powstają `be.fits`, `gcas.fits`, `sdor.fits`, `wr.fits` oraz
`be_gcas_sdor_wr_unknown.fits`. Jest to ostrożna klasyfikacja kandydatów na
podstawie wielu cech, a nie zastępstwo klasyfikacji spektroskopowej. Jeżeli
kompletu pięciu plików nie ma, tryb wszystkich katalogów zachowuje zgodność ze
starszym `lbv.fits`.

Klasyfikator wylicza `bp_rp_intrinsic = bp_rp - ebpminrp_gspphot` tylko
wtedy, gdy oszacowanie reddeningu jest użyteczne. Kolumny
`reddening_quality` i `extinction_flags` opisują brakujące, niepewne lub
nieważne przedziały oraz silną ekstynkcję. Gaia `ag_gspphot` i
`ebpminrp_gspphot` pozostają oszacowaniami z niepewnościami: wspierają wybór
bezpiecznej barwy, ale same nie ustalają typu gwiazdy.

Uruchomienie renderera:

```bash
python variables/aurora_variable_animation.py
```

Pytania pojawiają się w następującej kolejności:

1. rozdzielczość `8`, `16`, `32` lub `64`;
2. długość całej symulacji w dniach, bezpośrednio po rozdzielczości
   (domyślnie 500);
3. całe niebo albo gotowy region; dla całego nieba także tło czyste lub z
   gwiazdozbiorami;
4. wszystkie katalogi albo jeden katalog.

Menu pojedynczego katalogu pokazuje wyłącznie RR Lyrae, Cepheids, ZZ Ceti,
cataclysmic, other i LBV. Dopiero wybór LBV otwiera podmenu: BE, GCAS,
LBV/SDOR, Wolf–Rayet, niesklasyfikowane gorące zmienne lub wszystkie LBV.
Można podać kilka numerów oddzielonych przecinkami, spacjami lub średnikami,
np. `1,2,3,4`. Te same wybory można automatyzować przez
`AURORA_RESOLUTION_K`, `AURORA_ANIMATION_DAYS`,
`AURORA_SKY_MAP_MODE`, `AURORA_SKY_MAP_BACKGROUND`,
`AURORA_VARIABLE_MODE` i `AURORA_LBV_GROUPS`.

Tryb all-sky używa mapy Hammera i obejmuje całe niebo. Tryb region korzysta z
gotowej prostokątnej mapy i jej pliku layoutu, filtruje katalog do wybranego
pola, ale zachowuje tę samą długość animacji, modele zmian i skalowanie
rozmiaru względem wysokości obrazu.

Kolejność ustalania koloru gwiazdy jest stała:

1. wiarygodne `teff_gspphot`;
2. wiarygodne `bp_rp_intrinsic`;
3. surowe `bp_rp`;
4. gorący kolor zapasowy zależny od klasy.

Surowe `bp_rp` nie jest podstawą klasyfikacji gorących gwiazd. BE i GCAS bez
temperatury używają najpierw wiarygodnego `bp_rp_intrinsic`. WR bez
wiarygodnej temperatury i barwy skorygowanej zawsze otrzymuje bezpieczny
gorący kolor klasowy. Dla SDOR/LBV czerwone surowe `bp_rp` jest pomijane,
gdy `ag_gspphot`, `ebpminrp_gspphot` lub flagi wskazują silną ekstynkcję.

W trybie wszystkich katalogów sigma gwiazdy ma bazę 3,2 px przy 16K, minimum
2,2 px i maksimum 12,5 px. Minimum jest skalowane z wysokością obrazu, więc
najmniejsze źródła pozostają widoczne także bez powiększenia. Tryb pojedynczego
katalogu używa większej referencyjnej bazy 11,616 px (3,2 × 2,20 × 1,65),
minimum 2,2 px i maksimum 20 px; dla gorących katalogów ograniczone korekty
promienia, jasności, masy i pewności mogą ją nieznacznie zmienić.

SDOR/LBV mają największe regularne, wolne pulsacje. BE, GCAS, WR oraz
niesklasyfikowane gorące zmienne używają deterministycznie odtwarzalnych,
gładkich losowych fluktuacji bez stałego okresu. Pozostałe klasy zachowują
własne stylizowane krzywe. Jest to wizualizacja, a nie dopasowanie fizycznej
krzywej blasku każdego źródła.

Losowanie jest rozdzielone między LMC/SMC i Drogę Mleczną. Limit klasy działa
osobno dla obu koszyków, a końcowy wspólny limit LMC/SMC wynosi 80 obiektów.
Bieżące limity na koszyk to:

| Klasa | Limit |
|---|---:|
| Cefeidy | 3 000 |
| RR Lyrae | 4 500 |
| LBV, BE, GCAS, SDOR, WR, gorące niesklasyfikowane | 12 000 |
| ZZ Ceti | 4 500 |
| Kataklizmiczne | 4 500 |
| Każda klasa z `other_variables.fits` | 1 500 |

```text
maps/cache/videos/aurora_variable*_preview_*.png
maps/cache/videos/frames_variable_overlay*/
videos/aurora_variable*_animation.mp4
videos/aurora_variable*_hvc1.mp4
videos/aurora_variable*_mobile.mp4      # tylko po potwierdzeniu
```

## Modele i przekształcenia

### Mapa nieba

Strumień wizualny źródła jest wyznaczany z jasności Gaia G:

$$F \propto 10^{-0.4G}.$$

Brakująca temperatura `teff_gspphot` jest szacowana z `bp_rp` wzorem
Ballesterosa. Skrypt akumuluje strumień, strumień ważony temperaturą i liczbę
źródeł na siatce 16 384 × 8 192. Następnie stosuje:

1. połączone jądro Gaussa i dwóch profili Moffata;
2. kompresję `log1p`, normalizację percentylową i rozciąganie `arcsinh`;
3. kolor z aproksymacji temperatury ciała czarnego;
4. zwiększenie nasycenia i korekcję gamma;
5. kafelkową rasteryzację odwrotnego przekształcenia Hammera.

Jasne źródła są ograniczane na podstawie 0,1 percentyla próbki jasności, aby
pojedyncze gwiazdy nie zdominowały struktury Drogi Mlecznej.

Ten sam rdzeń obliczeń jest używany przez mapę Hammera i renderer wycinka.
Funkcje współdzielone w `core/aurora_render_core.py` obejmują przeliczanie
`bp_rp` na temperaturę i RGB, akumulację histogramów, PSF Gaussa–Moffata,
projekcję Hammera, krzywe zmienności, model Paczyńskiego, rysowanie plam
Gaussa oraz sygnatury cache.

### Mikrosoczewkowanie

Wzmocnienie Paczyńskiego:

$$
u(t)=\sqrt{u_0^2+\left(\frac{t-t_0}{t_E}\right)^2},
\qquad
A(t)=\frac{u(t)^2+2}{u(t)\sqrt{u(t)^2+4}}.
$$

Oś animacji obejmuje wszystkie czasy maksimum i margines równy trzykrotności
95. percentyla `tE` przed pierwszym oraz po ostatnim maksimum. Nakładka ma
spłaszczony profil Gaussa: poza maksimum pozostaje gwiazdą bazową, a jej
rozmiar i przezroczystość płynnie rosną wraz z logarytmem `A(t) - 1`.

## Dane wejściowe

### `assets/aurora_gaia_catalog_900m.fits`

| Kolumna | Znaczenie |
|---|---|
| `l`, `b` | długość i szerokość galaktyczna w stopniach |
| `phot_g_mean_mag` | średnia jasność w paśmie G |
| `teff_gspphot` | temperatura efektywna; może być pusta |
| `bp_rp` | wskaźnik koloru używany jako zastępstwo temperatury |

### `assets/aurora_microlensing.fits`

| Kolumna | Znaczenie |
|---|---|
| `source_id` | identyfikator źródła Gaia |
| `l`, `b` | współrzędne galaktyczne w stopniach |
| `paczynski0_tmax` | czas maksimum jako BJD(TCB) − 2455197,5 |
| `paczynski0_te` | czas Einsteina w dniach |
| `paczynski0_u0` | minimalny parametr zderzenia |
| `parallax` | paralaksa w mas |
| `parallax_error` | niepewność paralaksy w mas |
| `phot_g_mean_mag` | średnia jasność w paśmie G |
| `teff_gspphot` | temperatura efektywna źródła; może być pusta |
| `bp_rp` | wskaźnik koloru używany jako zastępstwo temperatury |

### Katalogi gwiazd zmiennych

Pliki `rr_lyrae.fits`, `cepheids.fits`, `zz_ceti.fits`, `lbv.fits`,
`cataclysmic_variables.fits` i opcjonalny `other_variables.fits` mają wspólny
schemat przygotowywany przez downloader:

| Kolumna | Znaczenie |
|---|---|
| `source_id`, `l`, `b` | identyfikator oraz współrzędne galaktyczne |
| `parallax`, `parallax_error` | paralaksa i jej niepewność w mas |
| `phot_g_mean_mag`, `teff_gspphot`, `bp_rp` | fotometria, temperatura i surowa barwa |
| `ag_gspphot`, `ebpminrp_gspphot` oraz granice przedziałów | oszacowania ekstynkcji i reddeningu Gaia wraz z zakresem niepewności |
| `bp_rp_intrinsic` | barwa skorygowana o reddening, wyliczana przez klasyfikator gorących kandydatów |
| `reddening_quality`, `extinction_flags` | jakość korekty i jawne flagi braków, niepewności lub silnej ekstynkcji |
| `period`, `amplitude`, `phase` | parametry stylizowanej krzywej blasku |
| `variable_class`, `variable_subclass` | klasa AURORA i etykieta źródłowa Gaia |
| `classification_score` | wynik klasyfikatora; pusty dla tabel SOS |
| `lum_flame`, `radius_gspphot`, `mass_flame` | opcjonalne oszacowania fizyczne używane tylko do ograniczonego skalowania obrazu |

## Cache, pamięć i wznawianie

- Fragmenty `assets/gaia_chunks/gaia_chunk_NN.fits` pozwalają wznowić pobieranie
  dużego katalogu.
- Mapa 16K zawiera 134 217 728 pikseli. Jedna tablica `float32` o tym
  rozmiarze zajmuje około 512 MiB; trzy histogramy zajmują około 1,5 GiB,
  jeszcze przed buforami splotu i projekcji.
- Render mapy korzysta z przetwarzania FITS partiami, kafelkowego tone mappingu,
  `numpy.memmap` i splotu overlap-add. Mimo to mapa 16K wymaga wielu GiB RAM i
  wolnego miejsca na cache.
- Cache mapy jest pomocniczy, nie stanowi pełnego systemu checkpointów.
  Diagnostyczne histogramy nie są automatycznie wczytywane przy wznowieniu.
- Cache wycinka nie jest używany, jeśli zapisane położenie pola, rozmiar lub
  wymiary obrazu różnią się od bieżącej konfiguracji.
- Renderery mikrosoczewkowania i gwiazd zmiennych zapisują małe, rzadkie
  parametry nakładek zamiast pełnych obrazów 16K. Cache zawiera sygnaturę
  wszystkich wejść wpływających na klatkę i jest odrzucany, gdy wersja,
  wymiary lub sygnatura przestają pasować.
- Oba renderery animacji ponownie wykorzystują jeden bufor obrazu 16K zamiast
  alokować około 384 MiB na każdą klatkę.

## Ważne uwagi

- Zapytania do Gaia Archive wymagają połączenia z internetem i mogą być
  ograniczane przez usługę.
- Domyślne rendery 16K HEVC są bardzo kosztowne obliczeniowo i dyskowo. Do prób
  warto wybrać profil 8K; parametry obrazu i animacji są zebrane na początku
  odpowiednich plików.
- Renderery map i animacji używają pytań terminalowych oraz zmiennych
  środowiskowych opisanych w poszczególnych sekcjach.
- Downloader katalogów zmiennych i interaktywny atlas mają CLI; dostępne opcje
  pokazuje `--help`.
- Pliki wyjściowe katalogów są zapisywane atomowo. Katalogi z CLI wymagają
  `--overwrite`, aby zastąpić istniejący wynik.

## Pliki z pełną rozdzielczością

Duże przykładowe pliki wynikowe mogą być publikowane w sekcji
[Releases](../../releases/latest), ponieważ nie nadają się do przechowywania
bezpośrednio w repozytorium.

## Licencja i źródło danych

Kod jest udostępniany na licencji MIT — zobacz
[`LICENSE.txt`](LICENSE.txt).

Dane pochodzą z **Gaia Data Release 3** i Gaia Archive. Korzystając z danych,
należy przestrzegać zasad cytowania i uznania autorstwa publikowanych przez
Gaia DPAC oraz dokumentacji Gaia Archive.

---

<a id="english"></a>

## English

AURORA downloads and prepares **Gaia DR3** data and produces:

- an 8-bit Hammer-projection all-sky map;
- a rectangular, projection-free map of a selected sky region;
- a Paczynski microlensing animation;
- an interactive atlas of all microlensing events;
- a stylised variable-star animation.

### Modules

| Script | Purpose |
|---|---|
| `main/aurora_gaia_catalog.py` | Downloads a resumable catalogue of up to 1,811,709,771 Gaia DR3 sources and combines the FITS chunks. |
| `main/aurora_sky_render.py` | Builds a 16K Hammer all-sky map from the Gaia catalogue. |
| `main/aurora_sky_region_render.py` | Renders a configurable rectangular sky region without a cartographic projection. |
| `microlensing/aurora_microlensing_catalog.py` | Downloads events from `gaiadr3.vari_microlensing` with source positions and photometry. |
| `microlensing/aurora_microlensing_render.py` | Overlays microlensing events on the Hammer map and encodes HEVC video. |
| `microlensing/aurora_microlensing_map.py` | Runs a local interactive event atlas with smooth zooming and full data cards. |
| `variables/aurora_variable_catalog.py` | Downloads and normalizes Gaia DR3 variable-star catalogues. |
| `variables/classify_hot_variables.py` | Splits the broad `BE|GCAS|SDOR|WR` catalogue into cautiously scored candidate catalogues and derives corrected colour. |
| `variables/aurora_variable_animation.py` | Overlays the core and remaining variable-star classes on the Hammer map and encodes HEVC video. |
| `core/aurora_console.py` | Standardises terminal headings, statuses, errors, and progress bars for every program. |
| `core/aurora_render_core.py` | Provides shared stellar colours, Hammer projection, PSF, light curves, rasterisation, and cache signatures. |
| `core/aurora_paths.py` | Defines canonical `assets/`, `maps/`, `maps/cache/`, and `videos/` paths. |

### Data flow

```text
ALL-SKY MAP
Gaia DR3
  └─ main/aurora_gaia_catalog.py
       └─ assets/aurora_gaia_catalog_900m.fits
            ├─ main/aurora_sky_render.py
            │    └─ maps/aurora_sky_map_hammer_16k.png
            └─ main/aurora_sky_region_render.py
                 └─ maps/regions/aurora_sky_region_rect_pic1_16k.png

MICROLENSING
Gaia DR3
  └─ microlensing/aurora_microlensing_catalog.py
       └─ assets/aurora_microlensing.fits
            ┐
16K map ────┴─ microlensing/aurora_microlensing_render.py
                 ├─ aurora_microlensing_animation.mp4
                 └─ aurora_microlensing_16k_hvc1.mp4
           └──── microlensing/aurora_microlensing_map.py
                  └─ local interactive atlas

VARIABLE STARS
Gaia DR3
  └─ variables/aurora_variable_catalog.py
       └─ assets/*.fits + 16K map
            └─ variables/aurora_variable_animation.py
                 ├─ aurora_variable_animation.mp4
                 └─ aurora_variable_16k_hvc1.mp4
```

### Shared assets

The `assets/` directory contains FITS input catalogues and JSON configuration.
Final maps are stored in `maps/`, regional maps and layouts in
`maps/regions/`, caches in `maps/cache/`, and MP4 outputs in `videos/`.

```text
assets/aurora_gaia_catalog_900m.fits
assets/aurora_microlensing.fits
assets/rr_lyrae.fits
assets/cepheids.fits
assets/zz_ceti.fits
assets/lbv.fits
assets/be.fits, gcas.fits, sdor.fits, wr.fits
assets/be_gcas_sdor_wr_unknown.fits
assets/cataclysmic_variables.fits
assets/other_variables.fits          # optional
```

All paths are resolved from the repository root, independently of the current
working directory.

## Installation

Python **3.12+** is required. The source syntax is compatible with older
versions, but the NumPy and SciPy releases pinned in `requirements.txt`
require Python 3.12 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\activate
```

### Terminal output

Every program uses `core/aurora_console.py`, so headings, separators,
diagnostic messages, and progress bars follow the same format:

```text
[AURORA] Stage name
────────────────────────────────────────────────────────────
  → operation in progress or information
  ✓ operation completed successfully
  ! warning after which the program can continue
  ✗ error written to stderr
  ? question that requires user input
  ↻ Stage:  50%|############            | 5/10 [00:01<00:01]
```

Progress bars animate only in an interactive terminal by default, which keeps
redirected logs clean. Set `AURORA_PROGRESS` to `auto` (the default), `always`,
or `never` to override this behaviour. Console formatting itself uses only the
standard library; `tqdm` from `requirements.txt` provides progress bars.

### FFmpeg

`microlensing/aurora_microlensing_render.py` and
`variables/aurora_variable_animation.py` require **FFmpeg** in `PATH` with the
`libx265` encoder.

```bash
# macOS
brew install ffmpeg

# Ubuntu / Debian
sudo apt update
sudo apt install ffmpeg

# verify
ffmpeg -hide_banner -encoders | grep libx265
```

## Quick start

The commands below assume the project directory as the working directory.
Paths are rooted at the project: inputs under `assets/`, maps and caches under
`maps/`, and videos under `videos/`.

### 1. All-sky map

```bash
python main/aurora_gaia_catalog.py
python main/aurora_sky_render.py
```

The Gaia downloader:

- splits the download into 61 deterministic `random_index` ranges;
- requests up to 30,000,000 records per range, up to the configured total of
  1,811,709,771 sources;
- stores chunks in `assets/gaia_chunks/gaia_chunk_NN.fits`;
- skips existing chunks, allowing interrupted downloads to resume;
- retries a query up to five times with a one-hour timeout per attempt;
- assembles the FITS file by copying binary table payloads directly, without
  materialising every row in RAM.

The large catalogue download requires authenticated Gaia Archive access.
Credentials may be entered interactively or supplied through the environment:

```bash
export GAIA_USER="username"
export GAIA_PASSWORD="password"
python main/aurora_gaia_catalog.py
```

The downloader writes the final catalogue directly to
`assets/aurora_gaia_catalog_900m.fits`. An existing output is not overwritten.
The historical `900m` filename is retained for path compatibility even though
the current configured catalogue limit is larger.

At startup `main/aurora_sky_render.py` asks for `8`, `16`, `32` or `64` (Enter
selects 16K). The selected profile supplies the raster dimensions, `BINS_L/B`,
and resolution-specific map/layout/cache names. For non-interactive runs set
`AURORA_RESOLUTION_K=8` (or 16/32/64). The sky renderer then produces the
matching variant, for example:

- `maps/aurora_sky_map_hammer_16k.png` — an 8-bit RGB,
  16,384 × 8,192 PNG;
- optional variants with independently selected constellation lines, a subtle
  Galactic coordinate grid, and two Poland visibility limits. The renderer
  asks three separate questions and gives every combination its own suffix,
  such as `_constellations.png`, `_coordinates.png`, `_poland_limits.png`, or
  `_constellations_coordinates_poland_limits.png`. The grid is neutral grey,
  constellations remain pale blue, and the `dec=-38 deg` theoretical horizon
  plus `dec=-28 deg` practical 10-degree-altitude limit use distinct gold and
  raspberry dash patterns;
- `maps/cache/full_sky/16k/aurora_rgb_projected_16k.npy` — a memory-mappable `float16` RGB
  cache;
- `maps/cache/full_sky/16k/aurora_sky_map_hammer_16k_layout.npz` — projection and dimension
  metadata;
- `debug_rgb.png` and downsampled diagnostic NPZ files when
  `SAVE_DEBUG_FRAMES = True`.

For non-interactive runs, set each switch separately with `--constellations`,
`--coordinate-grid`, and `--poland-limits`; each accepts `yes` or `no`.

An existing map is considered complete. If the PNG is missing but the `.npy`
cache exists, the renderer reconstructs the map without rebuilding the
histograms. Remove or rename stale outputs after changing the input or
resolution.

Magnitude sampling uses 64 contiguous FITS windows to avoid expensive random
reads across a very large column. The renderer also reports sampling time and
the memory allocated for its three histograms (flux, temperature-weighted flux,
and source count).

#### Rectangular sky region

```bash
python main/aurora_sky_region_render.py
```

This variant reuses the same Gaia input, stellar colours, PSF, and tone
mapping, but maps Galactic longitude and latitude directly onto a rectangle
without the Hammer projection. Configure the field with four constants:

- `REGION_L_CENTER_DEG` — the Galactic longitude at the centre of the frame;
- `REGION_B_CENTER_DEG` — the Galactic latitude at the centre of the frame;
- `REGION_L_WIDTH_DEG` — the horizontal Galactic-longitude span;
- `REGION_B_HEIGHT_DEG` — the vertical Galactic-latitude span.

In the image, longitude `l` decreases from left to right and latitude `b`
decreases from top to bottom. The longitude range may cross the 0°/360°
boundary. The current default values in the code match preset 2 below.

Ready-to-use optional presets that can be pasted into the configuration:

**Preset 1 — left part of the map**

```python
REGION_L_CENTER_DEG = 115.06
REGION_B_CENTER_DEG = -1.86
REGION_L_WIDTH_DEG = 125.38
REGION_B_HEIGHT_DEG = 35.47
```

Approximate bounds: left edge `l = 177.75°`, right edge `l = 52.37°`, top
`b = 15.87°`, bottom `b = -19.60°`.

**Preset 2 — central part of the map and the Galactic Centre (current
defaults)**

```python
REGION_L_CENTER_DEG = 3.23
REGION_B_CENTER_DEG = 0.59
REGION_L_WIDTH_DEG = 112.70
REGION_B_HEIGHT_DEG = 56.38
```

Approximate bounds: left edge `l = 59.58°`, right edge `l = 306.88°`, top
`b = 28.78°`, bottom `b = -27.60°`.

**Preset 3 — right part of the map**

```python
REGION_L_CENTER_DEG = 249.98
REGION_B_CENTER_DEG = -1.01
REGION_L_WIDTH_DEG = 140.13
REGION_B_HEIGHT_DEG = 34.37
```

Approximate bounds: left edge `l = 320.04°`, right edge `l = 179.92°`, top
`b = 16.18°`, bottom `b = -18.19°`.

The preset values were obtained by fitting frames to a reference Hammer map
at 16,384 × 8,192 px and applying the inverse projection. The full map uses
the Hammer projection, whereas `aurora_sky_region_render.py` creates a
rectangular map without a cartographic projection. A regional render will
therefore not be pixel-identical to a crop of the Hammer map: the Milky Way
will be straightened, and the curved black corners near the Hammer map's edges
will not appear in the regional render. The central preset will be visually
the closest match to the reference frame.

The 16K profile creates a 16,384 × 8,192 regional image, giving a 2:1 aspect
ratio. If `REGION_L_WIDTH_DEG / REGION_B_HEIGHT_DEG` differs from 2:1, the
program prints a warning and scales the image to the fixed dimensions. Do not
change the supplied parameters merely to remove the warning, because they
describe the fitted sky ranges.

The PNG and matching layout are stored in `maps/regions/`, with working caches
under `maps/cache/regions/<profile>/16k/`. Metadata stores the field position,
angular size, and output dimensions, so a mismatched cache is not reused after
these values change. Regional variants receive the same palette and independent
suffixes as the Hammer map; the grid and both Poland limits are clipped to the
selected field, so a curve that does not cross the region is not drawn.

### 2. Microlensing

```bash
python microlensing/aurora_microlensing_catalog.py
python microlensing/aurora_microlensing_render.py
```

The downloader joins `gaiadr3.vari_microlensing` with
`gaiadr3.gaia_source` and applies a `0.03 < parallax < 21` mas filter. An
existing `assets/aurora_microlensing.fits` is reused when it contains the
current set of required columns. An older or incomplete file is atomically
replaced with the current schema.
Like the large Gaia-catalogue downloader, it requires credentials entered
interactively or supplied through `GAIA_USER` and `GAIA_PASSWORD`.

The renderer always uses the fixed 16K profile because the microlensing buffer
and map format are coupled. It still asks for all-sky versus a prepared region;
in all-sky mode it also offers all eight pre-rendered combinations of the
plain map, constellations, coordinate grid, and Poland limits. These lines are
never redrawn for individual animation frames.

The renderer:

- uses the point-source, point-lens Paczynski model;
- creates 1,000 frames at 25 FPS (40 seconds);
- spans the earliest through latest `tmax`, with a margin equal to the
  95th percentile of `tE` on each side;
- renders at 16,384 × 8,192;
- caches sparse overlay parameters in `maps/cache/videos/`;
- encodes 10-bit HEVC (`yuv420p10le`) through `libx265`;
- creates the normal MP4 and an `hvc1`-tagged copy;
- optionally creates an 8K mobile version after an interactive prompt;
- reuses one 16K frame buffer and reports rasterisation and FFmpeg pipe-wait
  time separately.

The downloaded catalogue has no lens-mass column, so the renderer normally
uses a mass of `1`. It also recognises optional `paczynski0_mass`,
`paczynski_mass`, `lens_mass`, or `mass` columns.

Main renderer outputs:

```text
maps/cache/videos/aurora_microlensing*_preview_*.png
maps/cache/videos/frames_micro_overlay*/
videos/aurora_microlensing*_animation.mp4
videos/aurora_microlensing*_hvc1.mp4
videos/aurora_microlensing*_mobile.mp4    # only after confirmation
```

The interactive atlas uses the same FITS catalogue and Hammer image:

```bash
python microlensing/aurora_microlensing_map.py
```

It automatically discovers both files, chooses a free local port, and opens
the browser. The atlas supports smooth pan and zoom, filters, search, hover
previews, complete event cards, Paczynski curves, and a normalized
third-person view of the lensing geometry. Use `--catalog` and `--map` for
custom inputs, `--host` and `--port` to configure the server, and
`--no-browser` to disable automatic browser launch. `--verbose` enables HTTP
request logs. By default the server listens only on `127.0.0.1`.

If the catalogue contains
`lens_mass`, `mass`, `paczynski_mass`, or `paczynski0_mass`, the atlas uses
that value. Otherwise it can estimate the mass from `tE` and source parallax,
explicitly assuming a lens halfway to the source and a 100 km/s relative
transverse speed. The resulting low-confidence lens-type hypothesis is a
visual aid, not a measurement. The atlas also shows a scenario envelope for
speeds of 50–200 km/s and lens-distance fractions of 0.25–0.75. This is not a
confidence interval.

### 3. Variable stars

The core variable-star catalogues can be downloaded directly from Gaia DR3:

```bash
python variables/aurora_variable_catalog.py
```

The downloader writes `rr_lyrae.fits`, `cepheids.fits`, `zz_ceti.fits`,
`lbv.fits`, and `cataclysmic_variables.fits` under `assets/`;
`--include-other` adds `other_variables.fits`. RR Lyrae and Cepheids come
from dedicated SOS tables. The remaining catalogues use Gaia's general
classifier. Use `--dry-run` to inspect ADQL without downloading;
`--catalog`, `--row-limit`, `--min-score`, and retry options control the
request. Outputs are atomic, and existing files are preserved unless
`--overwrite` is supplied.

Gaia publishes hot candidates under the broad `BE|GCAS|SDOR|WR` class.
After downloading `lbv.fits`, prepare the subtype catalogues:

```bash
python variables/classify_hot_variables.py assets/lbv.fits \
  --output-dir assets --overwrite
```

This creates `be.fits`, `gcas.fits`, `sdor.fits`, `wr.fits`, and
`be_gcas_sdor_wr_unknown.fits`. The multi-feature rules produce cautious
candidate classifications, not replacements for spectroscopy. If the complete
set of five split files is unavailable, all-catalogue mode retains
compatibility with the older `lbv.fits`.

The classifier computes `bp_rp_intrinsic = bp_rp - ebpminrp_gspphot` only
when the reddening estimate is usable. `reddening_quality` and
`extinction_flags` record missing, uncertain, invalid intervals and strong
extinction. Gaia `ag_gspphot` and `ebpminrp_gspphot` remain estimates with
uncertainties: they support a safe colour choice but never determine the
stellar class by themselves.

Run the renderer with:

```bash
python variables/aurora_variable_animation.py
```

Prompts appear in this order:

1. resolution: `8`, `16`, `32`, or `64`;
2. the global simulated duration in days, immediately after resolution
   (default 500);
3. all-sky or a prepared region; all-sky additionally offers all eight
   pre-rendered combinations of constellations, coordinates, and Poland limits;
4. all catalogues or one catalogue.

The single-catalogue menu contains only RR Lyrae, Cepheids, ZZ Ceti,
cataclysmic, other, and LBV. Selecting LBV then opens a submenu for BE, GCAS,
LBV/SDOR, Wolf–Rayet, unclassified hot variables, or all LBV catalogues.
Multiple numbers may be separated by commas, spaces, or semicolons, for
example `1,2,3,4`. The same choices can be automated with
`AURORA_RESOLUTION_K`, `AURORA_ANIMATION_DAYS`,
`AURORA_SKY_MAP_MODE`, `AURORA_SKY_MAP_BACKGROUND`,
`AURORA_VARIABLE_MODE`, and `AURORA_LBV_GROUPS`.

All-sky mode uses the Hammer map and covers the complete sky. Region mode uses
a prepared rectangular map and its layout file, filters the catalogue to that
field, and keeps the same global duration, variability models, and
height-relative size scaling.

Stellar colour follows a fixed precedence:

1. reliable `teff_gspphot`;
2. reliable `bp_rp_intrinsic`;
3. raw `bp_rp`;
4. a hot class-dependent fallback.

Raw `bp_rp` is not the primary basis for classifying hot stars. BE and GCAS
without temperature first use reliable `bp_rp_intrinsic`. A WR star without
reliable temperature or corrected colour always receives a safely hot class
colour. For SDOR/LBV, red raw `bp_rp` is ignored when `ag_gspphot`,
`ebpminrp_gspphot`, or the flags indicate strong extinction.

In all-catalogue mode, stellar sigma has a 3.2 px base at 16K, a 2.2 px
minimum, and a 12.5 px maximum. The minimum scales with image height, keeping
the smallest sources visible without zooming. Single-catalogue mode uses the
larger 11.616 px reference base (3.2 × 2.20 × 1.65), a 2.2 px minimum, and a
20 px maximum; bounded radius, luminosity, mass, and confidence adjustments
may slightly modify hot-catalogue sizes.

SDOR/LBV receive the largest slow, regular pulsations. BE, GCAS, WR, and
unclassified hot variables use reproducible smooth random fluctuations with
no fixed period. Other classes retain their own stylised curves. This is a
visualisation rather than a fitted physical light curve for each source.

Sampling is split between the LMC/SMC and Milky Way. Each class limit applies
independently to both buckets, followed by a shared LMC/SMC cap of 80 objects.
Current per-bucket limits are:

| Class | Limit |
|---|---:|
| Cepheids | 3,000 |
| RR Lyrae | 4,500 |
| LBV, BE, GCAS, SDOR, WR, unclassified hot | 12,000 |
| ZZ Ceti | 4,500 |
| Cataclysmic | 4,500 |
| Each class in `other_variables.fits` | 1,500 |

```text
maps/cache/videos/aurora_variable*_preview_*.png
maps/cache/videos/frames_variable_overlay*/
videos/aurora_variable*_animation.mp4
videos/aurora_variable*_hvc1.mp4
videos/aurora_variable*_mobile.mp4      # only after confirmation
```

## Models and transformations

### Sky map

Visual flux is derived from Gaia G magnitude:

$$F \propto 10^{-0.4G}.$$

Missing `teff_gspphot` values are estimated from `bp_rp` with the Ballesteros
formula. The renderer accumulates flux, temperature-weighted flux, and source
count on a 16,384 × 8,192 grid, then applies:

1. a combined Gaussian and dual-Moffat kernel;
2. `log1p`, percentile normalisation, and `arcsinh` compression;
3. blackbody-like temperature colour;
4. saturation enhancement and gamma correction;
5. tiled inverse Hammer rasterisation.

Bright-source flux is clipped from a magnitude sample so a few stars do not
dominate the Milky Way structure.

The Hammer and regional maps now use the same computational core.
`core/aurora_render_core.py` centralises BP−RP temperature and RGB conversion,
histogram accumulation, the Gaussian–Moffat PSF, Hammer projection, variable
light curves, the Paczynski model, Gaussian spot drawing, and cache
signatures.

### Microlensing

The Paczynski amplification is:

$$
u(t)=\sqrt{u_0^2+\left(\frac{t-t_0}{t_E}\right)^2},
\qquad
A(t)=\frac{u(t)^2+2}{u(t)\sqrt{u(t)^2+4}}.
$$

The animation timeline covers every peak time, plus the 95th percentile of
`tE` before the first and after the last peak. The
flattened Gaussian overlay remains a base star away from the peak, then grows
smoothly in size and opacity with the logarithm of `A(t) - 1`.

## Input data

### `assets/aurora_gaia_catalog_900m.fits`

| Column | Meaning |
|---|---|
| `l`, `b` | Galactic longitude and latitude in degrees |
| `phot_g_mean_mag` | mean Gaia G magnitude |
| `teff_gspphot` | effective temperature; may be missing |
| `bp_rp` | colour index used as a temperature fallback |

### `assets/aurora_microlensing.fits`

| Column | Meaning |
|---|---|
| `source_id` | Gaia source identifier |
| `l`, `b` | Galactic coordinates in degrees |
| `paczynski0_tmax` | peak time as BJD(TCB) − 2455197.5 |
| `paczynski0_te` | Einstein time in days |
| `paczynski0_u0` | minimum impact parameter |
| `parallax` | parallax in mas |
| `parallax_error` | parallax uncertainty in mas |
| `phot_g_mean_mag` | mean Gaia G magnitude |
| `teff_gspphot` | source effective temperature; may be missing |
| `bp_rp` | colour index used as a temperature fallback |

### Variable-star catalogues

`rr_lyrae.fits`, `cepheids.fits`, `zz_ceti.fits`, `lbv.fits`,
`cataclysmic_variables.fits`, and optional `other_variables.fits` share the
schema produced by the downloader:

| Column | Meaning |
|---|---|
| `source_id`, `l`, `b` | source identifier and Galactic coordinates |
| `parallax`, `parallax_error` | parallax and its uncertainty in mas |
| `phot_g_mean_mag`, `teff_gspphot`, `bp_rp` | photometry, temperature, and raw colour |
| `ag_gspphot`, `ebpminrp_gspphot`, and interval bounds | Gaia extinction and reddening estimates with uncertainty ranges |
| `bp_rp_intrinsic` | reddening-corrected colour derived by the hot-candidate classifier |
| `reddening_quality`, `extinction_flags` | correction quality and explicit missing, uncertain, invalid, or strong-extinction flags |
| `period`, `amplitude`, `phase` | stylised light-curve parameters |
| `variable_class`, `variable_subclass` | AURORA class and source Gaia label |
| `classification_score` | classifier score; missing for SOS tables |
| `lum_flame`, `radius_gspphot`, `mass_flame` | optional physical estimates used only for bounded display scaling |

## Cache, memory, and resuming

- `assets/gaia_chunks/gaia_chunk_NN.fits` files make the large Gaia download
  resumable.
- A 16K map has 134,217,728 pixels. One `float32` array of that size is about
  512 MiB; the three base histograms use about 1.5 GiB before convolution and
  projection buffers.
- The sky renderer streams FITS rows, tiles tone mapping and projection, uses
  `numpy.memmap`, and performs overlap-add convolution. It still needs several
  GiB of RAM and cache space.
- Diagnostic histogram files are not automatically loaded as checkpoints.
- The regional cache is ignored when its stored field position, angular size,
  or output dimensions differ from the current configuration.
- The microlensing and variable-star renderers cache sparse overlay parameters
  instead of full 16K images. Each cache entry includes a signature of every
  input affecting the frame and is rejected when its version, dimensions, or
  signature no longer match.
- Both animation renderers reuse a single 16K image buffer instead of
  allocating roughly 384 MiB for every frame.

## Important notes

- Gaia Archive queries require internet access and may be rate-limited.
- Default 16K HEVC renders are computationally and storage intensive. Choose
  the 8K profile for trial runs; image and animation settings are grouped near
  the top of the relevant files.
- Map and animation renderers use terminal prompts and the environment
  variables documented in their respective sections.
- The variable-catalogue downloader and interactive atlas do have CLIs; use
  `--help` to list their options.
- Catalogue outputs are written atomically. CLI catalogue generators require
  `--overwrite` to replace an existing file.

## Full-resolution files

Large example outputs may be published under
[Releases](../../releases/latest) rather than stored directly in the
repository.

## Licence and data source

The software is available under the MIT Licence; see
[`LICENSE.txt`](LICENSE.txt).

Astronomical data come from **Gaia Data Release 3** and the Gaia Archive.
Follow the Gaia DPAC acknowledgement and citation guidance when publishing
work based on these data.
