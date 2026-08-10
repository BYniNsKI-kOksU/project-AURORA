# AURORA Supernova

Moduł `supernova` generuje półanalityczne krzywe bolometryczne i wielopasmowe oraz
renderuje pojedynczy, niecykliczny wybuch na tłach AURORA. Obsługuje typy
`II-P`, `II-L`, `IIn`, `Ib`, `Ic` oraz referencyjny model `Ia`.

To narzędzie edukacyjne i wizualizacyjne, a nie kod hydrodynamiki radiacyjnej
ani system przewidywania wybuchów. Szczególnie scenariusze Betelgezy i Eta
Carinae nie określają daty wybuchu ani jednego „prawdziwego” przebiegu.

## Struktura

| Plik | Odpowiedzialność |
|---|---|
| `progenitor.py` | Walidowane parametry progenitora, skład, typ SN i nadpisania kalibracyjne. |
| `units.py` | SI, lata świetlne, parseki, kiloparseki, megaparseki i płaska kosmologia ΛCDM. |
| `explosion.py` | Energia, masa wyrzutu, pozostałość, prędkość, dyfuzja i pułapkowanie gamma. |
| `light_curve.py` | Składowe krzywej blasku, temperatura, ekstynkcja i fotometria obserwowana. |
| `photometry.py` | Edukacyjne pasma UV/U/B/V/R/I/IR, ekstynkcja i skala AB. |
| `scenarios.py` | Odczyt JSON, niepewności i Monte Carlo. |
| `plotting.py` | Wykresy SVG, porównania oraz pasy niepewności Monte Carlo. |
| `background.py` | `all_sky`, region, własny obraz i własny katalog gwiazd. |
| `animation.py` | Jednorazowy rozbłysk, zanik i rozszerzająca się powłoka. |
| `text_overlay.py` | Etykiety filtra, czasu, typu i statusu bez dodatkowych zależności. |
| `run_supernova.py` | Jeden program startowy z ustawieniami wpisanymi w kodzie. |
| `cli.py` | Interfejs poleceń. |
| `scenarios/historical/*.json` | Zdarzenia historycznie obserwowane. |
| `scenarios/hypothetical/*.json` | Jawnie niepredykcyjne scenariusze przyszłe. |
| `tests/` | Testy jednostek, fizyki, scenariuszy i skali animacji. |

Kod istniejących modułów AURORA nie został zmieniony. Integracja wykorzystuje
te same funkcje projekcji Hammera i regionu, konfigurację osi redakcyjnej oraz
kodowanie HEVC co
`variables/aurora_variable_animation.py`. Zamiast okresowego
`variable_frame_parameters` animator dostaje jeden obserwatorowy czas od
wybuchu. Po maksimum czas nigdy nie jest zawijany modulo okresu.

## Uruchamianie

Moduł korzysta z zależności już wymienionych w głównym `requirements.txt`.
Renderowanie wideo wymaga także FFmpeg z `libx265`.

Z katalogu głównego projektu:

```bash
python -m supernova
```

Albo, jeśli wolisz jawnie wskazać plik startowy:

```bash
python supernova/run_supernova.py
```

Po komendzie `python run_supernova.py` uruchamia się krótki kreator zgodny ze
stylem `microlensing_render` i `variable_animation`. Interfejs programu jest po
angielsku. Kreator pyta wyłącznie o decyzje potrzebne do wyboru renderu:

- operację;
- jeden scenariusz albo grupę historyczną/hipotetyczną/wszystkie;
- osobne filmy albo jedną wspólną animację grupy;
- filtr `UV/U/B/V/R/I/IR`;
- standardową rozdzielczość AURORA `8K/16K/32K/64K`;
- pełne niebo lub region, gdy wybrany układ na to pozwala;
- gotowy wariant mapy tła AURORA.

Rozdzielczość, zasięg mapy, wariant tła i selektor regionów używają bezpośrednio
tych samych funkcji pytań co pozostałe renderery projektu. Dla wspólnej animacji
pełne niebo wybierane jest automatycznie, ponieważ pojedynczy region nie obejmie
obiektów rozrzuconych po niebie.

Parametry techniczne nie są pytaniami. FPS, długość filmu, CRF, preset x265,
liczba próbek, ścieżki, zakres czasu, powłoka, halo i etykiety korzystają z
ustalonych wartości projektu w sekcji `USTAWIENIA STARTOWE`. Nadal można je tam
zmienić dla niestandardowego renderu lub uruchomienia wsadowego.

Renderer korzysta z tego samego automatycznego ogranicznika RAM co
`variable_animation` i `microlensing_render`. Kontroler zachowuje dla systemu
15% pamięci fizycznej (minimum 2 GiB, maksimum 8 GiB) i spowalnia generowanie
klatek przy presji pamięci. Nie dodaje to żadnego pytania do kreatora. W trybie
wsadowym wspólne ustawienia `AURORA_MAX_RAM_GB`, `AURORA_MIN_FREE_RAM_GB` i
`AURORA_MEMORY_WAIT` mogą nadpisać wartości automatyczne.

```bash
cd supernova
python run_supernova.py
```

Podgląd klatki maksimum jest zapisywany obok filmu jako
`sn1987a_supernova_preview.png`.

Po ostatnim wyborze obliczenia rozpoczynają się od razu. Nie ma dodatkowego
podsumowania ani pytania potwierdzającego. W trybie regionu uruchamia się
istniejący selektor sparowanych map PNG/NPZ AURORA.

### Renderowanie całych grup scenariuszy

Nie trzeba wpisywać nazw scenariuszy ani używać opcji terminala. Kreator
wyświetla trzy grupy: wszystkie historyczne, wszystkie hipotetyczne oraz
wszystkie scenariusze. Następne pytanie pozwala wybrać osobne filmy albo
wszystkie zjawiska jednocześnie na jednej animacji:

```python
RUN_MODE = "render"

# Wszystkie obserwowane historycznie:
RENDER_SCENARIO_GROUP = "historical"

# Wszystkie hipotetyczne:
RENDER_SCENARIO_GROUP = "hypothetical"

# Wszystkie scenariusze, również referencyjna Ia:
RENDER_SCENARIO_GROUP = "all"

# Dotychczasowe osobne filmy, np. 12 MP4 dla "all":
RENDER_GROUP_LAYOUT = "separate"

# Jeden MP4 ze wszystkimi wybranymi zjawiskami na jednej mapie:
RENDER_GROUP_LAYOUT = "combined"
```

Dotychczasowy pojedynczy render pozostaje dostępny jako:

```python
RENDER_SCENARIO_GROUP = "single"
SCENARIO = "sn1987a"
```

Aktualny katalog zawiera 7 scenariuszy historycznych, 4 hipotetyczne i 12 po
wybraniu `all` — ostatnia liczba obejmuje również osobny model referencyjny Ia.
W układzie `separate` każdy scenariusz tworzy własny MP4 oraz PNG podglądu.
Render odbywa się sekwencyjnie, aby kilka procesów 16K nie zajmowało pamięci
jednocześnie. Ten dotychczasowy tryb nie został usunięty.

W układzie `combined` powstaje jeden MP4 i jeden podgląd PNG. Wszystkie wybuchy
mają wspólne `t=0`, dzięki czemu można porównać ich narastanie, maksimum,
plateau i ogon w tej samej animacji. Każdy obiekt zachowuje jednak własny model
czasu obserwatora, odległość, ekstynkcję, jasność pozorną, temperaturę i
ewolucję powłoki. Jasności nie są normalizowane do wspólnego efektownego
poziomu. Przy każdym punkcie widnieje klucz scenariusza i bieżąca magnitudo, a
legenda wyjaśnia, że warstwy powłoki i halo są ilustracyjne.

Animacja zbiorcza wymaga tła obejmującego wszystkie położenia. Kreator udostępnia
dla niej `all_sky`, własny pełnosferyczny obraz Hammera albo katalog gwiazd;
pojedynczy region jest celowo wyłączony, ponieważ nie może jednocześnie zawierać
obiektów rozrzuconych po całym niebie.

Jeżeli `VIDEO_OUTPUT_PATH = None`, nazwa scenariusza jest już częścią nazwy
każdego pliku. Jeżeli ustawiono własną nazwę, w renderze grupowym program
automatycznie dopisuje klucz scenariusza, np.
`kolekcja_sn1987a.mp4` i `kolekcja_betelgeuse.mp4`, więc pliki nie są
nadpisywane. Wspólna animacja dostaje nazwę w rodzaju
`combined_all_supernova_v.mp4`.

Render na gotowej mapie całego nieba AURORA:

```python
RUN_MODE = "simulate_and_render"
SCENARIO = "sn1987a"
BACKGROUND_MODE = "all_sky"
AURORA_RESOLUTION = "16k"
VIDEO_MATCH_BACKGROUND_SIZE = True
VIDEO_WIDTH = 16384
VIDEO_HEIGHT = 8192
VIDEO_CRF = 10
VIDEO_ENCODER_PRESET = "slow"
```

Mapa wejściowa jest wyszukiwana jako
`maps/aurora_sky_map_hammer_16k.png`. Ustawienie `SHOW_CONSTELLATIONS = True`
wybiera wariant z gwiazdozbiorami. Przy domyślnym
`VIDEO_MATCH_BACKGROUND_SIZE = True` film dostaje dokładnie wymiary obrazu,
czyli dla zwykłej mapy 16K `16384 × 8192`. Mapa nie przechodzi wtedy przez
filtr skalujący, więc zachowuje drobne gwiazdy i szczegóły przy powiększeniu.

Region nieba wymaga PNG i layoutu NPZ wygenerowanego przez istniejący renderer
regionu:

```python
RUN_MODE = "render"
SCENARIO = "eta_carinae"
BACKGROUND_MODE = "region"
REGION_NAME = "aurora_sky_region_rect_pic1_16k"
REGION_INDEX = 0
BACKGROUND_IMAGE_PATH = None
REGION_LAYOUT_PATH = None
```

Program korzysta z tego samego mechanizmu wykrywania regionów co
`variable_animation` i `microlensing_render`: skanuje `maps/regions`, paruje
PNG z NPZ według metadanych/nazwy, sprawdza zgodność wymiarów i dopiero potem
projektuje położenie progenitora. `REGION_NAME` może być pełną nazwą pliku,
nazwą bez `.png` albo jednoznacznym fragmentem. Gdy pozostaje `None`, wybierany
jest element o numerze `REGION_INDEX` z alfabetycznie uporządkowanej listy.
Program sprawdza też, czy współrzędne progenitora mieszczą się w regionie.

Można nadal wskazać konkretną parę w kodzie:

```python
BACKGROUND_MODE = "region"
BACKGROUND_IMAGE_PATH = Path("maps/regions/moj_region.png")
REGION_LAYOUT_PATH = Path("maps/regions/moj_region_layout.npz")
```

Nie wolno wskazać tylko jednego z tych plików — obraz bez geometrii NPZ nie
jest prawidłowym tłem regionu.

### Jakość filmu 16K

Film pozostaje MP4/HEVC 10-bit z tagiem `hvc1`, zgodnie z rendererami AURORA,
ale moduł supernowej jawnie ustawia `CRF = 10`, preset `slow` i strojenie
zachowujące drobne struktury pola gwiazd. Wcześniej brak jawnego CRF pozostawiał
kodekowi agresywną wartość domyślną, a równoczesne zmniejszenie do 1920 × 960
usuwało większość detalu. Teraz oba źródła rozmycia są wyeliminowane.

`VIDEO_LOSSLESS = True` włącza bezstratny x265 i ignoruje CRF. Ten tryb jest
przeznaczony do materiału pośredniego: pełny film 16K może zajmować bardzo dużo
miejsca. Render 16K również wymaga około 0,4 GiB na sam bufor RGB tła; animator
używa jednego bufora wielokrotnie zamiast alokować nową klatkę dla każdej klatki
filmu.

### Oś czasu wybuchu

Animacja nie zaczyna się już od gotowej supernowej. Domyślnie pokazuje zakres
od `-10` dni przed wybuchem (`t=0`) do `+450` dni po wybuchu:

```python
VIDEO_SIMULATED_DAYS = 450.0
VIDEO_PRE_EXPLOSION_DAYS = 10.0
VIDEO_EXPLOSION_POSITION_FRACTION = 0.18
VIDEO_TIMELINE_ALIGNMENT = "auto"
VIDEO_PLATEAU_THRESHOLD_DAYS = 45.0
VIDEO_PLATEAU_MAGNITUDE_WINDOW = 0.5
```

Oś montażowa ma trzy kotwice i dlatego nie musi płynąć ze stałą liczbą dni na
sekundę: początek odpowiada `-VIDEO_PRE_EXPLOSION_DAYS`, wybuch `t=0` przypada
domyślnie po 18% aktywnej części filmu, a maksimum filtra obserwacyjnego jest
ustawiane w środku filmu. Druga połowa pokazuje opadanie i późny ogon aż do
`VIDEO_SIMULATED_DAYS`.

W trybie `auto` program mierzy szerokość spójnej jasnej fazy w zadanym oknie
magnitudo. Jeżeli przekracza ona `VIDEO_PLATEAU_THRESHOLD_DAYS`, w środku filmu
umieszczany jest środek plateau zamiast pojedynczej próbki maksimum. Zapobiega
to ściskaniu długiego plateau II-P/IIn do kilku klatek. Etykieta nadal pokazuje
rzeczywisty czas obserwatora, również wartości ujemne przed wybuchem.

Własny obraz jest traktowany jak tło pełnego nieba w projekcji Hammera:

```python
RUN_MODE = "render"
SCENARIO = "betelgeuse"
BACKGROUND_MODE = "custom"
BACKGROUND_IMAGE_PATH = Path("/absolutna/sciezka/tlo.png")
```

Własny katalog może być CSV albo FITS. Wymagane kolumny to `l`, `b` w
stopniach galaktycznych. Opcjonalne kolumny `magnitude` i `temperature_k`
sterują jasnością i barwą:

```csv
l,b,magnitude,temperature_k
199.7,-8.9,5.2,4200
202.1,-7.4,8.7,8500
```

```python
RUN_MODE = "render"
SCENARIO = "betelgeuse"
BACKGROUND_MODE = "catalog"
STAR_CATALOG_PATH = Path("moje_gwiazdy.csv")
```

Stary interfejs argumentów nadal istnieje w `cli.py` jako warstwa techniczna,
ale zwykłe uruchamianie modułu idzie przez `run_supernova.py`.

## Parametry wejściowe

Każdy scenariusz zawiera:

- `initial_mass_solar`, `final_mass_solar` — masy początkową i
  przedwybuchową w masach Słońca;
- `metallicity` — ułamek masowy pierwiastków cięższych od helu;
- `radius_solar` — promień przed wybuchem;
- `star_type`, `age_years` — opis ewolucyjny i szacowany wiek;
- `composition` — ułamki masowe sumujące się w przybliżeniu do jedności;
- `total_mass_lost_solar` i `mass_loss_rate_solar_per_year` — całkowitą utratę
  masy i bieżące tempo wiatru; to drugie zasila model CSM typu IIn;
- `distance.value/unit/redshift` — odległość jasnościową i opcjonalnie
  zmierzony redshift;
- `supernova_type`, `extinction_av_mag` oraz pozycję galaktyczną.

W `model_overrides` można jawnie skalibrować energię w foe, masę `Ni-56`, masę
pozostałości, nieprzezroczystości, czas dyfuzji, siłę plateau, wydajność CSM i
podłogę temperatury. Brak pola oznacza wyprowadzenie wartości z progenitora i
domyślnej rodziny supernowej.

Walidacja odrzuca m.in. masy i odległości niedodatnie, `Ni-56` cięższy od
wyrzutu, niezgodne ułamki składu, niewodorowy model Type II, bogaty w wodór
model Ib/Ic, niepoprawne współrzędne oraz prędkość przekraczającą zakres
nierelatywistycznego przybliżenia.

## Równania i założenia

### Eksplozja i dyfuzja

Masa wyrzutu jest różnicą masy końcowej i przyjętej masy zwartej pozostałości.
Dla Ia cały biały karzeł stanowi wyrzut. Prędkość charakterystyczna wynika z
homologicznej kuli o jednorodnej gęstości:

```text
v = sqrt(10 E / 3 M_ej).
```

Skala dyfuzji Arnetta ma postać:

```text
t_d = sqrt(2 κ M_ej / β c v),   β = 13.8.
```

To jedna efektywna nieprzezroczystość, nie pełny transport wieloczęstościowy.

### Ni-56 → Co-56 → Fe-56

Chwilowe ogrzewanie na jednostkę początkowej masy niklu jest liczone jako:

```text
q(t) = ε_Ni exp(-t/τ_Ni)
     + ε_Co [exp(-t/τ_Co) - exp(-t/τ_Ni)],
```

gdzie `τ_Ni = 8.8 d`, `τ_Co = 111.3 d`, `ε_Ni = 3.90e6 W/kg`, a
`ε_Co = 6.78e5 W/kg`. Ucieczkę gamma przybliża:

```text
f_dep = 1 - exp[-(t_gamma/t)^2].
```

Wydostawanie się energii przez rozszerzający wyrzut jest reprezentowane przez
czynnik `1 - exp[-(t/t_d)^2]`. Dzięki temu krzywa ma narastanie, maksimum i
późny ogon zdominowany przez Co-56.

### Rodziny krzywych

| Typ | Dodatkowy składnik modelu |
|---|---|
| II-P | Skala plateau Popova z gładkim przejściem do ogona radioaktywnego. |
| II-L | Eksponencjalnie opadająca emisja rozległej otoczki wodorowej. |
| IIn | Moc szoku CSM `~ 0.5 ε (Mdot/v_w) v_sh^3`, później prawo potęgowe. |
| Ib | Krótkie chłodzenie zwartej, pozbawionej wodoru otoczki + radioaktywność. |
| Ic | Jeszcze słabsza otoczka i typowo większa prędkość/Ni niż Ib. |
| Ia | Osobny model referencyjny zdominowany przez dyfuzję energii radioaktywnej. |

Wczesne chłodzenie ma skalowanie z `R`, `E` i `M_ej`. Nie modeluje osobno
relatywistycznego błysku UV/X-ray podczas wyjścia fali uderzeniowej.

### Promień, temperatura i obserwator

Promień fotosferyczny rośnie w przybliżeniu jak `R0 + v_ph t`, z łagodnym
spadkiem efektywnej prędkości i późną recesją fotosfery w przestrzeni masowej.
Temperatura wynika z prawa Stefana–Boltzmanna:

```text
T = [L / (4 π σ R_ph^2)]^(1/4),
```

z ewoluującą podłogą rekombinacyjną. Temperatura obserwowana ma dodatkowy
czynnik `1/(1+z)`, a czas obserwatora `t_obs = (1+z)t_rest`.

Odległość podana przez użytkownika jest traktowana jako odległość jasnościowa:

```text
F_obs = L / (4 π D_L^2) × 10^(-0.4 A_bol),
A_bol ≈ 0.85 A_V,
M_bol = 4.74 - 2.5 log10(L/L_sun),
m_bol = M_bol + 5 log10(D_L/10 pc) + A_bol.
```

Dla odległości poniżej 10 Mpc domyślnie `z=0`, bo prędkości własne dominują
nad prostym prawem Hubble'a. Dla większych odległości kod numerycznie odwraca
`D_L(z)` w płaskiej kosmologii `H0=67.66 km/s/Mpc`, `Ωm=0.3111`; jawne
`redshift` ma pierwszeństwo. Odległość jasnościowa zawiera już utratę energii
fotonów i dylatację częstości ich przybywania, więc do strumienia nie jest
dodawany drugi czynnik `1+z`.

## Fotometria pasmowa i ekstynkcja

Bolometria i pasma obserwacyjne są osobnymi produktami. Widmo fotosfery jest
przybliżane ciałem doskonale czarnym, a edukacyjne filtry są szerokimi
prostokątnymi pasmami: UV, U, B, V, R, I i IR. Kod całkuje widmo w paśmie,
dzieli przez szerokość częstotliwości i wyznacza magnitudo AB względem 3631 Jy.
Nie jest to pełna fotometria syntetyczna konkretnego instrumentu: brak linii,
line blanketing, dokładnych krzywych przepuszczalności i pełnej poprawki K.

Pył stosuje `A_lambda = (A_lambda/A_V) A_V`. Przyjęte współczynniki dla
`R_V≈3.1` są zapisane wraz z granicami pasm w metadanych JSON. Dla każdego
filtra CSV zawiera strumień przed ekstynkcją, strumień obserwowany i magnitudo.
Wykres pokazuje obserwowaną krzywą V linią ciągłą, a wariant bez pyłu linią
przerywaną.

Shock breakout jest oddzielnym, krótkim składnikiem o własnej temperaturze,
jasności i czasie trwania. Jest eksportowany w bolometrii oraz dodawany do
edukacyjnej krzywej UV, ale nie jest automatycznie zamieniany na klasyczną
jasność V. Opcjonalne echo pyłowe ma niezależne opóźnienie, szerokość i
współczynnik odbicia; domyślnie jest wyłączone albo bardzo słabe.

## Animacja

Wspólna oś `build_editorial_timeline` daje krótki podgląd tła, jedną aktywną
fazę i wygaszenie. Renderer nie normalizuje już obiektu do jego własnego
maksimum. `point_source_intensity` wynika z obserwowanego strumienia w wybranym
filtrze przez `visual_flux_scale = 10^(-0.4 m)`. Nieliniowa odpowiedź ekranu
symuluje skończoną ekspozycję, lecz nie zmienia relacji odległości ani
ekstynkcji. Mały PSF punktu ma stały rozmiar kątowo-ekranowy i skaluje się tylko
z rozdzielczością filmu. Mieszanie typu screen zamiast dodawania z clippingiem
chroni tło LMC.

Model przechowuje fizyczny `angular_shell_radius` w sekundach łuku, liczony z
`R_ej = R0 + v t` i odległości. Jest on nierozdzielczy na mapie całego nieba.
Osobne `halo_radius` i ekranowy promień powłoki są małymi, ograniczonymi
elementami ilustracyjnymi. `point_source_intensity`, `halo_intensity` i emisja
powłoki są niezależne; włączenie powłoki nie zmienia fotometrii punktu.

Każda klatka podaje filtr, magnitudo pozorne, czas, typ i status. Legenda
oznacza punkt jako wynik modelu obserwowanego, a powłokę/halo jako ilustrację.

## SN1987A a scenariusze przyszłe

`historical/sn1987a.json` ma status `historical_observation`. Energia, masa wyrzutu, promień i
`Ni-56` są jawnie skalibrowane do opublikowanych analiz SN1987A, w tym do
kompaktowego niebieskiego nadolbrzyma. Nadal jest to przybliżenie i nie zastąpi
hydrodynamiki STELLA/SEDONA ani danych fotometrycznych.

Katalog historyczny zawiera SN 1006, SN 1054, SN 1181, SN 1572, SN 1604,
SN 1885A i SN 1987A. Pola historyczne przechowują rok, lokalizację, zakres
jasności, odległość, identyfikację pozostałości, obserwatorów oraz jawny poziom
niepewności. Dane historyczne są zakresem, nie sztucznie dokładnym pomiarem.

Betelgeza, Eta Carinae, Antares i R136a1 mają status `hypothetical`, widoczny
komunikat ostrzegawczy i rozkłady niepewności. Monte Carlo losuje tylko zakres
fizycznie plausybilnych wejść. Nie jest rozkładem prawdopodobieństwa daty
wybuchu ani wiarygodnym posterior distribution wszystkich możliwych wyników.
Eta Carinae wymaga szczególnej ostrożności: binarność, geometria gęstego CSM i
historia wielkich erupcji łamią sferyczne założenia modelu.

Każdy scenariusz hipotetyczny ma wariant `pessimistic`, `nominal` i
`optimistic`, zakres odległości i jasności oraz rozkłady Monte Carlo. Żaden z
nich nie modeluje daty wybuchu.

## Monte Carlo i porównania

Monte Carlo zapisuje CSV i JSON oraz SVG z pasem P5–P95, medianą i nominalnym
modelem. Metadane zawierają percentyle czasu i jasności maksimum. Tryb
`compare` zestawia scenariusze `historical`, `hypothetical`, `all` albo jawnie
podaną listę. Dane porównawcze zawierają jasność pozorną i absolutną, strumień,
temperaturę, promień fotosfery, energię, typ, odległość i filtr.

Punkty kalibracyjne oparto m.in. na analizach
[SN1987A (Utrobin)](https://arxiv.org/abs/astro-ph/9911205),
[modelach progenitora SN1987A po mergerze (Menon et al.)](https://academic.oup.com/mnras/article/482/1/438/5114585),
[modelach Betelgezy (Joyce et al.)](https://arxiv.org/abs/2006.09837) i
[modelu wiatru Eta Carinae (Hamaguchi et al.)](https://arxiv.org/abs/1603.01629).
Katalog historyczny opiera zakresy na
[przeglądzie Green & Stephenson](https://arxiv.org/abs/astro-ph/0301603),
a kalibracja SN1987A korzysta również z
[krzywej wizualnej AAVSO](https://www.aavso.org/vsots_sn1987a) i
[geometrycznej odległości 51,4±1,2 kpc](https://arxiv.org/abs/astro-ph/0309416).
Skala bolometryczna zachowuje zerowanie zgodne z
[rezolucją IAU 2015 B2](https://arxiv.org/abs/1510.06262).

## Ograniczenia

- Jedna strefa, symetria sferyczna i homologiczna ekspansja.
- Stałe szare nieprzezroczystości zamiast transportu zależnego od długości fali.
- Brak mieszania przestrzennego niklu, asymetrii, jetów, dust formation,
  magnetara, fallbacku, neutrin i szczegółowej nukleosyntezy.
- Prosty CSM typu IIn nie opisuje pierścieni, dysków ani układu podwójnego.
- Krzywe pasmowe są modelem czarnego ciała z prostokątnymi filtrami, a nie
  pełnym transportem widmowym ani rekonstrukcją konkretnego instrumentu.
- Dla ekstremalnie bliskich obiektów prawo odwrotności kwadratu nadal działa,
  lecz nasycenie oka/kamery jest problemem renderowania, nie fizyki strumienia.
- Dla wysokiego redshift ewolucja kosmologiczna, soczewkowanie i pochłanianie
  międzygalaktyczne nie są uwzględniane.

## Dodawanie nowego typu supernowej

1. Dodaj nazwę do `SUPPORTED_SUPERNOVA_TYPES` w `progenitor.py`.
2. Dodaj energię, `Ni-56`, nieprzezroczystość i pozostałość do
   `_TYPE_DEFAULTS` w `explosion.py`.
3. Dodaj składnik otoczki/CSM w `_envelope_components` oraz ewentualną podłogę
   temperatury w `light_curve.py`.
4. Dodaj scenariusz JSON z jawnym statusem i ograniczeniami.
5. Rozszerz test `test_all_required_types_produce_positive_curves` o nową
   rodzinę oraz dodaj test cechy charakterystycznej jej krzywej.

## Testy

Testy nie wymagają renderowania wideo:

```bash
python -m unittest discover -s supernova/tests -v
```

Sprawdzają m.in. `V_peak(SN1987A)≈3 mag`, odległość około 50 kpc, moduł
odległości, osłabienie przez pył, rozdział bolometrii i V, breakout UV,
rozdzielenie katalogów, zawieranie modelu nominalnego przez Monte Carlo,
niezależność punktu od powłoki, brak clippingu i zgodność jasności między
rozdzielczościami.

## Granica między fizyką i ilustracją

Fizyczne: bolometryczna moc źródła, rozpad Ni/Co, dyfuzja, temperatura,
fotosfera, pasma, ekstynkcja, odległość, strumień, magnitudo, czas obserwatora,
prędkość i rzeczywisty promień kątowy wyrzutu.

Ilustracyjne: ekranowy promień halo i powłoki, ich bardzo słaba emisja,
kolorystyczna prezentacja temperatury, etykiety oraz tone mapping ekranu.
Elementy ilustracyjne są ograniczone przestrzennie i nie zmieniają
`apparent_magnitude`, `bolometric_magnitude` ani `point_source_intensity`.
