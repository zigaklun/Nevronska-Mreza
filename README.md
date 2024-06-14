# Nevronska mreža za prepoznavanje ročno napisanih številk

Ta repozitorij vsebuje nevronsko mrežo, implementirano v Pythonu. Nevronska mreža je zasnovana za prepoznavanje ročno napisanih številk, pri čemer kot vhodne podatke sprejema `.txt` datoteke. Mreža je zgrajena brez uporabe visokorazrednih knjižnic, kot sta TensorFlow ali PyTorch, kar omogoča globlje razumevanje osnovnih mehanizmov nevronskih mrež.

## Kazalo

- [Uvod](#uvod)
- [Struktura projekta](#struktura-projekta)
- [Namestitev](#namestitev)
- [Uporaba](#uporaba)
- [Avtorja](#avtorja)
## Uvod

Ta projekt je namenjen učenju in raziskovanju osnov nevronskih mrež. Implementacija vključuje:
- Inicializacijo mreže s poljubnim številom plasti in nevronov.
- Aktivacijska funkcija, kot je sigmoid.
- Metode za naprej in nazaj širjenje (forward and backward propagation).
- Algoritem za posodabljanje uteži in popravkov.

## Struktura projekta

- `main.py`: Glavna datoteka s kodo za zagon in treniranje nevronske mreže.
- `shranjevanje`: Mapa v katero se shranjujejo podatki o utežeh in popravkih.
- `mnist_test.txt`: Datoteka z podatkih o ročno napisanih številkah 

## Namestitev

1. Klonirajte repozitorij:
    ```sh
    git clone https://github.com/zigaklun/Nevronska-Mreza.git
    ```
2. Premaknite se v direktorij projekta:
    ```sh
    cd Nevronska-Mreza
    ```
3. Namestite potrebne knjižnice:
    ```sh
    pip install -r requirements.txt
    ```

## Uporaba

1. Zaženite glavno datoteko za trening:
    ```sh
    python main.py
    ```
2. Med treningom bodo v konzoli prikazani napredki. Model bo periodično shranjen v mapo `shranjevanje/`.
> Pozor: nevronska mreža še ni trenirana zato bo najverjetneje kazala napačne rezultate. 

## Avtorja
Ta projekt sta razvila Žiga Klun in Vid Gantar.
