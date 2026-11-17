# kangaroo - CPU + GPU (CUDA) - v2 "haute vitesse"

La logique de résolution (`add_dp` / `collide` / `propagate` / `try_key`, table des DP, sauvegarde tame)
est **inchangée** dans `kangaroo.cpp`. Le GPU ne fait que produire des points distingués (DP) ;
seul le moteur de calcul GPU a été réécrit.

## Ce qui rend le GPU plus rapide

| # | Goulot de la v1 | Correction v2 |
|---|---|---|
| 1 | Multiplication 256 bits en `unsigned __int128` (émulé : plusieurs centaines d'instructions) | Limbes 32 bits + chaînes de retenue PTX (`mad.lo.cc` / `madc.hi.cc`), **carré dédié** (36 produits au lieu de 64), réduction spéciale p = 2²⁵⁶ − 2³² − 977 |
| 2 | Table de sauts en `__constant__` lue avec un indice différent par thread : le cache constant **sérialise** ces lectures | Table en mémoire globale lue via `__ldg` (cache L1) |
| 3 | Lot Montgomery limité à 16 (inversion de 255 carrés amortie sur 16 sauts) ; tableaux dimensionnés pour 32 | `-gbatch` = paramètre de template (8/16/**32**/64) ; à 64 il reste ≈ 9 multiplications-équivalent/saut contre ≈ 23 pour la v1 |
| 4 | Historique de cycle en 64 bits | 32 bits (moitié moins de trafic mémoire locale) |
| 5 | 1024 blocs fixes (≈ 100 « vagues » successives) ; le GPU attend le CPU entre deux lancements | Nombre de blocs **automatique** (une vague exacte via l'occupation réelle) ; lancement **asynchrone** : le GPU calcule le lot n+1 pendant que le CPU insère les DP du lot n |
| 6 | Besoin de CUDA ≥ 11.5 (`__int128` device) | Plus de `__int128` |

Gain attendu : de l'ordre de **×5 à ×15** sur la vitesse en sauts/s selon la carte. **Je n'ai pas de GPU ici : ces
chiffres ne sont pas mesurés.** Mesurez avec l'ancienne et la nouvelle version (voir plus bas).

## Ce qui a été vérifié (sans GPU)

- L'arithmétique (mul, carré, add, sub, neg, mod n, inversion : 24 000 comparaisons) est identique à Python.
- Le noyau complet (init, boucle, cycles, DP, multi-clés, tame/wild, `-mix`, lots 8/16/64, `-nosym`) tourne en
  émulation CPU avec `kangaroo.cpp` et retrouve les clés ; efficacité ≈ 1,7 × √W en moyenne, comme le CPU.
- **Non vérifié : la compilation `nvcc` réelle et la vitesse.** Pour couvrir le risque PTX, un **auto-test GPU**
  s'exécute à chaque démarrage de carte (mul/carré/add/sub/inv/courbe contre une implémentation de référence).
  S'il échoue, le programme le dit et s'arrête pour cette carte : recompilez avec `make portable`.

## Compilation

```bash
make            # CPU+GPU si nvcc présent (sm_75 = T4)
make kaggle     # idem
make info       # affiche registres/spills par noyau (utile pour régler -gbatch)
make portable   # sans asm PTX, si l'auto-test signale un écart
make native     # -arch=native (CUDA >= 11.6)
make GENCODE="-gencode arch=compute_80,code=sm_80"   # autre carte (A100) ; 86 = RTX30, 89 = RTX40
```

## Mesurer et régler

```bash
# vitesse pure (génération tame, sans clé) : lisez "Vitesse sauts" après quelques lancements
./kangaroo -bb 100 -s /tmp/t.dat -duree 40 -c 0 -gbatch 32
./kangaroo -bb 100 -s /tmp/t.dat -duree 40 -c 0 -gbatch 64
./kangaroo -bb 100 -s /tmp/t.dat -duree 40 -c 0 -gbatch 16
```
Gardez le `-gbatch` le plus rapide. Si le noyau manque de mémoire (petite carte) : `-gbatch 16`.

Test de bon fonctionnement (résout en quelques secondes) :
```bash
PUB=$(./kangaroo -genpub 1a2b3c4d5e)
./kangaroo -pub $PUB -bb 40 -gpu all -c 0 -t 5
```

## Options GPU

| Option | Rôle | Défaut |
|---|---|---|
| `-gpu <spec>` | `all`, `off`, ou liste `0,1` | `all` |
| `-gblocks n` | blocs CUDA par carte ; **0 = automatique** (une vague de blocs résidents) | 0 |
| `-gthreads n` | threads par bloc (sert au calcul du nombre de kangourous ; le noyau lance des blocs de 128) | 128 |
| `-gbatch n` | kangourous par thread = lot Montgomery : 8, 16, 32 ou 64 | 32 |
| `-giter n` | itérations par lancement (borne auto si `-dp` est petit, pour ne pas perdre de DP) | 128 |
| `-c 0` | pas de threads CPU (GPU seul) | - |

Pour un **petit intervalle** (≲ 2⁵⁰), le nombre de kangourous est plafonné automatiquement (trop de kangourous
retarde la détection des collisions quand √W est petit). Pour les grands intervalles (≥ 2⁶⁶), il reste
à une vague complète.

## Limites connues

- Détection des collisions par lots : la granularité d'un lancement (quelques dizaines à centaines de ms sur grand
  problème) est négligeable ; sur des tests minuscules (bb ≈ 30-40) elle fausse les compteurs « × √W ».
- Si le tampon de DP d'un lancement est saturé (`-dp` très petit), les DP en excès sont perdus (le programme
  réduit `-giter` automatiquement pour l'éviter).
- `-mix`/`-end` : la transition tame → wild a la granularité d'un lancement.
- Mémoire : ≈ 192 octets d'état par kangourou + tableaux locaux (≈ 250 octets × `-gbatch` par thread résident).
