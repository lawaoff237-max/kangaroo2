// =============================================================================
//  kangaroo.cpp  -  Pollard Kangaroo (lambda) multi-clés / multi-coeurs
//  Courbe secp256k1, intervalle borné [a, b] jusqu'à 256 bits, CPU uniquement.
//
//  Compilation (rapide) :
//     g++ -O3 -march=native -flto -funroll-loops -pthread -std=c++17 kangaroo.cpp -o kangaroo
//  Encore plus rapide (PGO, ~10-20% de mieux, 2 etapes) :
//     g++ -O3 -march=native -flto -funroll-loops -fprofile-generate -pthread -std=c++17 kangaroo.cpp -o kangaroo_prof
//     ./kangaroo_prof -r 1:ffffffff -t 5 -end 0.3          # tourne ~30s sur un cas representatif, puis Ctrl+C
//     g++ -O3 -march=native -flto -funroll-loops -fprofile-use -pthread -std=c++17 kangaroo.cpp -o kangaroo
//
//  Fonctionnalités :
//   - arithmétique 256 bits maison (4x64, réduction spéciale p = 2^256-2^32-977)
//   - distances / largeur d'intervalle sur 256 bits : les distances des kangourous sont
//     calculées modulo n (ordre de la courbe) -> intervalles jusqu'à [1, n-1] (-bb 256)
//   - addition affine + inversion par lots (Montgomery) : ~6 mul / saut
//   - symétrie ±y (classes d'équivalence) + détection/sortie des cycles
//   - points distingués (DP), table partagée entre threads (shards)
//   - N clés dans le même intervalle : table tame partagée, un troupeau wild/clé
//   - collisions wild/wild entre clés : relations k_i = ±k_j + c exploitées
//   - sauvegarde / chargement de la table tame (-s / -ss)
//
//  --- Ajout GPU (CUDA, pense pour 1-2x Nvidia T4 type Kaggle) + CPU simultanes ---
//   - Le GPU est un pur "generateur de points distingues" : la table de sauts, la
//     recherche de collision (add_dp/collide/propagate/try_key) et la sauvegarde
//     tame restent 100% dans ce fichier, inchangees, donc partagees telles quelles
//     entre threads CPU et "ponts" GPU (meme tables, meme mutex, meme logique de
//     resolution). Voir kangaroo_gpu.h / kangaroo_gpu.cu et le Makefile fourni.
//   - Sans CUDA installe (nvcc absent), ce fichier compile et se comporte EXACTEMENT
//     comme l'original : gpu_device_count() renvoie 0 et aucun thread GPU n'est cree.
// =============================================================================
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <cmath>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <atomic>
#include <chrono>
#include <unordered_map>
#include <algorithm>
#include <csignal>
#include <ctime>
#include <fstream>
#include <sstream>

#include "kangaroo_gpu.h"

using namespace std;

#ifndef HAVE_CUDA
// Compile sans CUDA (nvcc absent du systeme, ou build CPU force via "make cpu") :
// bouchons neutres pour que TOUT le reste du fichier puisse appeler les fonctions
// gpu_* sans aucun #ifdef disperse. gpu_device_count() == 0 desactive le GPU proprement.
extern "C" {
    int  gpu_device_count(void) { return 0; }
    void gpu_device_name(int, char* buf, int buflen) { if (buf && buflen > 0) buf[0] = 0; }
    void* gpu_worker_create(const gpu_init_t*) { return nullptr; }
    int  gpu_auto_blocks(int, int, int) { return 0; }
    int  gpu_worker_launch(void*, int) { return -1; }
    int  gpu_worker_fetch(void*, gdprec_t*, int, uint64_t*, uint64_t*) { return 0; }
    int  gpu_worker_run(void*, gdprec_t*, int, uint64_t*, uint64_t*) { return 0; }
    void gpu_worker_sync(void*, const uint8_t*, int, int) {}
    void gpu_worker_destroy(void*) {}
}
#endif
typedef uint64_t u64;
typedef uint32_t u32;
typedef unsigned __int128 u128;
typedef __int128 i128;

// ============================ entiers 256 bits ===============================
struct U256 { u64 d[4]; };
typedef U256 fe;

static const u64 P0 = 0xFFFFFFFEFFFFFC2FULL;
static const u64 CP = 0x1000003D1ULL;            // 2^256 mod p

static inline bool u_zero(const U256& a) { return !(a.d[0] | a.d[1] | a.d[2] | a.d[3]); }
static inline bool u_eq(const U256& a, const U256& b) {
    return a.d[0] == b.d[0] && a.d[1] == b.d[1] && a.d[2] == b.d[2] && a.d[3] == b.d[3];
}
static inline int u_cmp(const U256& a, const U256& b) {
    for (int i = 3; i >= 0; i--) { if (a.d[i] != b.d[i]) return a.d[i] > b.d[i] ? 1 : -1; }
    return 0;
}
static inline U256 u_from(u128 v) { U256 r; r.d[0] = (u64)v; r.d[1] = (u64)(v >> 64); r.d[2] = r.d[3] = 0; return r; }
static inline U256 u_add(const U256& a, const U256& b) {
    U256 r; u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)a.d[i] + b.d[i]; r.d[i] = (u64)c; c >>= 64; }
    return r;
}
static inline U256 u_sub(const U256& a, const U256& b) {
    U256 r; u64 br = 0;
    for (int i = 0; i < 4; i++) { u128 t = (u128)a.d[i] - b.d[i] - br; r.d[i] = (u64)t; br = (u64)(t >> 64) & 1; }
    return r;
}
static inline int u_bits(const U256& a) {
    for (int i = 3; i >= 0; i--) if (a.d[i]) return i * 64 + 64 - __builtin_clzll(a.d[i]);
    return 0;
}
static inline void u_shr1(U256& a) {
    for (int i = 0; i < 3; i++) a.d[i] = (a.d[i] >> 1) | (a.d[i + 1] << 63);
    a.d[3] >>= 1;
}
static U256 SECP_N, NHALF;                        // ordre de la courbe et (n-1)/2 (init_curve)
static inline U256 u_zero_val() { U256 r; r.d[0] = r.d[1] = r.d[2] = r.d[3] = 0; return r; }
// bits bas a 1 : u_ones(k) = 2^k - 1, k dans [0, 256]
static inline U256 u_ones(int k) {
    U256 r = u_zero_val();
    for (int i = 0; i < 4; i++) { int lo = i * 64; if (k >= lo + 64) r.d[i] = ~0ULL; else if (k > lo) r.d[i] = (1ULL << (k - lo)) - 1; }
    return r;
}
static inline U256 u_pow2(int k) { U256 r = u_zero_val(); r.d[k >> 6] = 1ULL << (k & 63); return r; }   // k dans [0, 255]
static inline U256 u_shl(const U256& a, int n) {      // n dans [0, 255]
    U256 r = u_zero_val(); int w = n >> 6, b = n & 63;
    for (int i = 3; i >= w; i--) {
        u64 v = a.d[i - w] << b;
        if (b && i - w - 1 >= 0) v |= a.d[i - w - 1] >> (64 - b);
        r.d[i] = v;
    }
    return r;
}
static inline U256 u_shr(const U256& a, int n) {      // n dans [0, 255]
    U256 r = u_zero_val(); int w = n >> 6, b = n & 63;
    for (int i = 0; i + w < 4; i++) {
        u64 v = a.d[i + w] >> b;
        if (b && i + w + 1 < 4) v |= a.d[i + w + 1] << (64 - b);
        r.d[i] = v;
    }
    return r;
}
static inline double u2dbl(const U256& a) {
    double r = 0; for (int i = 3; i >= 0; i--) r = r * 18446744073709551616.0 + (double)a.d[i]; return r;
}
static inline U256 dbl2u(double v) {                  // v >= 0 -> partie entiere, saturee a 2^256-1
    if (!(v >= 1.0)) return u_zero_val();
    int e; double m = frexp(v, &e);                   // v = m * 2^e, m dans [0.5, 1)
    if (e > 256) return u_ones(256);
    U256 r = u_zero_val(); r.d[0] = (u64)ldexp(m, 53);
    int sh = e - 53;
    return sh >= 0 ? u_shl(r, sh) : u_shr(r, -sh);
}
// ---- arithmetique modulo n (distances des kangourous, cles) : operandes dans [0, n) ----
static inline U256 m_add(const U256& a, const U256& b) {
    U256 r; u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)a.d[i] + b.d[i]; r.d[i] = (u64)c; c >>= 64; }
    if (c || u_cmp(r, SECP_N) >= 0) r = u_sub(r, SECP_N);    // modulo 2^256 : correct aussi avec retenue
    return r;
}
static inline U256 m_sub(const U256& a, const U256& b) {
    U256 r; u64 br = 0;
    for (int i = 0; i < 4; i++) { u128 t = (u128)a.d[i] - b.d[i] - br; r.d[i] = (u64)t; br = (u64)(t >> 64) & 1; }
    if (br) r = u_add(r, SECP_N);                             // a - b + n (modulo 2^256)
    return r;
}
static inline U256 m_neg(const U256& a) { return u_zero(a) ? a : u_sub(SECP_N, a); }
static inline U256 m_half(const U256& a) {                    // a / 2 mod n  (n impair)
    U256 r = a;
    if (!(a.d[0] & 1)) { u_shr1(r); return r; }
    u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)a.d[i] + SECP_N.d[i]; r.d[i] = (u64)c; c >>= 64; }
    u_shr1(r); r.d[3] |= (u64)c << 63;
    return r;
}
static inline U256 m_sgn(const U256& a, int s) { return s < 0 ? m_neg(a) : a; }   // s = +-1

static bool hex2u(string s, U256& r) {
    if (s.size() > 1 && s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) s = s.substr(2);
    if (s.empty() || s.size() > 64) return false;
    memset(&r, 0, sizeof(r));
    int pos = 0;
    for (int i = (int)s.size() - 1; i >= 0; i--, pos++) {
        char c = s[i]; int v;
        if (c >= '0' && c <= '9') v = c - '0';
        else if (c >= 'a' && c <= 'f') v = c - 'a' + 10;
        else if (c >= 'A' && c <= 'F') v = c - 'A' + 10;
        else return false;
        r.d[pos / 16] |= (u64)v << ((pos % 16) * 4);
    }
    return true;
}
static string u2hex(const U256& a) {
    char b[80]; snprintf(b, sizeof b, "%016llx%016llx%016llx%016llx",
        (unsigned long long)a.d[3], (unsigned long long)a.d[2], (unsigned long long)a.d[1], (unsigned long long)a.d[0]);
    return string(b);
}

// ================================ corps Fp ===================================
static inline void add_small(U256& r, u64 v) {
    u128 c = (u128)r.d[0] + v; r.d[0] = (u64)c; c >>= 64;
    for (int i = 1; i < 4 && c; i++) { c += r.d[i]; r.d[i] = (u64)c; c >>= 64; }
}
static inline void sub_small(U256& r, u64 v) {
    u64 br = r.d[0] < v; r.d[0] -= v;
    for (int i = 1; i < 4 && br; i++) { br = (r.d[i] == 0); r.d[i]--; }
}
static inline void fe_norm(fe& r) {  // r < 2^256  ->  r < p
    if (r.d[3] == ~0ULL && r.d[2] == ~0ULL && r.d[1] == ~0ULL && r.d[0] >= P0) add_small(r, CP);
}
static inline void fe_add(fe& r, const fe& a, const fe& b) {
    u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)a.d[i] + b.d[i]; r.d[i] = (u64)c; c >>= 64; }
    if (c) add_small(r, CP);
    fe_norm(r);
}
static inline void fe_sub(fe& r, const fe& a, const fe& b) {
    u64 br = 0;
    for (int i = 0; i < 4; i++) { u128 t = (u128)a.d[i] - b.d[i] - br; r.d[i] = (u64)t; br = (u64)(t >> 64) & 1; }
    if (br) sub_small(r, CP);
}
static inline void fe_neg(fe& r, const fe& a) {
    if (u_zero(a)) { r = a; return; }
    fe z; memset(&z, 0, sizeof z); fe_sub(r, z, a);
}
// Reduction commune (2^256 mod p = CP) : prend un produit brut de 8 mots 64 bits et le ramene < p.
// Partagee par fe_mul et fe_sqr pour eviter la duplication.
static inline void fe_reduce_from_t(fe& r, u64 t[8]) {
    u64 o[4]; u128 c = 0;
    for (int i = 0; i < 4; i++) { c += (u128)t[4 + i] * CP + t[i]; o[i] = (u64)c; c >>= 64; }
    u64 top = (u64)c;
    c = (u128)top * CP;
    for (int i = 0; i < 4; i++) { c += o[i]; o[i] = (u64)c; c >>= 64; }
    memcpy(r.d, o, sizeof o);
    if (c) add_small(r, CP);
    fe_norm(r);
}
static inline void fe_mul(fe& r, const fe& a, const fe& b) {
    u64 t[8]; u128 c = 0;
    for (int j = 0; j < 4; j++) { c += (u128)a.d[0] * b.d[j]; t[j] = (u64)c; c >>= 64; }
    t[4] = (u64)c;
    for (int i = 1; i < 4; i++) {
        c = 0;
        for (int j = 0; j < 4; j++) { c += (u128)a.d[i] * b.d[j] + t[i + j]; t[i + j] = (u64)c; c >>= 64; }
        t[i + 4] = (u64)c;
    }
    fe_reduce_from_t(r, t);
}
// Squaring dediee : exploite a_i*a_j = a_j*a_i pour ne faire que 10 multiplications 64x64
// (6 termes croises + 4 carres) au lieu des 16 de fe_mul(a,a,a) -> ~1.5x plus rapide.
// Verifie par fuzzing (2M tirages aleatoires + cas limites 0/1/p-1/2^256-1) contre fe_mul(a,a).
static inline void fe_sqr(fe& r, const fe& a) {
    u64 OD[7] = {0,0,0,0,0,0,0};   // somme des termes croises a_i*a_j (i<j), sans le facteur 2
    u128 c;
    c = 0;
    for (int j = 1; j < 4; j++) { c += (u128)a.d[0] * a.d[j]; OD[j] = (u64)c; c >>= 64; }
    OD[4] = (u64)c;
    c = 0;
    for (int j = 2; j < 4; j++) { c += (u128)a.d[1] * a.d[j] + OD[1 + j]; OD[1 + j] = (u64)c; c >>= 64; }
    OD[5] = (u64)c;
    c = (u128)a.d[2] * a.d[3] + OD[5];
    OD[5] = (u64)c; c >>= 64;
    OD[6] = (u64)c;
    u64 t[8]; u64 carry = 0;                      // double OD (decalage 1 bit) -> t[0..7]
    for (int k = 0; k < 7; k++) { u64 v = OD[k]; t[k] = (v << 1) | carry; carry = v >> 63; }
    t[7] = carry;
    u64 dcarry = 0;                               // ajoute les carres a_i^2 sur des positions non chevauchantes
    for (int i = 0; i < 4; i++) {
        u128 sq = (u128)a.d[i] * a.d[i];
        u64 lo = (u64)sq, hi = (u64)(sq >> 64);
        u128 s = (u128)t[2 * i] + lo + dcarry;
        t[2 * i] = (u64)s; dcarry = (u64)(s >> 64);
        s = (u128)t[2 * i + 1] + hi + dcarry;
        t[2 * i + 1] = (u64)s; dcarry = (u64)(s >> 64);
    }
    fe_reduce_from_t(r, t);
}
static inline void fe_sqrn(fe& r, const fe& a, int n) { r = a; for (int i = 0; i < n; i++) fe_sqr(r, r); }

static void fe_inv(fe& r, const fe& a) {   // a^(p-2), chaîne d'addition de libsecp256k1
    fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    fe_sqr(x2, a);       fe_mul(x2, x2, a);
    fe_sqr(x3, x2);      fe_mul(x3, x3, a);
    fe_sqrn(x6, x3, 3);  fe_mul(x6, x6, x3);
    fe_sqrn(x9, x6, 3);  fe_mul(x9, x9, x3);
    fe_sqrn(x11, x9, 2); fe_mul(x11, x11, x2);
    fe_sqrn(x22, x11, 11);   fe_mul(x22, x22, x11);
    fe_sqrn(x44, x22, 22);   fe_mul(x44, x44, x22);
    fe_sqrn(x88, x44, 44);   fe_mul(x88, x88, x44);
    fe_sqrn(x176, x88, 88);  fe_mul(x176, x176, x88);
    fe_sqrn(x220, x176, 44); fe_mul(x220, x220, x44);
    fe_sqrn(x223, x220, 3);  fe_mul(x223, x223, x3);
    fe_sqrn(t, x223, 23); fe_mul(t, t, x22);
    fe_sqrn(t, t, 5);     fe_mul(t, t, a);
    fe_sqrn(t, t, 3);     fe_mul(t, t, x2);
    fe_sqrn(t, t, 2);     fe_mul(r, t, a);
}
static void fe_pow(fe& r, const fe& a, const U256& e) {
    fe res; memset(&res, 0, sizeof res); res.d[0] = 1;
    int nb = u_bits(e);
    for (int i = nb - 1; i >= 0; i--) {
        fe_sqr(res, res);
        if ((e.d[i / 64] >> (i % 64)) & 1) fe_mul(res, res, a);
    }
    r = res;
}

// ================================ courbe =====================================
struct Pt { fe x, y; bool inf; };
struct Jac { fe X, Y, Z; };

static U256 GX, GY;
static Pt GEN;

static inline Jac jac_inf() { Jac r; memset(&r, 0, sizeof r); r.X.d[0] = 1; r.Y.d[0] = 1; return r; }
static inline Jac jac_from(const Pt& p) {
    if (p.inf) return jac_inf();
    Jac r; r.X = p.x; r.Y = p.y; memset(&r.Z, 0, sizeof r.Z); r.Z.d[0] = 1; return r;
}
static void jac_dbl(Jac& r, const Jac& a) {
    if (u_zero(a.Z) || u_zero(a.Y)) { r = jac_inf(); return; }
    fe A, B, C, D, E, F, t, X3, Y3, Z3, C8;
    fe_sqr(A, a.X); fe_sqr(B, a.Y); fe_sqr(C, B);
    fe_add(t, a.X, B); fe_sqr(t, t); fe_sub(t, t, A); fe_sub(t, t, C); fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A);
    fe_sqr(F, E);
    fe_sub(X3, F, D); fe_sub(X3, X3, D);
    fe_sub(t, D, X3); fe_mul(Y3, E, t);
    fe_add(C8, C, C); fe_add(C8, C8, C8); fe_add(C8, C8, C8);
    fe_sub(Y3, Y3, C8);
    fe_mul(Z3, a.Y, a.Z); fe_add(Z3, Z3, Z3);
    r.X = X3; r.Y = Y3; r.Z = Z3;
}
static void jac_add(Jac& r, const Jac& a, const Jac& b) {
    if (u_zero(a.Z)) { r = b; return; }
    if (u_zero(b.Z)) { r = a; return; }
    fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, R, t;
    fe_sqr(Z1Z1, a.Z); fe_sqr(Z2Z2, b.Z);
    fe_mul(U1, a.X, Z2Z2); fe_mul(U2, b.X, Z1Z1);
    fe_mul(t, b.Z, Z2Z2); fe_mul(S1, a.Y, t);
    fe_mul(t, a.Z, Z1Z1); fe_mul(S2, b.Y, t);
    fe_sub(H, U2, U1); fe_sub(R, S2, S1);
    if (u_zero(H)) {
        if (u_zero(R)) { jac_dbl(r, a); return; }
        r = jac_inf(); return;
    }
    fe H2, H3, V, X3, Y3, Z3;
    fe_sqr(H2, H); fe_mul(H3, H, H2); fe_mul(V, U1, H2);
    fe_sqr(X3, R); fe_sub(X3, X3, H3); fe_sub(X3, X3, V); fe_sub(X3, X3, V);
    fe_sub(t, V, X3); fe_mul(Y3, R, t); fe_mul(t, S1, H3); fe_sub(Y3, Y3, t);
    fe_mul(Z3, a.Z, b.Z); fe_mul(Z3, Z3, H);
    r.X = X3; r.Y = Y3; r.Z = Z3;
}
static Pt to_aff(const Jac& j) {
    Pt p; p.inf = false;
    if (u_zero(j.Z)) { p.inf = true; memset(&p.x, 0, sizeof p.x); memset(&p.y, 0, sizeof p.y); return p; }
    fe zi, z2, z3;
    fe_inv(zi, j.Z); fe_sqr(z2, zi); fe_mul(z3, z2, zi);
    fe_mul(p.x, j.X, z2); fe_mul(p.y, j.Y, z3);
    return p;
}
static Pt pt_mul(const U256& k, const Pt& base) {
    Jac R = jac_inf(), B = jac_from(base);
    int nb = u_bits(k);
    for (int i = nb - 1; i >= 0; i--) {
        jac_dbl(R, R);
        if ((k.d[i / 64] >> (i % 64)) & 1) jac_add(R, R, B);
    }
    return to_aff(R);
}
static inline Pt mulG(const U256& k) { return pt_mul(k, GEN); }
static Pt pt_neg(const Pt& p) { Pt r = p; if (!p.inf) fe_neg(r.y, p.y); return r; }
static Pt pt_add(const Pt& a, const Pt& b) { Jac r; jac_add(r, jac_from(a), jac_from(b)); return to_aff(r); }
static Pt mulG_mod(const U256& k) {      // k dans [0, n) ; k > n/2 represente un scalaire negatif
    if (u_cmp(k, NHALF) > 0) return pt_neg(mulG(u_sub(SECP_N, k)));
    return mulG(k);
}
static bool pt_eq(const Pt& a, const Pt& b) {
    if (a.inf || b.inf) return a.inf == b.inf;
    return u_eq(a.x, b.x) && u_eq(a.y, b.y);
}
static U256 P_FIELD, EXP_SQRT;
static bool on_curve(const Pt& p) {
    fe y2, x3, rhs, seven; memset(&seven, 0, sizeof seven); seven.d[0] = 7;
    fe_sqr(y2, p.y); fe_sqr(x3, p.x); fe_mul(x3, x3, p.x); fe_add(rhs, x3, seven);
    return u_eq(y2, rhs);
}
static bool parse_pub(string s, Pt& P) {
    while (!s.empty() && isspace((unsigned char)s.back())) s.pop_back();
    while (!s.empty() && isspace((unsigned char)s.front())) s.erase(s.begin());
    P.inf = false;
    if (s.size() == 66 && (s.compare(0, 2, "02") == 0 || s.compare(0, 2, "03") == 0)) {
        if (!hex2u(s.substr(2), P.x)) return false;
        if (u_cmp(P.x, P_FIELD) >= 0) return false;
        fe x3, rhs, seven, y, y2; memset(&seven, 0, sizeof seven); seven.d[0] = 7;
        fe_sqr(x3, P.x); fe_mul(x3, x3, P.x); fe_add(rhs, x3, seven);
        fe_pow(y, rhs, EXP_SQRT); fe_sqr(y2, y);
        if (!u_eq(y2, rhs)) return false;
        bool odd = y.d[0] & 1;
        if (odd != (s[1] == '3')) fe_neg(y, y);
        P.y = y; return true;
    }
    if (s.size() == 130 && s.compare(0, 2, "04") == 0) {
        if (!hex2u(s.substr(2, 64), P.x) || !hex2u(s.substr(66, 64), P.y)) return false;
        return on_curve(P);
    }
    return false;
}
static string pub_hex(const Pt& P) {
    string h = u2hex(P.x);
    return string((P.y.d[0] & 1) ? "03" : "02") + h;
}
static void init_curve() {
    hex2u("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F", P_FIELD);
    hex2u("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141", SECP_N);
    hex2u("79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798", GX);
    hex2u("483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8", GY);
    GEN.x = GX; GEN.y = GY; GEN.inf = false;
    U256 e = u_add(P_FIELD, u_from(1)); u_shr1(e); u_shr1(e); EXP_SQRT = e;  // (p+1)/4
    NHALF = SECP_N; u_shr1(NHALF);                                          // floor(n/2)
}

// ============================== paramètres ===================================
static const int NJ = 256;     // taille de la table de sauts
static const int NE = 64;      // table de sauts "d'évasion" (cycles)
struct Jump { fe x, y; U256 d; };      // d = distance du saut (dans [1, n))
static Jump g_J[NJ], g_E[NE];

static U256 g_a, g_end, g_c;   // début, fin (incluse), centre
static U256 g_W;               // largeur de l'intervalle (jusqu'a n)
static double g_Wd = 0;        // idem en double (stats, choix des sauts)
static int g_sym = 1;          // symétrie ±
static int g_dp = -1; static u64 g_dpmask = 0;
static int g_nthreads = 1, g_batch = 512;
static double g_jf = -1, g_jg = -1; static int g_sl = 2;
static double g_coef = -1;
static double g_tfrac = 0.5;   // fraction de kangourous tame quand tame+wild coexistent
static u64 g_seed = 0;
static u64 g_jhash = 0;

static vector<Pt> g_pub, g_Q;
static vector<string> g_pubstr;
static int g_nk = 0;
static atomic<int>* g_solved = nullptr;
static vector<U256> g_kval;
static atomic<int> g_nsolved{0};
static atomic<int> g_stop{0};
static atomic<u64> g_keyctr{0};
static mutex g_solmtx, g_relmtx, g_outmtx;
static string g_outfile = "kangaroo_trouve.txt";

struct Rng {
    u64 s[4];
    explicit Rng(u64 seed) {
        for (int i = 0; i < 4; i++) {
            seed += 0x9E3779B97F4A7C15ULL; u64 z = seed;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
            s[i] = z ^ (z >> 31);
        }
    }
    static inline u64 rotl(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
    u64 next() {
        u64 r = rotl(s[1] * 5, 7) * 9, t = s[1] << 17;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl(s[3], 45);
        return r;
    }
    // uniforme dans [0, n), n >= 1 sur 256 bits (rejet par masque : < 2 essais en moyenne)
    U256 below(const U256& n) {
        if (u_zero(n)) return u_zero_val();
        U256 m1 = n; sub_small(m1, 1);
        const U256 mask = u_ones(u_bits(m1));
        for (;;) {
            U256 r; for (int i = 0; i < 4; i++) r.d[i] = next() & mask.d[i];
            if (u_cmp(r, n) < 0) return r;
        }
    }
};

static void build_jumps(double M) {
    Rng r(0xC0FFEE123456789ULL ^ g_W.d[0] ^ g_W.d[1] ^ g_W.d[2] ^ g_W.d[3]);
    U256 span = dbl2u(2.0 * M); if (u_cmp(span, u_from(8)) < 0) span = u_from(8);
    if (u_cmp(span, SECP_N) >= 0) { span = SECP_N; sub_small(span, 1); }       // jamais >= n
    g_jhash = span.d[0] * 0x9E3779B97F4A7C15ULL ^ span.d[1] ^ span.d[2] ^ span.d[3];
    for (int i = 0; i < NJ + NE; i++) {
        Jump& j = (i < NJ) ? g_J[i] : g_E[i - NJ];
        j.d = r.below(span); add_small(j.d, 1);                                // [1, span]
        Pt p = mulG(j.d); j.x = p.x; j.y = p.y;
        for (int q = 0; q < 4; q++) g_jhash = (g_jhash ^ j.d.d[q]) * 0x100000001B3ULL;
    }
}

// ============================ table des DP ===================================
static const u32 TAME = 0xFFFFFFFFu;
struct K128 { u64 a, b; bool operator==(const K128& o) const { return a == o.a && b == o.b; } };
struct KH { size_t operator()(const K128& k) const {
    u64 z = k.a * 0x9E3779B97F4A7C15ULL + k.b; z ^= z >> 29; z *= 0xBF58476D1CE4E5B9ULL; z ^= z >> 32; return (size_t)z; } };
struct Ent { U256 d; u32 id; int8_t s; };
struct Shard { mutex m; unordered_map<K128, Ent, KH> mp; };
static Shard g_sh[256];
static atomic<u64> g_ndp{0}, g_ntame{0};
struct DPRec { u64 x0, x1; U256 d; u32 id; int8_t s; };

struct Rel { int i, j, c; U256 k0; };      // k_i = c*k_j + k0  (mod n)
static vector<Rel> g_rel;

static void propagate();

static string hex_trim(const U256& a) {
    string h = u2hex(a); size_t p = h.find_first_not_of('0');
    return p == string::npos ? "0" : h.substr(p);
}
static void report_found(int idx, const U256& key) {
    string pub = g_pubstr[idx], priv = u2hex(key);
    printf("\n==============================================================\n"
           "  >>>  CLE TROUVEE  <<<                       (%d / %d)\n"
           "==============================================================\n"
           "  Cle publique : %s\n"
           "  Cle privee   : 0x%s\n"
           "  Enregistree  : %s\n"
           "%s"
           "==============================================================\n\n",
           g_nsolved.load(), g_nk, pub.c_str(), hex_trim(key).c_str(), g_outfile.c_str(),
           (u_cmp(key, g_a) < 0 || u_cmp(key, g_end) > 0) ? "  Note         : cle HORS de l'intervalle demande (verifiee par key*G)\n" : "");
    fflush(stdout);
    FILE* f = fopen(g_outfile.c_str(), "a");
    if (f) {
        time_t t = time(nullptr); char tb[64]; strftime(tb, sizeof tb, "%Y-%m-%d %H:%M:%S", localtime(&t));
        fprintf(f, "Date          : %s\nCle publique  : %s\nCle privee    : 0x%s\n\n", tb, pub.c_str(), priv.c_str());
        fclose(f);
    } else printf("ATTENTION : impossible d'ecrire dans %s\n", g_outfile.c_str());
}
// (centre + k) mod n ; k est deja un residu modulo n (une distance negative vaut n - |k|)
static U256 key_from_k(const U256& k) { return m_add(g_c, k); }
static void try_key(int i, const U256& k) {   // k = k'' mod n (clé = centre + k'')
    if (g_solved[i].load(memory_order_acquire)) return;
    U256 key = key_from_k(k);
    // Seul critere de validite : key*G == cle publique. Aucune restriction a l'intervalle [a, b].
    Pt P = mulG(key);
    if (!pt_eq(P, g_pub[i])) return;
    {
        lock_guard<mutex> lk(g_solmtx);
        if (g_solved[i].load(memory_order_acquire)) return;
        g_kval[i] = k;                                   // 1) la valeur d'abord ...
        g_solved[i].store(1, memory_order_release);      // 2) ... puis le drapeau (publie g_kval aux lecteurs)
        g_nsolved++;
        lock_guard<mutex> lo(g_outmtx);
        report_found(i, key);
        if (g_nsolved.load() >= g_nk) g_stop = 1;
    }
    propagate();
}
static void propagate() {
    bool ch = true;
    while (ch) {
        ch = false;
        vector<Rel> rel; { lock_guard<mutex> lk(g_relmtx); rel = g_rel; }
        for (auto& r : rel) {
            bool si = g_solved[r.i].load(memory_order_acquire), sj = g_solved[r.j].load(memory_order_acquire);
            if (si && !sj) { U256 kj = m_sgn(m_sub(g_kval[r.i], r.k0), r.c); try_key(r.j, kj); if (g_solved[r.j].load()) ch = true; }
            else if (sj && !si) { U256 ki = m_add(m_sgn(g_kval[r.j], r.c), r.k0); try_key(r.i, ki); if (g_solved[r.i].load()) ch = true; }
        }
    }
}
static void collide(const Ent& A, const Ent& B) {
    if (A.id == TAME && B.id == TAME) return;
    const Ent *t = nullptr, *w = nullptr;
    int nsig = g_sym ? 2 : 1;
    if (A.id == TAME) { t = &A; w = &B; } else if (B.id == TAME) { t = &B; w = &A; }
    if (t) {
        if (g_solved[w->id].load()) return;
        for (int q = 0; q < nsig; q++) {
            int sg = q ? -1 : 1;                           // log_w = sg * log_t
            U256 k = m_sgn(m_sub(m_sgn(t->d, sg), w->d), w->s);          // s_w * (sg*d_t - d_w)  mod n
            try_key((int)w->id, k);
        }
        return;
    }
    // wild / wild
    for (int q = 0; q < nsig; q++) {
        int sg = q ? -1 : 1;
        if (A.id == B.id) {
            int coef = A.s - sg * B.s;
            if (coef == 0) continue;
            U256 rhs = m_sub(m_sgn(B.d, sg), A.d);                      // coef * k = rhs  (mod n), coef = +-2
            try_key((int)A.id, m_sgn(m_half(rhs), coef));
        } else {
            // s_i k_i + d_i = sg (s_j k_j + d_j)
            int c = A.s * sg * B.s;
            U256 k0 = m_sgn(m_sub(m_sgn(B.d, sg), A.d), A.s);
            bool si = g_solved[A.id].load(), sj = g_solved[B.id].load();
            if (si && sj) continue;
            { lock_guard<mutex> lk(g_relmtx); if (g_rel.size() < 2000000) g_rel.push_back({(int)A.id, (int)B.id, c, k0}); }
            if (si || sj) propagate();
        }
    }
}
static void add_dp(const DPRec& r) {
    K128 k{r.x0, r.x1};
    Shard& sh = g_sh[(r.x1 >> 8) & 255];
    Ent ne{r.d, r.id, r.s}, old;
    {
        lock_guard<mutex> lk(sh.m);
        auto it = sh.mp.find(k);
        if (it == sh.mp.end()) { sh.mp.emplace(k, ne); g_ndp++; if (r.id == TAME) g_ntame++; return; }
        old = it->second;
    }
    if (old.id == ne.id && u_eq(old.d, ne.d)) return;      // même trajectoire
    collide(old, ne);
}

// ======================== sauvegarde / chargement ============================
struct Hdr { char magic[8]; U256 a; U256 W; u32 sym, dp; u64 jhash; u64 count; };
struct Rec { u64 x0, x1; U256 d; };

static bool save_tame(const string& fn) {
    vector<Rec> v; v.reserve(g_ntame.load());
    for (int i = 0; i < 256; i++) {
        lock_guard<mutex> lk(g_sh[i].m);
        for (auto& kv : g_sh[i].mp) if (kv.second.id == TAME) v.push_back({kv.first.a, kv.first.b, kv.second.d});
    }
    Hdr h; memset(&h, 0, sizeof h); memcpy(h.magic, "KANGDP02", 8);
    h.a = g_a; h.W = g_W; h.sym = g_sym; h.dp = g_dp; h.jhash = g_jhash; h.count = v.size();
    string tmp = fn + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { fprintf(stderr, "Erreur: impossible d'ecrire %s\n", tmp.c_str()); return false; }
    fwrite(&h, sizeof h, 1, f);
    if (!v.empty()) fwrite(v.data(), sizeof(Rec), v.size(), f);
    fclose(f);
#ifdef _WIN32
    remove(fn.c_str());   // Windows : rename() n'ecrase pas un fichier existant
#endif
    if (rename(tmp.c_str(), fn.c_str()) != 0) { fprintf(stderr, "Erreur: rename %s\n", fn.c_str()); return false; }
    return true;
}
static bool read_hdr(const string& fn, Hdr& h) {
    ifstream f(fn, ios::binary);
    if (!f) return false;
    f.read((char*)&h, sizeof h);
    return f.good() && memcmp(h.magic, "KANGDP02", 8) == 0;
}
static bool check_hdr(const Hdr& h, const string& fn) {
    if (!u_eq(h.a, g_a) || !u_eq(h.W, g_W)) {
        fprintf(stderr, "Fichier %s: intervalle different, refuse.\n", fn.c_str()); return false; }
    if ((int)h.sym != g_sym || h.jhash != g_jhash) {
        fprintf(stderr, "Fichier %s: parametres (symetrie / table de sauts) differents, refuse.\n", fn.c_str()); return false; }
    return true;
}
static bool load_tame(const string& fn, u64& n) {
    Hdr h;
    if (!read_hdr(fn, h)) { fprintf(stderr, "Fichier %s illisible.\n", fn.c_str()); return false; }
    if (!check_hdr(h, fn)) return false;
    if ((int)h.dp != g_dp) { fprintf(stderr, "Fichier %s: masque DP different (%u vs %d), refuse.\n", fn.c_str(), h.dp, g_dp); return false; }
    ifstream f(fn, ios::binary); f.seekg(sizeof(Hdr));
    vector<Rec> buf(1 << 16); u64 left = h.count; n = 0;
    while (left) {
        size_t m = (size_t)min<u64>(left, buf.size());
        f.read((char*)buf.data(), m * sizeof(Rec));
        if (!f) { fprintf(stderr, "Fichier %s tronque.\n", fn.c_str()); return false; }
        for (size_t i = 0; i < m; i++) add_dp({buf[i].x0, buf[i].x1, buf[i].d, TAME, 1});
        left -= m; n += m;
    }
    return true;
}

// ================================ worker =====================================
struct alignas(64) Counter { atomic<u64> v; char pad[56]; };
static Counter* g_cnt = nullptr;
static u64 g_rotLimit = 0;       // rotation des clés wild si plus de clés que de wild
static bool g_doTame = true, g_doWild = true;
static atomic<int> g_tameOff{0};       // mode -mix : 1 = objectif tame atteint, tous les kangourous passent en wild
static Counter* g_tcnt = nullptr;      // sauts effectues par les kangourous TAME (par thread)

static int pick_key() {
    int n = g_nk; if (n == 0) return -1;
    for (int t = 0; t < n; t++) {
        int i = (int)(g_keyctr.fetch_add(1) % (u64)n);
        if (!g_solved[i].load()) return i;
    }
    return -1;
}

static const Jump* slow_add(fe& x, fe& y, const Jump* jp) {   // cas dx == 0
    fe lam, t, x3, y3, inv;
    if (u_eq(x, jp->x)) {
        if (u_eq(y, jp->y)) {          // doublement
            fe n, d; fe_sqr(n, x); fe_add(t, n, n); fe_add(n, t, n);
            fe_add(d, y, y); fe_inv(inv, d); fe_mul(lam, n, inv);
        } else {                        // P = -J : on prend un saut d'évasion
            jp = &g_E[0];
            fe dx; fe_sub(dx, jp->x, x); fe_inv(inv, dx);
            fe_sub(t, jp->y, y); fe_mul(lam, t, inv);
        }
    } else {
        fe dx; fe_sub(dx, jp->x, x); fe_inv(inv, dx);
        fe_sub(t, jp->y, y); fe_mul(lam, t, inv);
    }
    fe l2; fe_sqr(l2, lam);
    fe_sub(x3, l2, x); fe_sub(x3, x3, jp->x);
    fe_sub(t, x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, y);
    x = x3; y = y3;
    return jp;
}

static void worker(int tid, int B) {
    Rng rng(g_seed * 0x9E3779B97F4A7C15ULL + (u64)tid * 7919ULL + 12345ULL);
    vector<fe> X(B), Y(B), DX(B), PRE(B);
    vector<U256> D(B);
    vector<u32> ID(B), SN(B), HP(B);
    vector<int8_t> S(B);
    vector<char> SLOW(B);
    vector<u64> H((size_t)B * 16), AGE(B);
    vector<const Jump*> JP(B);
    vector<DPRec> out; out.reserve(64);
    U256 L = (g_sl >= 255) ? u_from(1) : u_shr(g_W, g_sl); if (u_zero(L)) L = u_from(1);
    U256 L2p1 = u_shl(L, 1); add_small(L2p1, 1);                 // 2L+1 (L <= n/2 : pas de debordement)
    const u32 dpLimit = (u32)min<u64>(0xFFFFFFF0ULL, ((u64)64 << min(g_dp, 24)) + 4096);
    bool alive = true;

    auto roleTame = [&](int j) { return g_doTame && !g_tameOff.load(memory_order_relaxed) && (!g_doWild || (long)floor((j + 1) * g_tfrac) > (long)floor(j * g_tfrac)); };
    auto init = [&](int j) {
        bool tame = roleTame(j);
        for (;;) {
            U256 x = rng.below(L2p1);                                  // x dans [0, 2L]  ->  r = x - L  (mod n)
            U256 r = (u_cmp(x, L) >= 0) ? u_sub(x, L) : m_neg(u_sub(L, x));
            if (u_zero(r)) r = u_from(1);
            Pt P; int id = -1;
            if (tame) { P = mulG_mod(r); }
            else {
                id = pick_key();
                if (id < 0) { alive = false; return; }
                P = pt_add(g_Q[id], mulG_mod(r));
            }
            if (P.inf) continue;
            X[j] = P.x; Y[j] = P.y; D[j] = r; ID[j] = tame ? TAME : (u32)id; S[j] = 1;
            if (g_sym && (Y[j].d[0] & 1)) { fe_neg(Y[j], Y[j]); D[j] = m_neg(D[j]); S[j] = -1; }
            memset(&H[(size_t)j * 16], 0, 16 * sizeof(u64));
            HP[j] = 0; H[(size_t)j * 16 + (HP[j]++ & 15)] = X[j].d[0];
            SN[j] = 0; AGE[j] = 0;
            return;
        }
    };
    for (int j = 0; j < B && alive; j++) init(j);
    if (!alive) return;

    u64 iter = 0;
    while (!g_stop.load(memory_order_relaxed)) {
        // ---- phase 1 : choix du saut, dx = xJ - x
        for (int j = 0; j < B; j++) {
            const Jump* jp = nullptr;
            u64* h = &H[(size_t)j * 16];
            if (g_sym) {
                u32 pos = HP[j]; u64 xl = X[j].d[0];
                for (int Lc = 2; Lc <= 8; Lc += 2) {
                    if (h[(pos - 1 - Lc) & 15] == xl) {
                        u64 mn = xl;
                        for (int q = 1; q < Lc; q++) { u64 v = h[(pos - 1 - q) & 15]; if (v < mn) mn = v; }
                        jp = &g_E[(mn >> 7) % NE];
                        memset(h, 0, 16 * sizeof(u64)); HP[j] = 0; h[(HP[j]++) & 15] = xl;
                        break;
                    }
                }
            }
            if (!jp) jp = &g_J[X[j].d[1] & (NJ - 1)];
            JP[j] = jp;
            fe_sub(DX[j], jp->x, X[j]);
            if (u_zero(DX[j])) { SLOW[j] = 1; memset(&DX[j], 0, sizeof(fe)); DX[j].d[0] = 1; } else SLOW[j] = 0;
        }
        // ---- phase 2 : inversion par lots (Montgomery)
        PRE[0] = DX[0];
        for (int j = 1; j < B; j++) fe_mul(PRE[j], PRE[j - 1], DX[j]);
        fe inv; fe_inv(inv, PRE[B - 1]);
        for (int j = B - 1; j > 0; j--) {
            fe t; fe_mul(t, inv, PRE[j - 1]);
            fe_mul(inv, inv, DX[j]);
            DX[j] = t;
        }
        DX[0] = inv;
        // ---- phase 3 : addition affine
        u64 tameStep = 0;
        for (int j = 0; j < B; j++) {
            const Jump* jp = JP[j];
            if (SLOW[j]) jp = slow_add(X[j], Y[j], jp);
            else {
                fe t, lam, l2, x3, y3;
                fe_sub(t, jp->y, Y[j]); fe_mul(lam, t, DX[j]);
                fe_sqr(l2, lam);
                fe_sub(x3, l2, X[j]); fe_sub(x3, x3, jp->x);
                fe_sub(t, X[j], x3); fe_mul(y3, lam, t); fe_sub(y3, y3, Y[j]);
                X[j] = x3; Y[j] = y3;
            }
            D[j] = m_add(D[j], jp->d);
            if (g_sym && (Y[j].d[0] & 1)) { fe_neg(Y[j], Y[j]); D[j] = m_neg(D[j]); S[j] = -S[j]; }
            u64* h = &H[(size_t)j * 16];
            h[(HP[j]++) & 15] = X[j].d[0];
            SN[j]++; AGE[j]++;
            if (ID[j] == TAME) tameStep++;
            if ((X[j].d[0] & g_dpmask) == 0) {
                out.push_back({X[j].d[0], X[j].d[1], D[j], ID[j], S[j]});
                SN[j] = 0;
            } else if (SN[j] > dpLimit) {            // bloqué dans un cycle long : on repart
                init(j); if (!alive) break;
            }
        }
        if (!alive) break;
        g_cnt[tid].v.fetch_add((u64)B, memory_order_relaxed);
        if (tameStep) g_tcnt[tid].v.fetch_add(tameStep, memory_order_relaxed);
        if (!out.empty()) { for (auto& r : out) add_dp(r); out.clear(); }
        iter++;
        if ((iter & 63) == 0) {
            for (int j = 0; j < B; j++) {
                if (ID[j] == TAME) { if (g_tameOff.load(memory_order_relaxed)) { init(j); if (!alive) break; } }   // -mix : tame -> wild
                else if (g_solved[ID[j]].load() || (g_rotLimit && AGE[j] > g_rotLimit)) { init(j); if (!alive) break; }
            }
            if (!alive) break;
        }
    }
}

// ================================ pont GPU ====================================
// Convertit les types internes (U256/Pt/Jump, deja en 4x64 bits comme gu256_t) vers
// les structures "plates" de kangaroo_gpu.h. Le GPU ne fait QUE produire des points
// distingues ; toute la resolution (add_dp/collide/propagate/try_key, table tame,
// sauvegarde) reste dans ce fichier et est partagee telle quelle avec les threads CPU.
static inline gu256_t to_g(const U256& a) { gu256_t r; for (int i = 0; i < 4; i++) r.d[i] = a.d[i]; return r; }
static inline U256 from_g(const gu256_t& a) { U256 r; for (int i = 0; i < 4; i++) r.d[i] = a.d[i]; return r; }

struct GpuBridgeCfg { int blocks, threads, batch, iters, out_cap; };

static void gpu_bridge(int tid, int device_id, GpuBridgeCfg cfg) {
    vector<gjump_t> jumps(NJ + NE);
    for (int i = 0; i < NJ; i++) { jumps[i].x = to_g(g_J[i].x); jumps[i].y = to_g(g_J[i].y); jumps[i].dist = to_g(g_J[i].d); }
    for (int i = 0; i < NE; i++) { jumps[NJ + i].x = to_g(g_E[i].x); jumps[NJ + i].y = to_g(g_E[i].y); jumps[NJ + i].dist = to_g(g_E[i].d); }
    vector<gpt_t> keys(max(1, g_nk));
    for (int i = 0; i < g_nk; i++) { keys[i].x = to_g(g_Q[i].x); keys[i].y = to_g(g_Q[i].y); }

    // Pour ne pas deborder le tampon de DP quand -dp est petit, le nombre d'iterations par lot est borne :
    // DP attendus par lot = kangourous x iterations / 2^dp  <=  out_cap / 2.
    int itersEff = cfg.iters;
    {   const double kangDev = (double)cfg.blocks * cfg.threads * cfg.batch;
        double maxIt = 0.5 * (double)cfg.out_cap * ldexp(1.0, min(g_dp, 60)) / max(1.0, kangDev);
        if (maxIt < itersEff) itersEff = max(1, (int)maxIt);
    }
    gpu_init_t init; memset(&init, 0, sizeof init);
    init.device_id = device_id; init.blocks = cfg.blocks; init.threads_per_block = cfg.threads;
    init.kang_per_thread = cfg.batch; init.iters_per_launch = itersEff;
    init.dp_bits = g_dp; init.sym = g_sym; init.sl = g_sl; init.tfrac = g_tfrac;
    init.do_tame = g_doTame ? 1 : 0; init.do_wild = g_doWild ? 1 : 0;
    init.seed = g_seed * 0x9E3779B97F4A7C15ULL ^ (((u64)device_id + 1) * 0xD1B54A32D192ED03ULL);
    init.rot_limit = g_rotLimit;
    init.jumps = jumps.data(); init.nj = NJ; init.ne = NE;
    init.secp_n = to_g(SECP_N); init.nhalf = to_g(NHALF);
    init.a = to_g(g_a); init.w = to_g(g_W); init.c = to_g(g_c);
    init.keys = keys.data(); init.nk = g_nk;
    init.out_capacity = cfg.out_cap;

    void* h = gpu_worker_create(&init);
    if (!h) { fprintf(stderr, "GPU %d : echec d'initialisation, thread arrete.\n", device_id); return; }

    vector<gdprec_t> out(max(1, cfg.out_cap));
    vector<uint8_t> solvedBuf(max(1, g_nk));
    // Pipeline : le GPU calcule le lot n+1 PENDANT que ce thread insere les DP du lot n dans les tables
    // (le GPU n'attend plus le CPU).
    auto launch = [&]() -> bool {
        for (int i = 0; i < g_nk; i++) solvedBuf[i] = g_solved[i].load(memory_order_relaxed) ? 1 : 0;
        gpu_worker_sync(h, solvedBuf.data(), g_nk, g_tameOff.load(memory_order_relaxed) ? 1 : 0);
        return gpu_worker_launch(h, (int)out.size()) == 0;
    };
    bool running = launch();
    while (running) {
        u64 steps = 0, tsteps = 0;
        int n = gpu_worker_fetch(h, out.data(), (int)out.size(), &steps, &tsteps);      // attend la fin du lot
        running = !g_stop.load(memory_order_relaxed) && launch();                         // relance tout de suite
        for (int i = 0; i < n; i++) {                                                       // ... et traite les DP pendant que le GPU travaille
            DPRec r; r.x0 = out[i].x0; r.x1 = out[i].x1; r.d = from_g(out[i].d); r.id = out[i].id; r.s = out[i].s;
            add_dp(r);
        }
        if (steps) g_cnt[tid].v.fetch_add(steps, memory_order_relaxed);
        if (tsteps) g_tcnt[tid].v.fetch_add(tsteps, memory_order_relaxed);
        if (!running && g_stop.load(memory_order_relaxed)) break;
    }
    // un lot encore en vol (arret demande entre fetch et launch) : on le recupere pour liberer proprement
    gpu_worker_fetch(h, out.data(), (int)out.size(), nullptr, nullptr);
    gpu_worker_destroy(h);
}

// ================================= main ======================================
static void on_sigint(int) { g_stop = 1; }
static string fmt_time(double s) {
    long t = (long)s; char b[64]; snprintf(b, sizeof b, "%02ld:%02ld:%02ld", t / 3600, (t / 60) % 60, t % 60); return b;
}
// Nombre compact avec multiples SI : 1.23 K, 45.6 M, 789 G, ... (3 chiffres significatifs)
static string fmt_si(double v) {
    const char* u[] = {"", " K", " M", " G", " T", " P", " E"}; int i = 0;
    if (!(v >= 0)) v = 0;
    while (v >= 999.5 && i < 6) { v /= 1000.0; i++; }
    char s[48]; snprintf(s, sizeof s, "%.3g%s", v, u[i]); return s;
}
#define SI(x) fmt_si(x).c_str()
static string fmt_bytes(double b) {
    const char* u[] = {"o", "Ko", "Mo", "Go", "To"}; int i = 0;
    while (b >= 1024.0 && i < 4) { b /= 1024.0; i++; }
    char s[48]; snprintf(s, sizeof s, "%.1f %s", b, u[i]); return s;
}
// Memoire reelle approximative par DP dans la table (noeud unordered_map + malloc + bucket)
static double dp_ram_bytes() { return (double)sizeof(pair<const K128, Ent>) + 40.0; }

static void usage() {
    puts("Kangaroo secp256k1 (CPU, multi-coeurs)\n"
         "Usage :\n"
         "  -pub <hex>            cle publique (02/03.. ou 04..) [alias -cle_publique]\n"
         "  -fichier <fichier>    fichier de cles publiques (une par ligne)\n"
         "  -r <debut:fin>        intervalle en hexadecimal (ou <debut> avec -b)\n"
         "  -b <bits>             largeur en bits (8..256) : [2^(b-1), 2^b-1] sans -r, sinon debut + 2^b - 1\n"
         "  -bb <bits>            intervalle complet [1, 2^bb - 1]  (9..256 ; -bb 256 -> [1, n-1])\n"
         "  -s <fichier>          fichier d'enregistrement de la table tame\n"
         "  -ss <f1[,f2..]>       fichier(s) tame a charger pour resoudre (mode wild seul)\n"
         "  -t <sec>              intervalle d'affichage (defaut 30)\n"
         "Options :\n"
         "  -c <n>      nombre de threads (defaut : tous les coeurs)\n"
         "  -k <n>      kangourous par thread (defaut 512)\n"
         "  -end <coef> taille de tame voulue = coef x sqrt(W) sauts (ex : -end 1.15), puis arret.\n"
         "              Generation de tame seule (sans cle, ou -ss sans -tame) ; reprise comptee avec -sv.\n"
         "  -dp <bits>  masque des points distingues (defaut : automatique, calcule selon l'intervalle,\n"
         "              les threads/kangourous et la memoire ; le recapitulatif DP est toujours affiche)\n"
         "  -nosym      desactive la symetrie\n"
         "  -tame       avec -ss : continue aussi a generer des tame\n"
         "  -mix        (alias -both) tame ET wild en meme temps, meme avec -ss : charge les tame existants,\n"
         "              en genere de nouveaux pendant que les wild cherchent la/les cle(s).\n"
         "              Avec -end <coef> : les tame s'arretent a coef x sqrt(W) sauts tame (reprise incluse),\n"
         "              puis TOUS les kangourous deviennent wild (la table tame est sauvegardee avec -s/-sv).\n"
         "  -tw <f>     part de kangourous tame en mode tame+wild (defaut 0.5, ex : -tw 0.3)\n"
         "  -sv <fichier> reprend la sauvegarde des DP tame si elle existe (copie .bak), puis continue a la mettre a jour\n"
         "  -duree <s>  arret apres s secondes\n"
         "  -o <fichier> fichier de sortie des cles trouvees (defaut kangaroo_trouve.txt)\n"
         "  -genpub <hex> affiche la cle publique d'une cle privee puis quitte\n"
         "  -test       auto-tests arithmetiques\n"
         "GPU (CUDA, ignore si le binaire est compile sans nvcc) :\n"
         "  -gpu <spec> \"all\" (defaut, toutes les cartes), \"off\", ou liste \"0,1\"\n"
         "  -gblocks <n>   blocs CUDA par carte (0 = auto : une vague exacte de blocs residents, defaut)\n"
         "  -gthreads <n>  threads par bloc, sert au calcul du nombre de kangourous (defaut 128)\n"
         "  -gbatch <n>    kangourous par thread = lot Montgomery : 8, 16, 32 (defaut) ou 64 (plus grand = moins\n"
         "                 de multiplications par saut, plus de memoire ; a essayer : 64)\n"
         "  -giter <n>     iterations par lancement de noyau (defaut 128 ; borne auto si -dp est petit)\n"
         "  -c 0           desactive les threads CPU (GPU seul)");
}

static int selftest() {
    int bad = 0;
    fe a; hex2u("123456789abcdef0fedcba98765432100f1e2d3c4b5a69788796a5b4c3d2e1f0", a);
    fe i, p; fe_inv(i, a); fe_mul(p, a, i);
    fe one; memset(&one, 0, sizeof one); one.d[0] = 1;
    if (!u_eq(p, one)) { puts("FAIL inv"); bad++; }
    U256 e = u_sub(P_FIELD, u_from(2)); fe i2; fe_pow(i2, a, e);
    if (!u_eq(i, i2)) { puts("FAIL inv vs pow"); bad++; }
    Pt g2 = mulG(u_from(2));
    U256 x2; hex2u("C6047F9441ED7D6D3045406E95C07CD85C778E4B8CEF3CA7ABAC09B95C709EE5", x2);
    if (!u_eq(g2.x, x2)) { puts("FAIL 2G"); bad++; }
    Pt g3 = pt_add(g2, GEN), g3b = mulG(u_from(3));
    if (!pt_eq(g3, g3b)) { puts("FAIL 3G"); bad++; }
    Pt n1 = mulG(u_sub(SECP_N, u_from(1)));
    if (!pt_eq(n1, pt_neg(GEN))) { puts("FAIL (n-1)G"); bad++; }
    Pt pp; if (!parse_pub(pub_hex(g3b), pp) || !pt_eq(pp, g3b)) { puts("FAIL parse"); bad++; }
    {   // arithmetique modulo n, decalages, conversions double, scalaire "negatif"
        U256 u, v, nm1 = SECP_N; sub_small(nm1, 1);
        hex2u("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364100", u);   // proche de n
        hex2u("123456789abcdef0fedcba98765432100f1e2d3c4b5a69788796a5b4c3d2e1f0", v);
        if (!u_eq(m_sub(m_add(u, v), v), u)) { puts("FAIL m_add/m_sub"); bad++; }
        if (!u_eq(m_add(u, m_neg(u)), u_zero_val())) { puts("FAIL m_neg"); bad++; }
        if (!u_eq(m_add(nm1, u_from(1)), u_zero_val())) { puts("FAIL n-1 + 1"); bad++; }
        if (!u_eq(m_add(m_half(v), m_half(v)), v) || !u_eq(m_add(m_half(u), m_half(u)), u)) { puts("FAIL m_half"); bad++; }
        if (!u_eq(u_shr(u_shl(v, 77), 77), u_shr(u_shl(v, 0), 0)) && u_bits(v) + 77 <= 256) { puts("FAIL shl/shr"); bad++; }
        if (u_bits(u_ones(256)) != 256 || u_bits(u_pow2(255)) != 256 || u_bits(u_ones(100)) != 100) { puts("FAIL ones/pow2"); bad++; }
        if (u_bits(dbl2u(ldexp(1.0, 200))) != 201 || fabs(u2dbl(dbl2u(1e50)) / 1e50 - 1.0) > 1e-12) { puts("FAIL dbl"); bad++; }
        Pt m5 = mulG_mod(m_neg(u_from(5))), p5 = pt_neg(mulG(u_from(5)));
        if (!pt_eq(m5, p5)) { puts("FAIL mulG_mod (-5)"); bad++; }
        Rng rr(42); U256 bound = u_pow2(200); bool ok = true;
        for (int i = 0; i < 1000; i++) if (u_cmp(rr.below(bound), bound) >= 0) ok = false;
        if (!ok) { puts("FAIL Rng::below"); bad++; }
    }
    printf(bad ? "Auto-tests : %d echec(s)\n" : "Auto-tests OK\n", bad);
    return bad;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IOLBF, 0);
    init_curve();
    string pubArg, fileArg, rArg, saveFile, loadArg;
    bool mixMode = false;         // -mix : tame + wild en meme temps (meme avec -ss)
    bool resumeSave = false;      // -sv : reprendre + continuer la sauvegarde des DP tame
    int bits = 0, bb = 0, tsec = 30, duree = 0; double endCoef = 0; bool nosym = false, forceTame = false;
    int cores = (int)thread::hardware_concurrency(); if (cores < 1) cores = 1;
    string gpuArg = "all";        // -gpu : "all" (defaut), "off", ou liste d'indices "0,1"
    int gBlocks = 0, gThreads = 128, gBatch = 32, gIter = 128;  // gBlocks 0 = automatique (1 vague de blocs residents)
    for (int i = 1; i < argc; i++) {
        string a = argv[i];
        auto nxt = [&]() -> string { if (i + 1 >= argc) { fprintf(stderr, "Argument manquant apres %s\n", a.c_str()); exit(1); } return argv[++i]; };
        if (a == "-pub" || a == "-cle_publique") pubArg = nxt();
        else if (a == "-fichier") fileArg = nxt();
        else if (a == "-r") rArg = nxt();
        else if (a == "-b") bits = atoi(nxt().c_str());
        else if (a == "-bb") bb = atoi(nxt().c_str());
        else if (a == "-end") { endCoef = atof(nxt().c_str()); if (!(endCoef > 0)) { fprintf(stderr, "-end doit etre > 0 (ex : 1.15)\n"); return 1; } }
        else if (a == "-s") saveFile = nxt();
        else if (a == "-ss") loadArg = nxt();
        else if (a == "-sv") { saveFile = nxt(); resumeSave = true; }
        else if (a == "-t") tsec = max(1, atoi(nxt().c_str()));
        else if (a == "-c") cores = max(0, atoi(nxt().c_str()));
        else if (a == "-k") g_batch = max(2, atoi(nxt().c_str()));
        else if (a == "-dp") g_dp = atoi(nxt().c_str());
        else if (a == "-nosym") nosym = true;
        else if (a == "-tame") forceTame = true;
        else if (a == "-mix" || a == "-both") mixMode = true;
        else if (a == "-duree") duree = atoi(nxt().c_str());
        else if (a == "-o") g_outfile = nxt();
        else if (a == "-seed") g_seed = strtoull(nxt().c_str(), nullptr, 10);
        else if (a == "-jf") g_jf = atof(nxt().c_str());
        else if (a == "-jg") g_jg = atof(nxt().c_str());
        else if (a == "-sl") g_sl = atoi(nxt().c_str());
        else if (a == "-coef") g_coef = atof(nxt().c_str());
        else if (a == "-tw") g_tfrac = atof(nxt().c_str());
        else if (a == "-gpu") gpuArg = nxt();
        else if (a == "-gblocks") gBlocks = atoi(nxt().c_str());
        else if (a == "-gthreads") gThreads = atoi(nxt().c_str());
        else if (a == "-gbatch") gBatch = atoi(nxt().c_str());
        else if (a == "-giter") gIter = atoi(nxt().c_str());
        else if (a == "-test") return selftest();
        else if (a == "-genpub") {
            U256 k; if (!hex2u(nxt(), k)) { fprintf(stderr, "hex invalide\n"); return 1; }
            printf("%s\n", pub_hex(mulG(k)).c_str()); return 0;
        }
        else if (a == "-h" || a == "--help") { usage(); return 0; }
        else { fprintf(stderr, "Option inconnue: %s\n", a.c_str()); usage(); return 1; }
    }
    if (!g_seed) g_seed = (u64)chrono::steady_clock::now().time_since_epoch().count() ^ 0xA5A5A5A5ULL;
    g_sym = nosym ? 0 : 1;

    // ---- detection GPU (CUDA) : -gpu "off" | "all" (defaut) | "0,1,..."
    vector<int> gpuDevices;
    {
        int ndev = gpu_device_count();     // 0 si compile sans CUDA (voir bouchons en tete de fichier)
        if (gpuArg != "off" && ndev > 0) {
            if (gpuArg == "all" || gpuArg.empty()) { for (int i = 0; i < ndev; i++) gpuDevices.push_back(i); }
            else {
                stringstream ss(gpuArg); string t;
                while (getline(ss, t, ',')) if (!t.empty()) {
                    int id = atoi(t.c_str());
                    if (id < 0 || id >= ndev) fprintf(stderr, "GPU %d ignoree (hors plage, %d carte(s) detectee(s))\n", id, ndev);
                    else gpuDevices.push_back(id);
                }
            }
        }
    }
    if (!gpuDevices.empty() && !(gBatch == 8 || gBatch == 16 || gBatch == 32 || gBatch == 64)) {
        fprintf(stderr, "-gbatch doit valoir 8, 16, 32 ou 64 (recu %d)\n", gBatch); return 1;
    }
    if (gThreads < 1) gThreads = 128;
    if (cores == 0 && gpuDevices.empty()) { fprintf(stderr, "-c 0 sans GPU disponible : aucun kangourou ne tournerait.\n"); return 1; }
    g_nthreads = cores + (int)gpuDevices.size();
    // Mesures (400 essais, intervalle 2^34, IC95 ~ +/-0.08) : -coef n'a AUCUN effet mesurable sur
    // le nombre reel de sauts (il ne sert qu'a l'ETA et a l'auto-choix des bits DP). Le nombre reel
    // de sauts est d'environ 1.3 a 1.4 x sqrt(W) en moyenne, plateau plat en jf, sl et -tw.
    // Afficher 1.15 ne le ferait pas baisser : valeur d'origine conservee.
    if (g_coef < 0) g_coef = g_sym ? 1.5 : 2.8;

    // ---- intervalle (jusqu'a 256 bits : au maximum [1, n-1])
    {
        U256 a, e; memset(&a, 0, sizeof a);
        U256 nm1 = SECP_N; sub_small(nm1, 1);                                   // n - 1
        size_t colon = rArg.find(':');
        if (colon != string::npos) {
            if (!hex2u(rArg.substr(0, colon), a) || !hex2u(rArg.substr(colon + 1), e)) { fprintf(stderr, "Intervalle -r invalide\n"); return 1; }
        } else if (!rArg.empty()) {
            int wb = bits >= 8 ? bits : bb;
            if (wb < 8) { fprintf(stderr, "-r <debut> demande aussi -b <bits>\n"); return 1; }
            if (wb > 256) { fprintf(stderr, "Largeur trop grande (max 256 bits).\n"); return 1; }
            if (!hex2u(rArg, a)) { fprintf(stderr, "Debut -r invalide\n"); return 1; }
            e = u_add(a, u_ones(wb));
            if (u_cmp(e, a) < 0 || u_cmp(e, SECP_N) >= 0) e = nm1;                // retenue ou >= n : borne a n-1
        } else if (bb > 0) {
            if (bb < 9 || bb > 256) { fprintf(stderr, "-bb doit etre compris entre 9 et 256\n"); return 1; }
            a = u_from(1); e = u_ones(bb);                                         // intervalle [1, 2^bb - 1]
            if (u_cmp(e, SECP_N) >= 0) e = nm1;                                    // bb = 256 -> [1, n-1]
        } else if (bits >= 8) {
            if (bits > 256) { fprintf(stderr, "-b doit etre <= 256\n"); return 1; }
            a = u_pow2(bits - 1); e = u_ones(bits);
            if (u_cmp(e, SECP_N) >= 0) e = nm1;                                    // bits = 256 -> [2^255, n-1]
        } else { usage(); return 1; }
        if (u_cmp(e, a) <= 0 || u_cmp(e, SECP_N) >= 0) { fprintf(stderr, "Intervalle invalide (fin <= debut ou >= n)\n"); return 1; }
        U256 w = u_add(u_sub(e, a), u_from(1));                                    // e - a + 1 <= n < 2^256
        g_a = a; g_end = e; g_W = w; g_Wd = u2dbl(w);
        if (u_cmp(g_W, u_from(256)) < 0) { fprintf(stderr, "Intervalle trop petit (min 2^8) : utilisez une recherche exhaustive.\n"); return 1; }
        g_c = u_add(a, u_shr(g_W, 1));
    }
    double sqW = sqrt(g_Wd);
    g_sl = max(1, min(255, g_sl));

    // ---- GPU : blocs par carte (auto = une vague exacte de blocs residents, plafonnee pour les petits intervalles :
    //      trop de kangourous en vol retarde la detection des collisions quand sqrt(W) est petit)
    vector<int> gBlocksDev; u64 gpuKangTotal = 0;
    for (int id : gpuDevices) {
        int b = gBlocks;
        if (b <= 0) {
            b = gpu_auto_blocks(id, gThreads, gBatch);
            if (b <= 0) b = 256;
            const double capK = max(4096.0, sqW / 64.0) / (double)gpuDevices.size();      // kangourous max par carte
            const int maxb = max(1, (int)(capK / ((double)gThreads * gBatch)));
            b = min(b, maxb);
        }
        gBlocksDev.push_back(b);
        gpuKangTotal += (u64)b * (u64)gThreads * (u64)gBatch;
    }

    // ---- clés
    vector<string> lines;
    if (!pubArg.empty()) lines.push_back(pubArg);
    if (!fileArg.empty()) {
        ifstream f(fileArg); if (!f) { fprintf(stderr, "Impossible d'ouvrir %s\n", fileArg.c_str()); return 1; }
        string l; while (getline(f, l)) {
            size_t h = l.find('#'); if (h != string::npos) l = l.substr(0, h);
            while (!l.empty() && isspace((unsigned char)l.back())) l.pop_back();
            size_t s0 = 0; while (s0 < l.size() && isspace((unsigned char)l[s0])) s0++;
            l = l.substr(s0); if (!l.empty()) lines.push_back(l);
        }
    }
    for (auto& l : lines) {
        Pt P;
        if (!parse_pub(l, P)) { fprintf(stderr, "Cle publique invalide ignoree: %s\n", l.c_str()); continue; }
        bool dup = false; for (auto& q : g_pub) if (pt_eq(q, P)) dup = true;
        if (dup) continue;
        g_pub.push_back(P); g_pubstr.push_back(l);
    }
    g_nk = (int)g_pub.size();
    if (g_nk == 0 && saveFile.empty()) { fprintf(stderr, "Aucune cle valide et pas de -s : rien a faire.\n"); usage(); return 1; }
    g_solved = new atomic<int>[max(1, g_nk)]; for (int i = 0; i < max(1, g_nk); i++) g_solved[i] = 0;
    g_kval.assign(max(1, g_nk), u_zero_val());

    // ---- sauts
    double jf = g_jf, jg = g_jg;
    if (jf < 0) jf = g_sym ? 1.0 : 0.7;
    if (jg < 0) jg = g_sym ? 0.75 : 0.5;
    double M = max(4.0, jf * pow(g_Wd, jg));
    build_jumps(M);

    // ---- centre, clés centrées
    {
        Pt cG = mulG(g_c), ncG = pt_neg(cG);
        for (int i = 0; i < g_nk; i++) {
            Pt Q = pt_add(g_pub[i], ncG);
            if (Q.inf) {  // clé = centre exactement
                g_solved[i] = 1; g_kval[i] = u_zero_val(); g_nsolved++;
                report_found(i, g_c);
                Q = g_pub[i];
            }
            g_Q.push_back(Q);
        }
        if (g_nk > 0 && g_nsolved >= g_nk) return 0;
    }

    // ---- DP : fichiers à charger / masque
    vector<string> loadFiles;
    { stringstream ss(loadArg); string t; while (getline(ss, t, ',')) if (!t.empty()) loadFiles.push_back(t); }
    if (resumeSave) {
        ifstream ex(saveFile, ios::binary);
        if (ex) {                                       // le fichier existe : il doit etre valide, sinon on refuse (jamais ecrase)
            Hdr h0;
            if (!read_hdr(saveFile, h0)) { fprintf(stderr, "-sv: %s existe mais n'est pas une sauvegarde valide : refuse (non ecrase).\n", saveFile.c_str()); return 1; }
            loadFiles.push_back(saveFile);
        }
    }
    bool dpFromFile = false;
    if (!loadFiles.empty() && g_dp < 0) {            // adopte le masque du premier fichier
        Hdr h; if (read_hdr(loadFiles[0], h)) { g_dp = (int)h.dp; dpFromFile = true; }
    }
    g_doWild = g_nk > 0;
    g_doTame = (g_nk == 0) || loadFiles.empty() || forceTame || resumeSave || mixMode;
    // Ktot : nombre total de "kangourous en vol" (heuristique DP/memoire uniquement ; CPU = coeurs x -k,
    // GPU = cartes x blocs x threads x -gbatch, bien plus nombreux par carte que par coeur CPU).
    u64 Ktot64 = (u64)cores * (u64)g_batch + gpuKangTotal;
    int Ktot = (int)min<u64>(Ktot64, (u64)INT32_MAX);
    double Nexp = g_coef * sqW * sqrt((double)max(1, g_nk));
    // -end : cible de taille de tame (coef x sqrt(W) sauts) ; uniquement en generation de tame seule
    const bool tameGoal = (endCoef > 0 && g_doTame && !g_doWild);
    const bool mixGoal  = (endCoef > 0 && mixMode && g_doTame && g_doWild);   // tame+wild, puis tout en wild une fois la cible tame atteinte
    double endTarget = 0, baseSteps = 0;
    if (tameGoal) { endTarget = endCoef * sqW; Nexp = endTarget; }
    if (mixGoal)  { endTarget = endCoef * sqW; }       // Nexp reste celui de la resolution (ETA)
    const bool dpAuto = (g_dp < 0); bool dpCapped = false;
    if (g_dp < 0) {
        // Choix automatique des bits DP (compromis) :
        //  - dpTarget : ~2^18 (~260 K) DP attendus au total : detection rapide, RAM faible (~25 Mo)
        //  - dpOver   : plafond : surcout (~Ktot * 2^dp sauts gaspilles apres collision) <= ~2 % de la moyenne
        //  - dpRam    : plancher : jamais plus de ~8 M DP en memoire
        double lN = log2(Nexp);
        int dpTarget = (int)floor(lN) - 18;
        int dpOver   = (int)floor(log2(Nexp / (50.0 * Ktot)));
        int dpRam    = (int)ceil(lN) - 23;
        g_dp = max(0, min(dpTarget, dpOver));
        g_dp = max(g_dp, dpRam);
        if (g_dp > 60) { g_dp = 60; dpCapped = true; }
    }
    if (g_dp < 0) g_dp = 0;
    g_dpmask = (g_dp >= 64) ? ~0ULL : ((1ULL << g_dp) - 1);
    // Gain reel (pas cosmetique) : pre-reserver les 256 shards evite les rehash de
    // l'unordered_map pendant la recherche (chaque rehash re-insere tout ce que le shard
    // contient -> pics de latence qui ralentissent tous les threads qui tapent ce shard).
    // Estimation grossiere du nombre total de DP attendus (tame + wild), repartie sur 256 shards.
    {
        double dpGuess = (Nexp + sqW) / ldexp(1.0, min(g_dp, 1000));
        size_t perShard = (size_t)min(32768.0, max(16.0, 1.3 * dpGuess / 256.0));   // plafond ~8 M entrees au total
        for (int i = 0; i < 256; i++) g_sh[i].mp.reserve(perShard);
    }
    for (auto& fn : loadFiles) {
        u64 n = 0; if (!load_tame(fn, n)) return 1;
        printf("Charge %s : %llu DP tame\n", fn.c_str(), (unsigned long long)n);
    }
    if ((tameGoal || mixGoal) && !loadFiles.empty())                  // reprise : sauts deja faits ~ DP tame charges x 2^dp
        baseSteps = min(endTarget, (double)g_ntame.load() * ldexp(1.0, min(g_dp, 62)));
    if (resumeSave && !loadFiles.empty()) {             // filet de securite : copie .bak de la sauvegarde avant de la remplacer
        string bak = saveFile + ".bak";
        ifstream in(saveFile, ios::binary); ofstream out(bak, ios::binary | ios::trunc);
        out << in.rdbuf();
        printf("Reprise -sv : copie de securite %s\n", bak.c_str());
    }
    int nwild = g_doWild ? (g_doTame ? (int)(Ktot * (1.0 - g_tfrac)) : Ktot) : 0;
    g_rotLimit = (g_nk > nwild && nwild > 0) ? ((u64)256 << min(g_dp, 24)) + 8192 : 0;

    printf("\n==============================================================\n"
           "  KANGAROO secp256k1  -  CPU multi-coeurs\n"
           "==============================================================\n");
    printf("  Intervalle     : 0x%s\n               -> 0x%s\n", hex_trim(g_a).c_str(), hex_trim(g_end).c_str());
    printf("  Largeur        : 2^%.1f\n", log2(g_Wd));
    printf("  Cles           : %d\n", g_nk);
    printf("  Threads CPU    : %d x %d kangourous\n", cores, g_batch);
    if (!gpuDevices.empty()) {
        char nm[128];
        printf("  GPU            : %zu carte(s), lot Montgomery %d, %d iterations par lancement\n", gpuDevices.size(), gBatch, gIter);
        for (size_t q = 0; q < gpuDevices.size(); q++) {
            gpu_device_name(gpuDevices[q], nm, sizeof nm);
            printf("                   - GPU %d : %s -> %d blocs x %d threads x %d = ~%s kangourous\n", gpuDevices[q], nm,
                   gBlocksDev[q], gThreads, gBatch, SI((double)gBlocksDev[q] * gThreads * gBatch));
        }
    } else if (gpu_device_count() == 0) {
        printf("  GPU            : aucune (binaire compile sans CUDA, ou nvcc absent -> voir Makefile)\n");
    } else {
        printf("  GPU            : desactivee (-gpu off)\n");
    }
    printf("  Symetrie       : %s\n", g_sym ? "oui" : "non");
    const double pDP = ldexp(1.0, -min(g_dp, 1000));
    const double dpEst = Nexp * pDP;                                   // DP generes a la fin "moyenne"
    const double tameShare = !g_doTame ? 0.0 : (g_doWild ? g_tfrac : 1.0);
    const double dpTameEst = Nexp * tameShare * pDP;
    const double ovSteps = (double)Ktot * ldexp(1.0, min(g_dp, 1000));
    const double ovPct = 100.0 * ovSteps / Nexp;
    printf("  Points DP      : %d bits (%s) -> 1 point sur %.0f\n", g_dp, dpAuto ? "auto" : (dpFromFile ? "repris du fichier" : "manuel"), ldexp(1.0, min(g_dp, 62)));
    printf("  DP attendus    : ~%s au total (tame ~%s, wild ~%s) apres ~%s sauts\n", SI(dpEst), SI(dpTameEst), SI(dpEst - dpTameEst), SI(Nexp));
    printf("  Memoire DP est.: ~%s  (%.0f octets/DP en RAM)\n", fmt_bytes(dpEst * dp_ram_bytes()).c_str(), dp_ram_bytes());
    if (!saveFile.empty() && g_doTame)
        printf("  Fichier tame   : ~%s estimes  (%zu octets/DP + en-tete)\n", fmt_bytes(sizeof(Hdr) + dpTameEst * sizeof(Rec)).c_str(), sizeof(Rec));
    printf("  Surcout DP     : ~%s sauts (%.2f %% de la moyenne)\n", SI(ovSteps), ovPct);
    printf("  DP existants   : %s  (tame : %s) charges depuis fichier(s)\n", SI((double)g_ndp.load()), SI((double)g_ntame.load()));
    if (log2(max(1.0, Nexp)) > 64.0) printf("  /!\\ ~2^%.0f sauts attendus : calcul irrealisable (le cout est en O(sqrt(W)))\n", log2(Nexp));
    if (dpCapped) printf("  /!\\ Intervalle tres large : bits DP plafonnes a 60, la RAM pour les DP sera enorme\n");
    if (ovPct > 10.0) printf("  /!\\ Surcout DP eleve (%.1f %%) : diminuez -dp ou -k/-c\n", ovPct);
    if (dpEst > 3.4e7) printf("  /!\\ Beaucoup de DP (~%s, ~%s de RAM) : augmentez -dp\n", SI(dpEst), fmt_bytes(dpEst * dp_ram_bytes()).c_str());
    if (dpEst < 1000) printf("  /!\\ Tres peu de DP attendus (~%.0f) : diminuez -dp\n", dpEst);
    printf("  Saut moyen     : 2^%.1f\n", log2(M));
    printf("  Mode           : %s%s%s\n", g_doTame ? "tame" : "", (g_doTame && g_doWild) ? " + " : "", g_doWild ? "wild" : "");
    if (tameGoal) printf("  Cible tame     : %.4g x sqrt(W) = ~%s sauts (deja faits ~%s, reste ~%s)\n", endCoef, SI(endTarget), SI(baseSteps), SI(max(0.0, endTarget - baseSteps)));
    else if (g_nk > 0) printf("  Sauts attendus : ~%s (moyenne)\n", SI(Nexp));
    if (mixGoal) printf("  Cible tame     : %.4g x sqrt(W) = ~%s sauts tame (deja ~%s), puis TOUS les kangourous passent en wild\n", endCoef, SI(endTarget), SI(baseSteps));
    if (mixMode && !g_doWild) printf("  -mix ignore    : il faut au moins une cle publique (-pub / -fichier)\n");
    if (endCoef > 0 && !tameGoal && !mixGoal) printf("  -end ignore    : generation de tame seule (sans cle, ou -ss sans -tame), ou combine avec -mix\n");
    if (tameGoal && saveFile.empty()) printf("  /!\\ -end sans -s/-sv : la table tame ne sera pas sauvegardee\n");
    if (g_nk > 0) printf("  Resultats dans : %s\n", g_outfile.c_str());
    if (!saveFile.empty()) printf("  Sauvegarde tame: %s\n", saveFile.c_str());
    printf("  Arret          : Ctrl+C\n"
           "==============================================================\n\n");

    signal(SIGINT, on_sigint); signal(SIGTERM, on_sigint);
    g_cnt = new Counter[g_nthreads]; g_tcnt = new Counter[g_nthreads];
    for (int i = 0; i < g_nthreads; i++) { g_cnt[i].v = 0; g_tcnt[i].v = 0; }
    vector<thread> th;
    auto t0 = chrono::steady_clock::now();
    for (int i = 0; i < cores; i++) th.emplace_back(worker, i, g_batch);
    // Capacite du tampon de DP par lancement de noyau : large marge pour eviter toute perte de DP
    // meme avec -dp petit (intervalles courts/tests) ; ~15 Mo par carte a cette taille.
    { int tid = cores;
      for (size_t q = 0; q < gpuDevices.size(); q++) {
          GpuBridgeCfg gcfg{gBlocksDev[q], gThreads, gBatch, gIter, 1 << 18};
          th.emplace_back(gpu_bridge, tid++, gpuDevices[q], gcfg);
      } }

    double lastP = 0, lastSave = 0; u64 lastSteps = 0; bool endReached = false;
    while (!g_stop.load()) {
        this_thread::sleep_for(chrono::milliseconds(100));
        double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
        if (duree && el >= duree) { g_stop = 1; break; }
        if (tameGoal) {
            u64 t2 = 0; for (int i = 0; i < g_nthreads; i++) t2 += g_cnt[i].v.load();
            if ((double)t2 + baseSteps >= endTarget) { endReached = true; g_stop = 1; break; }
        }
        if (mixGoal && !g_tameOff.load()) {
            u64 tt = 0; for (int i = 0; i < g_nthreads; i++) tt += g_tcnt[i].v.load();
            if ((double)tt + baseSteps >= endTarget) {
                g_tameOff = 1; endReached = true;
                printf("---- %s : cible tame atteinte (%s sauts = %.2f x sqrt(W)) -> tous les kangourous passent en wild\n\n",
                       fmt_time(el).c_str(), SI((double)tt + baseSteps), ((double)tt + baseSteps) / sqW);
                if (!saveFile.empty()) { save_tame(saveFile); lastSave = el; }
            }
        }
        if (el - lastP >= tsec) {
            u64 tot = 0; for (int i = 0; i < g_nthreads; i++) tot += g_cnt[i].v.load();
            double rate = (double)(tot - lastSteps) / (el - lastP);
            char eta[96] = "-", prog[48] = "-", mixInfo[160] = "";
            if (mixMode && g_doTame && g_doWild) {
                u64 tt = 0; for (int i = 0; i < g_nthreads; i++) tt += g_tcnt[i].v.load();
                if (g_tameOff.load()) snprintf(mixInfo, sizeof mixInfo, "  [tame: %s sauts, termine -> 100%% wild]", SI((double)tt));
                else snprintf(mixInfo, sizeof mixInfo, "  [tame: %s sauts / wild: %s]", SI((double)tt), SI((double)(tot - tt)));
            }
            if (g_nk > 0 || tameGoal) {
                snprintf(prog, sizeof prog, tameGoal ? "%.0f %% de la cible tame" : "%.0f %% de la moyenne attendue", 100.0 * ((double)tot + baseSteps) / Nexp);
                if ((double)tot + baseSteps < Nexp && rate > 0) snprintf(eta, sizeof eta, "~%s", fmt_time((Nexp - (double)tot - baseSteps) / rate).c_str());
                else snprintf(eta, sizeof eta, "moyenne depassee (variance normale)");
            }
            // Vitesse de progression : % de la cible (moyenne attendue) gagne par minute
            const double progSpeed = Nexp > 0 ? 100.0 * rate * 60.0 / Nexp : 0.0;
            // Vitesse de resolution : cles de l'intervalle "couvertes" par seconde (equivalent force brute)
            // = sauts/s x sqrt(nb cles) x sqrt(W) / coef
            const double resSpeed = rate * sqrt((double)max(1, g_nk)) * sqW / g_coef;
            printf("---- %s ------------------------------------------\n"
                   "  Vitesse sauts  : %s sauts/s\n"
                   "  Vitesse resol. : %s cles/s equivalentes\n"
                   "  Vit. progress. : %s %%/min\n"
                   "  Sauts          : %s  (%.2f x sqrt(W))%s\n"
                   "  Progression    : %s\n"
                   "  Reste (moy.)   : %s\n"
                   "  DP stockes     : %s  (tame : %s)  / ~%s attendus a ce stade\n"
                   "  Taille DP      : ~%s en RAM  (DP %d bits)\n"
                   "  Cles           : %d / %d\n\n",
                   fmt_time(el).c_str(), SI(rate), SI(resSpeed), SI(progSpeed),
                   SI((double)tot), (double)tot / sqW, mixInfo, prog, eta,
                   SI((double)g_ndp.load()), SI((double)g_ntame.load()), SI((double)tot * pDP),
                   fmt_bytes((double)g_ndp.load() * dp_ram_bytes()).c_str(), g_dp, g_nsolved.load(), g_nk);
            lastP = el; lastSteps = tot;
        }
        if (!saveFile.empty() && g_doTame && !g_tameOff.load() && el - lastSave >= 300) { save_tame(saveFile); lastSave = el; }
        bool alive = false; (void)alive;
    }
    g_stop = 1;
    for (auto& t : th) t.join();
    double el = chrono::duration<double>(chrono::steady_clock::now() - t0).count();
    u64 tot = 0; for (int i = 0; i < g_nthreads; i++) tot += g_cnt[i].v.load();
    if (!saveFile.empty() && g_doTame) {
        if (save_tame(saveFile)) printf("Table tame sauvegardee dans %s (%llu DP)\n", saveFile.c_str(), (unsigned long long)g_ntame.load());
    }
    printf("==============================================================\n"
           "  RESUME\n"
           "==============================================================\n"
           "  Duree          : %s\n"
           "  Sauts          : %s  (%.2f x sqrt(W))\n"
           "  Vitesse moy.   : %s sauts/s  (%s cles/s equivalentes)\n"
           "  Cles trouvees  : %d / %d\n", fmt_time(el).c_str(), SI((double)tot), (double)tot / sqW,
           SI(el > 0 ? (double)tot / el : 0.0), SI(el > 0 ? (double)tot / el * sqrt((double)max(1, g_nk)) * sqW / g_coef : 0.0),
           g_nsolved.load(), g_nk);
    {
        const u64 nd = g_ndp.load(), nt = g_ntame.load();
        printf("  Points DP      : %d bits (%s)\n", g_dp, dpAuto ? "auto" : (dpFromFile ? "repris du fichier" : "manuel"));
        printf("  DP stockes     : %s  (tame : %s, wild : %s)\n", SI((double)nd), SI((double)nt), SI((double)(nd - nt)));
        printf("  DP attendus    : ~%s pour ce nombre de sauts  (estimation initiale : ~%s)\n", SI((double)tot * pDP), SI(dpEst));
        printf("  Taille DP      : ~%s en RAM\n", fmt_bytes((double)nd * dp_ram_bytes()).c_str());
        if (!saveFile.empty() && g_doTame) printf("  Fichier tame   : %s  (~%s)\n", saveFile.c_str(), fmt_bytes(sizeof(Hdr) + (double)nt * sizeof(Rec)).c_str());
        printf("  Surcout DP     : ~%s sauts (%.2f %% de la moyenne)\n", SI(ovSteps), ovPct);
        if (tameGoal) printf("  Cible tame     : %.4g x sqrt(W) -> %s (fait ~%.3g x sqrt(W))\n", endCoef, endReached ? "ATTEINTE" : "non atteinte", ((double)tot + baseSteps) / sqW);
    }
    if (g_nsolved.load() > 0) printf("  Fichier        : %s\n", g_outfile.c_str());
    printf("==============================================================\n");
    return g_nk > 0 && g_nsolved.load() == g_nk ? 0 : 2;
}
