// =============================================================================
//  kangaroo_gpu.cu  -  backend CUDA du kangourou de Pollard (secp256k1), v2 "haute vitesse"
//
//  Role inchange : le GPU ne produit QUE des points distingues (DP) bruts ; toute la
//  resolution (collisions, relations, table tame, sauvegarde) reste dans kangaroo.cpp.
//
//  Ce qui a change par rapport a la v1 (goulots corriges) :
//   1. Multiplication / carre 256 bits : limbes de 32 bits + chaines de retenue PTX
//      (mad.lo.cc / madc.hi.cc), carre dedie (36 produits au lieu de 64), reduction
//      speciale p = 2^256 - 2^32 - 977.  (v1 : unsigned __int128 emule = tres lent sur GPU.)
//      Un auto-test GPU compare ces fonctions a une implementation de reference simple
//      AVANT de lancer le calcul ; compilez avec -DKG_PORTABLE pour desactiver l'asm PTX.
//   2. Table de sauts : memoire globale lue via __ldg (cache L1), au lieu de __constant__
//      (une lecture constante a adresse DIVERGENTE est serialisee thread par thread).
//   3. Taille de lot Montgomery (-gbatch) = parametre de template (8/16/32/64) : les
//      tableaux locaux ont une taille connue du compilateur, et l'inversion (255 carres)
//      est amortie sur davantage de kangourous (16 -> 64 : ~-40 % de multiplications).
//   4. Historique de detection de cycle en 32 bits (moitie moins de memoire/trafic).
//   5. Nombre de blocs automatique (1 "vague" exacte via l'occupation reelle du noyau),
//      lancement asynchrone (gpu_worker_launch / gpu_worker_fetch) : le GPU calcule le
//      lot suivant pendant que le CPU insere les DP du lot precedent.
//   6. Plus de dependance a unsigned __int128 : fonctionne avec tout CUDA >= 10.
//
//  Compile uniquement avec nvcc (voir le Makefile).
// =============================================================================
#include <cuda_runtime.h>
#include <cstdio>
#include <cstring>
#include <cstdint>
#include <cmath>
#include "kangaroo_gpu.h"

typedef uint64_t u64;
typedef uint32_t u32;

#define DV __device__ __forceinline__
#define NJ 256
#define NE 64
#define HBUF 16
#define TAME 0xFFFFFFFFu
#define KG_TPB 128          // taille de bloc du noyau principal
#define KG_MINB 4           // blocs/SM vises (plafonne les registres a 128 sur 64K registres)

#ifndef KG_LAUNCH
#define KG_LAUNCH(k, g, b, ...) k<<<(g), (b)>>>(__VA_ARGS__)
#endif

// ====================== primitives a retenue (PTX ou portable) =================
#if defined(__CUDA_ARCH__) && !defined(KG_PORTABLE)
  #define KG_ASM 1
#endif
#ifdef KG_ASM
  #define KG_DECL_CF
  #define ADD_CC(r,a,b)       asm volatile("add.cc.u32 %0, %1, %2;"   : "=r"(r) : "r"(a), "r"(b))
  #define ADDC_CC(r,a,b)      asm volatile("addc.cc.u32 %0, %1, %2;"  : "=r"(r) : "r"(a), "r"(b))
  #define ADDC(r,a,b)         asm volatile("addc.u32 %0, %1, %2;"     : "=r"(r) : "r"(a), "r"(b))
  #define SUB_CC(r,a,b)       asm volatile("sub.cc.u32 %0, %1, %2;"   : "=r"(r) : "r"(a), "r"(b))
  #define SUBC_CC(r,a,b)      asm volatile("subc.cc.u32 %0, %1, %2;"  : "=r"(r) : "r"(a), "r"(b))
  #define SUBC(r,a,b)         asm volatile("subc.u32 %0, %1, %2;"     : "=r"(r) : "r"(a), "r"(b))
  #define MAD_LO_CC(r,a,b,c)  asm volatile("mad.lo.cc.u32 %0, %1, %2, %3;"  : "=r"(r) : "r"(a), "r"(b), "r"(c))
  #define MADC_LO_CC(r,a,b,c) asm volatile("madc.lo.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
  #define MAD_HI_CC(r,a,b,c)  asm volatile("mad.hi.cc.u32 %0, %1, %2, %3;"  : "=r"(r) : "r"(a), "r"(b), "r"(c))
  #define MADC_HI_CC(r,a,b,c) asm volatile("madc.hi.cc.u32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(c))
  #define MADC_HI(r,a,b,c)    asm volatile("madc.hi.u32 %0, %1, %2, %3;"    : "=r"(r) : "r"(a), "r"(b), "r"(c))
#else
  // version C pure : memes sequences, retenue dans une variable locale (sert aussi aux tests CPU)
  #define KG_DECL_CF u32 kg_cf = 0; (void)kg_cf;
  #define ADD_CC(r,a,b)  do { u64 s_ = (u64)(u32)(a) + (u32)(b); (r) = (u32)s_; kg_cf = (u32)(s_ >> 32); } while (0)
  #define ADDC_CC(r,a,b) do { u64 s_ = (u64)(u32)(a) + (u32)(b) + kg_cf; (r) = (u32)s_; kg_cf = (u32)(s_ >> 32); } while (0)
  #define ADDC(r,a,b)    do { u64 s_ = (u64)(u32)(a) + (u32)(b) + kg_cf; (r) = (u32)s_; } while (0)
  #define SUB_CC(r,a,b)  do { u32 a_ = (a), b_ = (b); (r) = a_ - b_; kg_cf = (a_ < b_) ? 1u : 0u; } while (0)
  #define SUBC_CC(r,a,b) do { u32 a_ = (a), b_ = (b); u64 t_ = (u64)b_ + kg_cf; (r) = (u32)((u64)a_ - t_); kg_cf = ((u64)a_ < t_) ? 1u : 0u; } while (0)
  #define SUBC(r,a,b)    do { u32 a_ = (a), b_ = (b); u64 t_ = (u64)b_ + kg_cf; (r) = (u32)((u64)a_ - t_); } while (0)
  #define MAD_LO_CC(r,a,b,c)  do { u64 p_ = (u64)(u32)(a) * (u32)(b); u64 s_ = (p_ & 0xFFFFFFFFull) + (u32)(c); (r) = (u32)s_; kg_cf = (u32)(s_ >> 32); } while (0)
  #define MADC_LO_CC(r,a,b,c) do { u64 p_ = (u64)(u32)(a) * (u32)(b); u64 s_ = (p_ & 0xFFFFFFFFull) + (u32)(c) + kg_cf; (r) = (u32)s_; kg_cf = (u32)(s_ >> 32); } while (0)
  #define MAD_HI_CC(r,a,b,c)  do { u64 p_ = (u64)(u32)(a) * (u32)(b); u64 s_ = (p_ >> 32) + (u32)(c); (r) = (u32)s_; kg_cf = (u32)(s_ >> 32); } while (0)
  #define MADC_HI_CC(r,a,b,c) do { u64 p_ = (u64)(u32)(a) * (u32)(b); u64 s_ = (p_ >> 32) + (u32)(c) + kg_cf; (r) = (u32)s_; kg_cf = (u32)(s_ >> 32); } while (0)
  #define MADC_HI(r,a,b,c)    do { u64 p_ = (u64)(u32)(a) * (u32)(b); u64 s_ = (p_ >> 32) + (u32)(c) + kg_cf; (r) = (u32)s_; } while (0)
#endif

// ============================ entiers 256 bits (8 x u32) =======================
// Meme disposition memoire que gu256_t (4 x u64 little-endian) : conversion directe.
struct u256 { u32 v[8]; };
typedef u256 fe;

#define P0_ 0xFFFFFC2Fu
DV u256 u_zero() { u256 r; for (int i = 0; i < 8; i++) r.v[i] = 0u; return r; }
DV u256 u_from32(u32 x) { u256 r = u_zero(); r.v[0] = x; return r; }
DV bool u_is_zero(const u256& a) { return (a.v[0] | a.v[1] | a.v[2] | a.v[3] | a.v[4] | a.v[5] | a.v[6] | a.v[7]) == 0u; }
DV bool u_equal(const u256& a, const u256& b) {
    return ((a.v[0] ^ b.v[0]) | (a.v[1] ^ b.v[1]) | (a.v[2] ^ b.v[2]) | (a.v[3] ^ b.v[3]) |
            (a.v[4] ^ b.v[4]) | (a.v[5] ^ b.v[5]) | (a.v[6] ^ b.v[6]) | (a.v[7] ^ b.v[7])) == 0u;
}
DV int u_cmp(const u256& a, const u256& b) { for (int i = 7; i >= 0; i--) if (a.v[i] != b.v[i]) return a.v[i] > b.v[i] ? 1 : -1; return 0; }
DV int clz32(u32 x) {
#ifdef __CUDA_ARCH__
    return x ? __clz((int)x) : 32;
#else
    return x ? __builtin_clz(x) : 32;
#endif
}
DV int u_bits(const u256& a) { for (int i = 7; i >= 0; i--) if (a.v[i]) return i * 32 + 32 - clz32(a.v[i]); return 0; }
DV void u_inc(u256& r) { for (int i = 0; i < 8; i++) { r.v[i] += 1u; if (r.v[i]) break; } }
DV void u_dec(u256& r) { for (int i = 0; i < 8; i++) { u32 old = r.v[i]; r.v[i] = old - 1u; if (old) break; } }
DV u256 u_shl1(const u256& a) { u256 r; for (int i = 7; i > 0; i--) r.v[i] = (a.v[i] << 1) | (a.v[i - 1] >> 31); r.v[0] = a.v[0] << 1; return r; }
DV u256 u_shr(const u256& a, int n) {      // n dans [0, 255]
    u256 r = u_zero(); int w = n >> 5, b = n & 31;
    for (int i = 0; i + w < 8; i++) {
        u32 x = a.v[i + w] >> b;
        if (b && i + w + 1 < 8) x |= a.v[i + w + 1] << (32 - b);
        r.v[i] = x;
    }
    return r;
}
DV u256 u_ones(int k) {                     // 2^k - 1, k dans [0, 256]
    u256 r = u_zero();
    for (int i = 0; i < 8; i++) { int lo = i * 32; if (k >= lo + 32) r.v[i] = 0xFFFFFFFFu; else if (k > lo) r.v[i] = (1u << (k - lo)) - 1u; }
    return r;
}
// a - b (modulo 2^256), sans controle d'emprunt (usage init uniquement)
DV u256 u_sub(const u256& a, const u256& b) {
    KG_DECL_CF
    u256 r;
    SUB_CC(r.v[0], a.v[0], b.v[0]);
    SUBC_CC(r.v[1], a.v[1], b.v[1]); SUBC_CC(r.v[2], a.v[2], b.v[2]); SUBC_CC(r.v[3], a.v[3], b.v[3]);
    SUBC_CC(r.v[4], a.v[4], b.v[4]); SUBC_CC(r.v[5], a.v[5], b.v[5]); SUBC_CC(r.v[6], a.v[6], b.v[6]);
    SUBC(r.v[7], a.v[7], b.v[7]);
    return r;
}
DV u256 u_add(const u256& a, const u256& b) {      // modulo 2^256
    KG_DECL_CF
    u256 r;
    ADD_CC(r.v[0], a.v[0], b.v[0]);
    ADDC_CC(r.v[1], a.v[1], b.v[1]); ADDC_CC(r.v[2], a.v[2], b.v[2]); ADDC_CC(r.v[3], a.v[3], b.v[3]);
    ADDC_CC(r.v[4], a.v[4], b.v[4]); ADDC_CC(r.v[5], a.v[5], b.v[5]); ADDC_CC(r.v[6], a.v[6], b.v[6]);
    ADDC(r.v[7], a.v[7], b.v[7]);
    return r;
}

// ============================ corps Fp : p = 2^256 - 2^32 - 977 ================
DV void fe_sub(fe& r, const fe& a, const fe& b) {
    KG_DECL_CF
    u32 r0,r1,r2,r3,r4,r5,r6,r7,m_;
    SUB_CC(r0, a.v[0], b.v[0]);
    SUBC_CC(r1, a.v[1], b.v[1]);
    SUBC_CC(r2, a.v[2], b.v[2]);
    SUBC_CC(r3, a.v[3], b.v[3]);
    SUBC_CC(r4, a.v[4], b.v[4]);
    SUBC_CC(r5, a.v[5], b.v[5]);
    SUBC_CC(r6, a.v[6], b.v[6]);
    SUBC_CC(r7, a.v[7], b.v[7]);
    SUBC(m_, 0u, 0u);
    SUB_CC(r0, r0, m_ & 0x3D1u);
    SUBC_CC(r1, r1, m_ & 1u);
    SUBC_CC(r2, r2, 0u);
    SUBC_CC(r3, r3, 0u);
    SUBC_CC(r4, r4, 0u);
    SUBC_CC(r5, r5, 0u);
    SUBC_CC(r6, r6, 0u);
    SUBC(r7, r7, 0u);
    r.v[0] = r0;
    r.v[1] = r1;
    r.v[2] = r2;
    r.v[3] = r3;
    r.v[4] = r4;
    r.v[5] = r5;
    r.v[6] = r6;
    r.v[7] = r7;
}
DV void fe_add(fe& r, const fe& a, const fe& b) {
    KG_DECL_CF
    u32 r0,r1,r2,r3,r4,r5,r6,r7,c_;
    ADD_CC(r0, a.v[0], b.v[0]);
    ADDC_CC(r1, a.v[1], b.v[1]);
    ADDC_CC(r2, a.v[2], b.v[2]);
    ADDC_CC(r3, a.v[3], b.v[3]);
    ADDC_CC(r4, a.v[4], b.v[4]);
    ADDC_CC(r5, a.v[5], b.v[5]);
    ADDC_CC(r6, a.v[6], b.v[6]);
    ADDC_CC(r7, a.v[7], b.v[7]);
    ADDC(c_, 0u, 0u);
    if (c_ || ((r2 & r3 & r4 & r5 & r6 & r7) == 0xFFFFFFFFu && (r1 == 0xFFFFFFFFu || (r1 == 0xFFFFFFFEu && r0 >= 0xFFFFFC2Fu)))) {
        ADD_CC(r0, r0, 0x3D1u);
        ADDC_CC(r1, r1, 1u);
        ADDC_CC(r2, r2, 0u);
        ADDC_CC(r3, r3, 0u);
        ADDC_CC(r4, r4, 0u);
        ADDC_CC(r5, r5, 0u);
        ADDC_CC(r6, r6, 0u);
        ADDC(r7, r7, 0u);
    }
    r.v[0] = r0;
    r.v[1] = r1;
    r.v[2] = r2;
    r.v[3] = r3;
    r.v[4] = r4;
    r.v[5] = r5;
    r.v[6] = r6;
    r.v[7] = r7;
}
DV void fe_neg(fe& r, const fe& a) {
    if (u_is_zero(a)) { r = a; return; }
    KG_DECL_CF
    u32 r0,r1,r2,r3,r4,r5,r6,r7;
    SUB_CC(r0, 0xFFFFFC2Fu, a.v[0]);
    SUBC_CC(r1, 0xFFFFFFFEu, a.v[1]);
    SUBC_CC(r2, 0xFFFFFFFFu, a.v[2]);
    SUBC_CC(r3, 0xFFFFFFFFu, a.v[3]);
    SUBC_CC(r4, 0xFFFFFFFFu, a.v[4]);
    SUBC_CC(r5, 0xFFFFFFFFu, a.v[5]);
    SUBC_CC(r6, 0xFFFFFFFFu, a.v[6]);
    SUBC(r7, 0xFFFFFFFFu, a.v[7]);
    r.v[0] = r0;
    r.v[1] = r1;
    r.v[2] = r2;
    r.v[3] = r3;
    r.v[4] = r4;
    r.v[5] = r5;
    r.v[6] = r6;
    r.v[7] = r7;
}
DV void fe_reduce(fe& r, const u32 t[16]) {
    KG_DECL_CF
    u32 x[10];
    MAD_LO_CC(x[0], t[8], 977u, t[0]);
    MADC_LO_CC(x[1], t[9], 977u, t[1]);
    MADC_LO_CC(x[2], t[10], 977u, t[2]);
    MADC_LO_CC(x[3], t[11], 977u, t[3]);
    MADC_LO_CC(x[4], t[12], 977u, t[4]);
    MADC_LO_CC(x[5], t[13], 977u, t[5]);
    MADC_LO_CC(x[6], t[14], 977u, t[6]);
    MADC_LO_CC(x[7], t[15], 977u, t[7]);
    ADDC(x[8], 0u, 0u);
    MAD_HI_CC(x[1], t[8], 977u, x[1]);
    MADC_HI_CC(x[2], t[9], 977u, x[2]);
    MADC_HI_CC(x[3], t[10], 977u, x[3]);
    MADC_HI_CC(x[4], t[11], 977u, x[4]);
    MADC_HI_CC(x[5], t[12], 977u, x[5]);
    MADC_HI_CC(x[6], t[13], 977u, x[6]);
    MADC_HI_CC(x[7], t[14], 977u, x[7]);
    MADC_HI_CC(x[8], t[15], 977u, x[8]);
    ADDC(x[9], 0u, 0u);
    ADD_CC(x[1], x[1], t[8]);
    ADDC_CC(x[2], x[2], t[9]);
    ADDC_CC(x[3], x[3], t[10]);
    ADDC_CC(x[4], x[4], t[11]);
    ADDC_CC(x[5], x[5], t[12]);
    ADDC_CC(x[6], x[6], t[13]);
    ADDC_CC(x[7], x[7], t[14]);
    ADDC_CC(x[8], x[8], t[15]);
    ADDC(x[9], x[9], 0u);
    u64 A_ = (u64)x[8] * 977u;
    u32 m0 = (u32)A_;
    u64 s_ = (A_ >> 32) + (u64)x[8] + (u64)x[9] * 977u;
    u32 m1 = (u32)s_, m2 = (u32)(s_ >> 32) + x[9];
    u32 r0, r1, r2, r3, r4, r5, r6, r7, c_;
    ADD_CC(r0, x[0], m0);
    ADDC_CC(r1, x[1], m1);
    ADDC_CC(r2, x[2], m2);
    ADDC_CC(r3, x[3], 0u);
    ADDC_CC(r4, x[4], 0u);
    ADDC_CC(r5, x[5], 0u);
    ADDC_CC(r6, x[6], 0u);
    ADDC_CC(r7, x[7], 0u);
    ADDC(c_, 0u, 0u);
    if (c_) {
        ADD_CC(r0, r0, 0x3D1u);
        ADDC_CC(r1, r1, 1u);
        ADDC_CC(r2, r2, 0u);
        ADDC_CC(r3, r3, 0u);
        ADDC_CC(r4, r4, 0u);
        ADDC_CC(r5, r5, 0u);
        ADDC_CC(r6, r6, 0u);
        ADDC(r7, r7, 0u);
    }
    if ((r2 & r3 & r4 & r5 & r6 & r7) == 0xFFFFFFFFu && (r1 == 0xFFFFFFFFu || (r1 == 0xFFFFFFFEu && r0 >= 0xFFFFFC2Fu))) {
        ADD_CC(r0, r0, 0x3D1u);
        ADDC_CC(r1, r1, 1u);
        ADDC_CC(r2, r2, 0u);
        ADDC_CC(r3, r3, 0u);
        ADDC_CC(r4, r4, 0u);
        ADDC_CC(r5, r5, 0u);
        ADDC_CC(r6, r6, 0u);
        ADDC(r7, r7, 0u);
    }
    r.v[0] = r0;
    r.v[1] = r1;
    r.v[2] = r2;
    r.v[3] = r3;
    r.v[4] = r4;
    r.v[5] = r5;
    r.v[6] = r6;
    r.v[7] = r7;
}
DV void fe_mul(fe& r, const fe& a, const fe& b) {
    KG_DECL_CF
    u32 t[16];
    t[0] = a.v[0] * b.v[0];
    MAD_HI_CC(t[1], a.v[0], b.v[0], a.v[0] * b.v[1]);
    MADC_HI_CC(t[2], a.v[0], b.v[1], a.v[0] * b.v[2]);
    MADC_HI_CC(t[3], a.v[0], b.v[2], a.v[0] * b.v[3]);
    MADC_HI_CC(t[4], a.v[0], b.v[3], a.v[0] * b.v[4]);
    MADC_HI_CC(t[5], a.v[0], b.v[4], a.v[0] * b.v[5]);
    MADC_HI_CC(t[6], a.v[0], b.v[5], a.v[0] * b.v[6]);
    MADC_HI_CC(t[7], a.v[0], b.v[6], a.v[0] * b.v[7]);
    MADC_HI(t[8], a.v[0], b.v[7], 0u);
    // ligne 1
    MAD_LO_CC(t[1], a.v[1], b.v[0], t[1]);
    MADC_LO_CC(t[2], a.v[1], b.v[1], t[2]);
    MADC_LO_CC(t[3], a.v[1], b.v[2], t[3]);
    MADC_LO_CC(t[4], a.v[1], b.v[3], t[4]);
    MADC_LO_CC(t[5], a.v[1], b.v[4], t[5]);
    MADC_LO_CC(t[6], a.v[1], b.v[5], t[6]);
    MADC_LO_CC(t[7], a.v[1], b.v[6], t[7]);
    MADC_LO_CC(t[8], a.v[1], b.v[7], t[8]);
    ADDC(t[9], 0u, 0u);
    MAD_HI_CC(t[2], a.v[1], b.v[0], t[2]);
    MADC_HI_CC(t[3], a.v[1], b.v[1], t[3]);
    MADC_HI_CC(t[4], a.v[1], b.v[2], t[4]);
    MADC_HI_CC(t[5], a.v[1], b.v[3], t[5]);
    MADC_HI_CC(t[6], a.v[1], b.v[4], t[6]);
    MADC_HI_CC(t[7], a.v[1], b.v[5], t[7]);
    MADC_HI_CC(t[8], a.v[1], b.v[6], t[8]);
    MADC_HI_CC(t[9], a.v[1], b.v[7], t[9]);
    ADDC(t[10], 0u, 0u);
    // ligne 2
    MAD_LO_CC(t[2], a.v[2], b.v[0], t[2]);
    MADC_LO_CC(t[3], a.v[2], b.v[1], t[3]);
    MADC_LO_CC(t[4], a.v[2], b.v[2], t[4]);
    MADC_LO_CC(t[5], a.v[2], b.v[3], t[5]);
    MADC_LO_CC(t[6], a.v[2], b.v[4], t[6]);
    MADC_LO_CC(t[7], a.v[2], b.v[5], t[7]);
    MADC_LO_CC(t[8], a.v[2], b.v[6], t[8]);
    MADC_LO_CC(t[9], a.v[2], b.v[7], t[9]);
    ADDC(t[10], t[10], 0u);
    MAD_HI_CC(t[3], a.v[2], b.v[0], t[3]);
    MADC_HI_CC(t[4], a.v[2], b.v[1], t[4]);
    MADC_HI_CC(t[5], a.v[2], b.v[2], t[5]);
    MADC_HI_CC(t[6], a.v[2], b.v[3], t[6]);
    MADC_HI_CC(t[7], a.v[2], b.v[4], t[7]);
    MADC_HI_CC(t[8], a.v[2], b.v[5], t[8]);
    MADC_HI_CC(t[9], a.v[2], b.v[6], t[9]);
    MADC_HI_CC(t[10], a.v[2], b.v[7], t[10]);
    ADDC(t[11], 0u, 0u);
    // ligne 3
    MAD_LO_CC(t[3], a.v[3], b.v[0], t[3]);
    MADC_LO_CC(t[4], a.v[3], b.v[1], t[4]);
    MADC_LO_CC(t[5], a.v[3], b.v[2], t[5]);
    MADC_LO_CC(t[6], a.v[3], b.v[3], t[6]);
    MADC_LO_CC(t[7], a.v[3], b.v[4], t[7]);
    MADC_LO_CC(t[8], a.v[3], b.v[5], t[8]);
    MADC_LO_CC(t[9], a.v[3], b.v[6], t[9]);
    MADC_LO_CC(t[10], a.v[3], b.v[7], t[10]);
    ADDC(t[11], t[11], 0u);
    MAD_HI_CC(t[4], a.v[3], b.v[0], t[4]);
    MADC_HI_CC(t[5], a.v[3], b.v[1], t[5]);
    MADC_HI_CC(t[6], a.v[3], b.v[2], t[6]);
    MADC_HI_CC(t[7], a.v[3], b.v[3], t[7]);
    MADC_HI_CC(t[8], a.v[3], b.v[4], t[8]);
    MADC_HI_CC(t[9], a.v[3], b.v[5], t[9]);
    MADC_HI_CC(t[10], a.v[3], b.v[6], t[10]);
    MADC_HI_CC(t[11], a.v[3], b.v[7], t[11]);
    ADDC(t[12], 0u, 0u);
    // ligne 4
    MAD_LO_CC(t[4], a.v[4], b.v[0], t[4]);
    MADC_LO_CC(t[5], a.v[4], b.v[1], t[5]);
    MADC_LO_CC(t[6], a.v[4], b.v[2], t[6]);
    MADC_LO_CC(t[7], a.v[4], b.v[3], t[7]);
    MADC_LO_CC(t[8], a.v[4], b.v[4], t[8]);
    MADC_LO_CC(t[9], a.v[4], b.v[5], t[9]);
    MADC_LO_CC(t[10], a.v[4], b.v[6], t[10]);
    MADC_LO_CC(t[11], a.v[4], b.v[7], t[11]);
    ADDC(t[12], t[12], 0u);
    MAD_HI_CC(t[5], a.v[4], b.v[0], t[5]);
    MADC_HI_CC(t[6], a.v[4], b.v[1], t[6]);
    MADC_HI_CC(t[7], a.v[4], b.v[2], t[7]);
    MADC_HI_CC(t[8], a.v[4], b.v[3], t[8]);
    MADC_HI_CC(t[9], a.v[4], b.v[4], t[9]);
    MADC_HI_CC(t[10], a.v[4], b.v[5], t[10]);
    MADC_HI_CC(t[11], a.v[4], b.v[6], t[11]);
    MADC_HI_CC(t[12], a.v[4], b.v[7], t[12]);
    ADDC(t[13], 0u, 0u);
    // ligne 5
    MAD_LO_CC(t[5], a.v[5], b.v[0], t[5]);
    MADC_LO_CC(t[6], a.v[5], b.v[1], t[6]);
    MADC_LO_CC(t[7], a.v[5], b.v[2], t[7]);
    MADC_LO_CC(t[8], a.v[5], b.v[3], t[8]);
    MADC_LO_CC(t[9], a.v[5], b.v[4], t[9]);
    MADC_LO_CC(t[10], a.v[5], b.v[5], t[10]);
    MADC_LO_CC(t[11], a.v[5], b.v[6], t[11]);
    MADC_LO_CC(t[12], a.v[5], b.v[7], t[12]);
    ADDC(t[13], t[13], 0u);
    MAD_HI_CC(t[6], a.v[5], b.v[0], t[6]);
    MADC_HI_CC(t[7], a.v[5], b.v[1], t[7]);
    MADC_HI_CC(t[8], a.v[5], b.v[2], t[8]);
    MADC_HI_CC(t[9], a.v[5], b.v[3], t[9]);
    MADC_HI_CC(t[10], a.v[5], b.v[4], t[10]);
    MADC_HI_CC(t[11], a.v[5], b.v[5], t[11]);
    MADC_HI_CC(t[12], a.v[5], b.v[6], t[12]);
    MADC_HI_CC(t[13], a.v[5], b.v[7], t[13]);
    ADDC(t[14], 0u, 0u);
    // ligne 6
    MAD_LO_CC(t[6], a.v[6], b.v[0], t[6]);
    MADC_LO_CC(t[7], a.v[6], b.v[1], t[7]);
    MADC_LO_CC(t[8], a.v[6], b.v[2], t[8]);
    MADC_LO_CC(t[9], a.v[6], b.v[3], t[9]);
    MADC_LO_CC(t[10], a.v[6], b.v[4], t[10]);
    MADC_LO_CC(t[11], a.v[6], b.v[5], t[11]);
    MADC_LO_CC(t[12], a.v[6], b.v[6], t[12]);
    MADC_LO_CC(t[13], a.v[6], b.v[7], t[13]);
    ADDC(t[14], t[14], 0u);
    MAD_HI_CC(t[7], a.v[6], b.v[0], t[7]);
    MADC_HI_CC(t[8], a.v[6], b.v[1], t[8]);
    MADC_HI_CC(t[9], a.v[6], b.v[2], t[9]);
    MADC_HI_CC(t[10], a.v[6], b.v[3], t[10]);
    MADC_HI_CC(t[11], a.v[6], b.v[4], t[11]);
    MADC_HI_CC(t[12], a.v[6], b.v[5], t[12]);
    MADC_HI_CC(t[13], a.v[6], b.v[6], t[13]);
    MADC_HI_CC(t[14], a.v[6], b.v[7], t[14]);
    ADDC(t[15], 0u, 0u);
    // ligne 7
    MAD_LO_CC(t[7], a.v[7], b.v[0], t[7]);
    MADC_LO_CC(t[8], a.v[7], b.v[1], t[8]);
    MADC_LO_CC(t[9], a.v[7], b.v[2], t[9]);
    MADC_LO_CC(t[10], a.v[7], b.v[3], t[10]);
    MADC_LO_CC(t[11], a.v[7], b.v[4], t[11]);
    MADC_LO_CC(t[12], a.v[7], b.v[5], t[12]);
    MADC_LO_CC(t[13], a.v[7], b.v[6], t[13]);
    MADC_LO_CC(t[14], a.v[7], b.v[7], t[14]);
    ADDC(t[15], t[15], 0u);
    MAD_HI_CC(t[8], a.v[7], b.v[0], t[8]);
    MADC_HI_CC(t[9], a.v[7], b.v[1], t[9]);
    MADC_HI_CC(t[10], a.v[7], b.v[2], t[10]);
    MADC_HI_CC(t[11], a.v[7], b.v[3], t[11]);
    MADC_HI_CC(t[12], a.v[7], b.v[4], t[12]);
    MADC_HI_CC(t[13], a.v[7], b.v[5], t[13]);
    MADC_HI_CC(t[14], a.v[7], b.v[6], t[14]);
    MADC_HI(t[15], a.v[7], b.v[7], t[15]);
    fe_reduce(r, t);
}
DV void fe_sqr(fe& r, const fe& a) {
    KG_DECL_CF
    u32 t[16];
    for (int z = 0; z < 16; z++) t[z] = 0u;
    // ligne croisee 0
    MAD_LO_CC(t[1], a.v[0], a.v[1], t[1]);
    MADC_LO_CC(t[2], a.v[0], a.v[2], t[2]);
    MADC_LO_CC(t[3], a.v[0], a.v[3], t[3]);
    MADC_LO_CC(t[4], a.v[0], a.v[4], t[4]);
    MADC_LO_CC(t[5], a.v[0], a.v[5], t[5]);
    MADC_LO_CC(t[6], a.v[0], a.v[6], t[6]);
    MADC_LO_CC(t[7], a.v[0], a.v[7], t[7]);
    ADDC(t[8], t[8], 0u);
    MAD_HI_CC(t[2], a.v[0], a.v[1], t[2]);
    MADC_HI_CC(t[3], a.v[0], a.v[2], t[3]);
    MADC_HI_CC(t[4], a.v[0], a.v[3], t[4]);
    MADC_HI_CC(t[5], a.v[0], a.v[4], t[5]);
    MADC_HI_CC(t[6], a.v[0], a.v[5], t[6]);
    MADC_HI_CC(t[7], a.v[0], a.v[6], t[7]);
    MADC_HI_CC(t[8], a.v[0], a.v[7], t[8]);
    ADDC(t[9], t[9], 0u);
    // ligne croisee 1
    MAD_LO_CC(t[3], a.v[1], a.v[2], t[3]);
    MADC_LO_CC(t[4], a.v[1], a.v[3], t[4]);
    MADC_LO_CC(t[5], a.v[1], a.v[4], t[5]);
    MADC_LO_CC(t[6], a.v[1], a.v[5], t[6]);
    MADC_LO_CC(t[7], a.v[1], a.v[6], t[7]);
    MADC_LO_CC(t[8], a.v[1], a.v[7], t[8]);
    ADDC(t[9], t[9], 0u);
    MAD_HI_CC(t[4], a.v[1], a.v[2], t[4]);
    MADC_HI_CC(t[5], a.v[1], a.v[3], t[5]);
    MADC_HI_CC(t[6], a.v[1], a.v[4], t[6]);
    MADC_HI_CC(t[7], a.v[1], a.v[5], t[7]);
    MADC_HI_CC(t[8], a.v[1], a.v[6], t[8]);
    MADC_HI_CC(t[9], a.v[1], a.v[7], t[9]);
    ADDC(t[10], t[10], 0u);
    // ligne croisee 2
    MAD_LO_CC(t[5], a.v[2], a.v[3], t[5]);
    MADC_LO_CC(t[6], a.v[2], a.v[4], t[6]);
    MADC_LO_CC(t[7], a.v[2], a.v[5], t[7]);
    MADC_LO_CC(t[8], a.v[2], a.v[6], t[8]);
    MADC_LO_CC(t[9], a.v[2], a.v[7], t[9]);
    ADDC(t[10], t[10], 0u);
    MAD_HI_CC(t[6], a.v[2], a.v[3], t[6]);
    MADC_HI_CC(t[7], a.v[2], a.v[4], t[7]);
    MADC_HI_CC(t[8], a.v[2], a.v[5], t[8]);
    MADC_HI_CC(t[9], a.v[2], a.v[6], t[9]);
    MADC_HI_CC(t[10], a.v[2], a.v[7], t[10]);
    ADDC(t[11], t[11], 0u);
    // ligne croisee 3
    MAD_LO_CC(t[7], a.v[3], a.v[4], t[7]);
    MADC_LO_CC(t[8], a.v[3], a.v[5], t[8]);
    MADC_LO_CC(t[9], a.v[3], a.v[6], t[9]);
    MADC_LO_CC(t[10], a.v[3], a.v[7], t[10]);
    ADDC(t[11], t[11], 0u);
    MAD_HI_CC(t[8], a.v[3], a.v[4], t[8]);
    MADC_HI_CC(t[9], a.v[3], a.v[5], t[9]);
    MADC_HI_CC(t[10], a.v[3], a.v[6], t[10]);
    MADC_HI_CC(t[11], a.v[3], a.v[7], t[11]);
    ADDC(t[12], t[12], 0u);
    // ligne croisee 4
    MAD_LO_CC(t[9], a.v[4], a.v[5], t[9]);
    MADC_LO_CC(t[10], a.v[4], a.v[6], t[10]);
    MADC_LO_CC(t[11], a.v[4], a.v[7], t[11]);
    ADDC(t[12], t[12], 0u);
    MAD_HI_CC(t[10], a.v[4], a.v[5], t[10]);
    MADC_HI_CC(t[11], a.v[4], a.v[6], t[11]);
    MADC_HI_CC(t[12], a.v[4], a.v[7], t[12]);
    ADDC(t[13], t[13], 0u);
    // ligne croisee 5
    MAD_LO_CC(t[11], a.v[5], a.v[6], t[11]);
    MADC_LO_CC(t[12], a.v[5], a.v[7], t[12]);
    ADDC(t[13], t[13], 0u);
    MAD_HI_CC(t[12], a.v[5], a.v[6], t[12]);
    MADC_HI_CC(t[13], a.v[5], a.v[7], t[13]);
    ADDC(t[14], t[14], 0u);
    // ligne croisee 6
    MAD_LO_CC(t[13], a.v[6], a.v[7], t[13]);
    ADDC(t[14], t[14], 0u);
    MAD_HI_CC(t[14], a.v[6], a.v[7], t[14]);
    ADDC(t[15], t[15], 0u);
    // doublement
    ADD_CC(t[1], t[1], t[1]);
    ADDC_CC(t[2], t[2], t[2]);
    ADDC_CC(t[3], t[3], t[3]);
    ADDC_CC(t[4], t[4], t[4]);
    ADDC_CC(t[5], t[5], t[5]);
    ADDC_CC(t[6], t[6], t[6]);
    ADDC_CC(t[7], t[7], t[7]);
    ADDC_CC(t[8], t[8], t[8]);
    ADDC_CC(t[9], t[9], t[9]);
    ADDC_CC(t[10], t[10], t[10]);
    ADDC_CC(t[11], t[11], t[11]);
    ADDC_CC(t[12], t[12], t[12]);
    ADDC_CC(t[13], t[13], t[13]);
    ADDC_CC(t[14], t[14], t[14]);
    ADDC(t[15], t[15], t[15]);
    // carres a_i^2
    MAD_LO_CC(t[0], a.v[0], a.v[0], t[0]);
    MADC_HI_CC(t[1], a.v[0], a.v[0], t[1]);
    MADC_LO_CC(t[2], a.v[1], a.v[1], t[2]);
    MADC_HI_CC(t[3], a.v[1], a.v[1], t[3]);
    MADC_LO_CC(t[4], a.v[2], a.v[2], t[4]);
    MADC_HI_CC(t[5], a.v[2], a.v[2], t[5]);
    MADC_LO_CC(t[6], a.v[3], a.v[3], t[6]);
    MADC_HI_CC(t[7], a.v[3], a.v[3], t[7]);
    MADC_LO_CC(t[8], a.v[4], a.v[4], t[8]);
    MADC_HI_CC(t[9], a.v[4], a.v[4], t[9]);
    MADC_LO_CC(t[10], a.v[5], a.v[5], t[10]);
    MADC_HI_CC(t[11], a.v[5], a.v[5], t[11]);
    MADC_LO_CC(t[12], a.v[6], a.v[6], t[12]);
    MADC_HI_CC(t[13], a.v[6], a.v[6], t[13]);
    MADC_LO_CC(t[14], a.v[7], a.v[7], t[14]);
    MADC_HI(t[15], a.v[7], a.v[7], t[15]);
    fe_reduce(r, t);
}

// versions non inlinees (utilisees hors boucle chaude : inversion, init, courbe) -> code compact
__device__ __noinline__ void nmul(fe& r, const fe& a, const fe& b) { fe_mul(r, a, b); }
__device__ __noinline__ void nsqr(fe& r, const fe& a) { fe_sqr(r, a); }
__device__ __noinline__ void nsqrn(fe& r, const fe& a, int n) { fe t = a; for (int i = 0; i < n; i++) fe_sqr(t, t); r = t; }

__device__ __noinline__ void fe_inv(fe& r, const fe& a) {          // a^(p-2), chaine d'addition de libsecp256k1
    fe x2, x3, x6, x9, x11, x22, x44, x88, x176, x220, x223, t;
    nsqr(x2, a);          nmul(x2, x2, a);
    nsqr(x3, x2);         nmul(x3, x3, a);
    nsqrn(x6, x3, 3);     nmul(x6, x6, x3);
    nsqrn(x9, x6, 3);     nmul(x9, x9, x3);
    nsqrn(x11, x9, 2);    nmul(x11, x11, x2);
    nsqrn(x22, x11, 11);  nmul(x22, x22, x11);
    nsqrn(x44, x22, 22);  nmul(x44, x44, x22);
    nsqrn(x88, x44, 44);  nmul(x88, x88, x44);
    nsqrn(x176, x88, 88); nmul(x176, x176, x88);
    nsqrn(x220, x176, 44);nmul(x220, x220, x44);
    nsqrn(x223, x220, 3); nmul(x223, x223, x3);
    nsqrn(t, x223, 23);   nmul(t, t, x22);
    nsqrn(t, t, 5);       nmul(t, t, a);
    nsqrn(t, t, 3);       nmul(t, t, x2);
    nsqrn(t, t, 2);       nmul(r, t, a);
}

// ============================ scalaires modulo n ===============================
// n (ordre de secp256k1) est code en dur dans les chaines ci-dessous ; gpu_worker_create
// verifie que la valeur envoyee par l'hote est identique.
DV void sc_add(u256& r, const u256& a, const u256& b) {    // (a + b) mod n, a et b dans [0, n)
    KG_DECL_CF
    u32 s0,s1,s2,s3,s4,s5,s6,s7,t0,t1,t2,t3,t4,t5,t6,t7,c_,bm_;
    ADD_CC(s0, a.v[0], b.v[0]);
    ADDC_CC(s1, a.v[1], b.v[1]);
    ADDC_CC(s2, a.v[2], b.v[2]);
    ADDC_CC(s3, a.v[3], b.v[3]);
    ADDC_CC(s4, a.v[4], b.v[4]);
    ADDC_CC(s5, a.v[5], b.v[5]);
    ADDC_CC(s6, a.v[6], b.v[6]);
    ADDC_CC(s7, a.v[7], b.v[7]);
    ADDC(c_, 0u, 0u);
    SUB_CC(t0, s0, 0xD0364141u);
    SUBC_CC(t1, s1, 0xBFD25E8Cu);
    SUBC_CC(t2, s2, 0xAF48A03Bu);
    SUBC_CC(t3, s3, 0xBAAEDCE6u);
    SUBC_CC(t4, s4, 0xFFFFFFFEu);
    SUBC_CC(t5, s5, 0xFFFFFFFFu);
    SUBC_CC(t6, s6, 0xFFFFFFFFu);
    SUBC_CC(t7, s7, 0xFFFFFFFFu);
    SUBC(bm_, 0u, 0u);
    const bool use_t = (c_ != 0u) || (bm_ == 0u);
    r.v[0] = use_t ? t0 : s0;
    r.v[1] = use_t ? t1 : s1;
    r.v[2] = use_t ? t2 : s2;
    r.v[3] = use_t ? t3 : s3;
    r.v[4] = use_t ? t4 : s4;
    r.v[5] = use_t ? t5 : s5;
    r.v[6] = use_t ? t6 : s6;
    r.v[7] = use_t ? t7 : s7;
}
DV void sc_neg(u256& r, const u256& a) {                   // (-a) mod n
    if (u_is_zero(a)) { r = a; return; }
    KG_DECL_CF
    u32 r0,r1,r2,r3,r4,r5,r6,r7;
    SUB_CC(r0, 0xD0364141u, a.v[0]);
    SUBC_CC(r1, 0xBFD25E8Cu, a.v[1]);
    SUBC_CC(r2, 0xAF48A03Bu, a.v[2]);
    SUBC_CC(r3, 0xBAAEDCE6u, a.v[3]);
    SUBC_CC(r4, 0xFFFFFFFEu, a.v[4]);
    SUBC_CC(r5, 0xFFFFFFFFu, a.v[5]);
    SUBC_CC(r6, 0xFFFFFFFFu, a.v[6]);
    SUBC(r7, 0xFFFFFFFFu, a.v[7]);
    r.v[0] = r0;
    r.v[1] = r1;
    r.v[2] = r2;
    r.v[3] = r3;
    r.v[4] = r4;
    r.v[5] = r5;
    r.v[6] = r6;
    r.v[7] = r7;
}
__device__ __constant__ u32 c_nlimbs[8] = {0xD0364141u,0xBFD25E8Cu,0xAF48A03Bu,0xBAAEDCE6u,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
__device__ __constant__ u32 c_nhalf_l[8] = {0x681B20A0u,0xDFE92F46u,0x57A4501Du,0x5D576E73u,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0x7FFFFFFFu};
DV u256 n_val() { u256 r; for (int i = 0; i < 8; i++) r.v[i] = c_nlimbs[i]; return r; }
DV u256 nhalf_val() { u256 r; for (int i = 0; i < 8; i++) r.v[i] = c_nhalf_l[i]; return r; }

// ================================== courbe =====================================
struct Jac { fe X, Y, Z; };
struct GPt { fe x, y; bool inf; };
DV Jac jac_inf() { Jac r; r.X = u_from32(1); r.Y = u_from32(1); r.Z = u_zero(); return r; }
__device__ __noinline__ void jac_dbl(Jac& r, const Jac& a) {
    if (u_is_zero(a.Z) || u_is_zero(a.Y)) { r = jac_inf(); return; }
    fe A, B, C, D, E, F, t, X3, Y3, Z3, C8;
    nsqr(A, a.X); nsqr(B, a.Y); nsqr(C, B);
    fe_add(t, a.X, B); nsqr(t, t); fe_sub(t, t, A); fe_sub(t, t, C); fe_add(D, t, t);
    fe_add(E, A, A); fe_add(E, E, A); nsqr(F, E);
    fe_sub(X3, F, D); fe_sub(X3, X3, D);
    fe_sub(t, D, X3); nmul(Y3, E, t);
    fe_add(C8, C, C); fe_add(C8, C8, C8); fe_add(C8, C8, C8);
    fe_sub(Y3, Y3, C8);
    nmul(Z3, a.Y, a.Z); fe_add(Z3, Z3, Z3);
    r.X = X3; r.Y = Y3; r.Z = Z3;
}
__device__ __noinline__ void jac_add(Jac& r, const Jac& a, const Jac& b) {
    if (u_is_zero(a.Z)) { r = b; return; }
    if (u_is_zero(b.Z)) { r = a; return; }
    fe Z1Z1, Z2Z2, U1, U2, S1, S2, H, R, t;
    nsqr(Z1Z1, a.Z); nsqr(Z2Z2, b.Z);
    nmul(U1, a.X, Z2Z2); nmul(U2, b.X, Z1Z1);
    nmul(t, b.Z, Z2Z2); nmul(S1, a.Y, t);
    nmul(t, a.Z, Z1Z1); nmul(S2, b.Y, t);
    fe_sub(H, U2, U1); fe_sub(R, S2, S1);
    if (u_is_zero(H)) { if (u_is_zero(R)) { jac_dbl(r, a); return; } r = jac_inf(); return; }
    fe H2, H3, V, X3, Y3, Z3;
    nsqr(H2, H); nmul(H3, H, H2); nmul(V, U1, H2);
    nsqr(X3, R); fe_sub(X3, X3, H3); fe_sub(X3, X3, V); fe_sub(X3, X3, V);
    fe_sub(t, V, X3); nmul(Y3, R, t); nmul(t, S1, H3); fe_sub(Y3, Y3, t);
    nmul(Z3, a.Z, b.Z); nmul(Z3, Z3, H);
    r.X = X3; r.Y = Y3; r.Z = Z3;
}
__device__ __noinline__ GPt to_aff(const Jac& j) {
    GPt p; p.inf = false;
    if (u_is_zero(j.Z)) { p.inf = true; p.x = u_zero(); p.y = u_zero(); return p; }
    fe zi, z2, z3; fe_inv(zi, j.Z); nsqr(z2, zi); nmul(z3, z2, zi);
    nmul(p.x, j.X, z2); nmul(p.y, j.Y, z3);
    return p;
}
DV Jac jac_from(const GPt& p) { if (p.inf) return jac_inf(); Jac r; r.X = p.x; r.Y = p.y; r.Z = u_from32(1); return r; }
__device__ __noinline__ GPt pt_mul(const u256& k, const GPt& base) {
    Jac R = jac_inf(), B = jac_from(base);
    int nb = u_bits(k);
    for (int i = nb - 1; i >= 0; i--) { jac_dbl(R, R); if ((k.v[i >> 5] >> (i & 31)) & 1u) jac_add(R, R, B); }
    return to_aff(R);
}
__device__ __noinline__ GPt pt_add(const GPt& a, const GPt& b) { Jac r; jac_add(r, jac_from(a), jac_from(b)); return to_aff(r); }
DV GPt pt_neg(const GPt& p) { GPt r = p; if (!p.inf) fe_neg(r.y, p.y); return r; }

__device__ __constant__ u32 c_Gx[8] = {0x16F81798u,0x59F2815Bu,0x2DCE28D9u,0x029BFCDBu,0xCE870B07u,0x55A06295u,0xF9DCBBACu,0x79BE667Eu};
__device__ __constant__ u32 c_Gy[8] = {0xFB10D4B8u,0x9C47D08Fu,0xA6855419u,0xFD17B448u,0x0E1108A8u,0x5DA4FBFCu,0x26A3C465u,0x483ADA77u};
DV GPt GEN_pt() { GPt g; for (int i = 0; i < 8; i++) { g.x.v[i] = c_Gx[i]; g.y.v[i] = c_Gy[i]; } g.inf = false; return g; }
__device__ __noinline__ GPt mulG(const u256& k) { return pt_mul(k, GEN_pt()); }
__device__ __noinline__ GPt mulG_mod(const u256& k) {      // k dans [0, n) ; k > n/2 = scalaire negatif
    if (u_cmp(k, nhalf_val()) > 0) return pt_neg(mulG(u_sub(n_val(), k)));
    return mulG(k);
}

// ---- generateur xoshiro256** (meme que cote CPU) ----
DV u64 rotl64(u64 x, int k) { return (x << k) | (x >> (64 - k)); }
DV void rng_seed(u64 s[4], u64 seed) {
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL; u64 z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        s[i] = z ^ (z >> 31);
    }
}
DV u64 rng_next(u64 s[4]) {
    u64 r = rotl64(s[1] * 5, 7) * 9, t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t; s[3] = rotl64(s[3], 45);
    return r;
}
DV u256 rng_below(u64 s[4], const u256& n) {
    if (u_is_zero(n)) return u_zero();
    u256 m1 = n; u_dec(m1);
    u256 mask = u_ones(u_bits(m1));
    for (;;) {
        u256 r;
        for (int i = 0; i < 4; i++) { u64 w = rng_next(s); r.v[2 * i] = (u32)w & mask.v[2 * i]; r.v[2 * i + 1] = (u32)(w >> 32) & mask.v[2 * i + 1]; }
        if (u_cmp(r, n) < 0) return r;
    }
}

// =============================== etat GPU =====================================
__device__ __constant__ int c_sym, c_dp_bits, c_sl, c_do_tame, c_do_wild, c_nk, c_tame_off;
__device__ __constant__ double c_tfrac;
__device__ __constant__ u64 c_dpmask;
__device__ __constant__ u64 c_rotLimit;            // rotation forcee des kangourous wild (0 = desactivee)
__device__ __constant__ u32 c_w_l[8];               // largeur de l'intervalle W (limbes 32 bits)

// Etat persistant d'UN kangourou (192 octets, aligne 16 pour des acces vectoriels).
struct __align__(16) KReg {
    u256 x, y, d;          // point (symetrise : y pair), distance mod n
    u32 id, sn, age, hp;   // TAME/indice de cle, pas depuis le dernier DP, age, curseur d'historique
    int s;                 // signe +-1 (0 = "gare" : aucune cle disponible)
    u32 pad[3];
    u32 h[HBUF];           // historique (32 bits bas de x) pour la detection de cycles courts
};

DV bool roleTame(int j) {
    if (!c_do_tame || c_tame_off) return false;
    if (!c_do_wild) return true;
    double tf = c_tfrac;
    return (long)floor((j + 1) * tf) > (long)floor(j * tf);
}

__device__ u64 d_keyctr;
__device__ int device_pick_key(const uint8_t* solved, int nk) {   // round-robin global, comme pick_key() cote CPU
    if (nk <= 0) return -1;
    for (int t = 0; t < nk; t++) {
        u64 v = atomicAdd((unsigned long long*)&d_keyctr, 1ULL);
        int i = (int)(v % (u64)nk);
        if (!solved[i]) return i;
    }
    return -1;
}
DV u256 from_g(const gu256_t& g) { u256 r; for (int i = 0; i < 4; i++) { r.v[2 * i] = (u32)g.d[i]; r.v[2 * i + 1] = (u32)(g.d[i] >> 32); } return r; }
DV gu256_t to_g(const u256& a) { gu256_t r; for (int i = 0; i < 4; i++) r.d[i] = (u64)a.v[2 * i] | ((u64)a.v[2 * i + 1] << 32); return r; }

// initialise (ou reinitialise) le kangourou j. false si aucune cle disponible : il est "gare" (s = 0).
__device__ __noinline__ bool init_one(u64 rs[4], KReg& k, int j, const gpt_t* keys, const uint8_t* solved, int nk) {
    bool tame = roleTame(j);
    u256 L;
    if (c_sl >= 255) L = u_from32(1);
    else { u256 w; for (int i = 0; i < 8; i++) w.v[i] = c_w_l[i]; L = u_shr(w, c_sl); }
    if (u_is_zero(L)) L = u_from32(1);
    u256 L2p1 = u_shl1(L); u_inc(L2p1);
    for (int tries = 0; tries < 64; tries++) {
        u256 x = rng_below(rs, L2p1);
        u256 r;
        if (u_cmp(x, L) >= 0) r = u_sub(x, L); else sc_neg(r, u_sub(L, x));
        if (u_is_zero(r)) r = u_from32(1);
        GPt P; int id = -1;
        if (tame) P = mulG_mod(r);
        else {
            id = device_pick_key(solved, nk);
            if (id < 0) { k.s = 0; return false; }
            GPt Qi; Qi.x = from_g(keys[id].x); Qi.y = from_g(keys[id].y); Qi.inf = false;
            P = pt_add(Qi, mulG_mod(r));
        }
        if (P.inf) continue;
        k.x = P.x; k.y = P.y; k.d = r; k.id = tame ? TAME : (u32)id; k.s = 1;
        if (c_sym && (k.y.v[0] & 1u)) { fe_neg(k.y, k.y); sc_neg(k.d, k.d); k.s = -1; }
        for (int h = 0; h < HBUF; h++) k.h[h] = 0u;
        k.hp = 0; k.h[k.hp++ & (HBUF - 1)] = k.x.v[0];
        k.sn = 0; k.age = 0;
        return true;
    }
    k.s = 0; return false;     // intervalle degenere : abandonne ce kangourou pour cet appel
}

// ---- chargement d'une entree de la table de sauts (x, y, d : 3 x 32 octets) via le cache read-only ----
DV void ld_fe(fe& r, const u32* p) {
    const uint4* q = (const uint4*)p;
    uint4 a = __ldg(q), b = __ldg(q + 1);
    r.v[0] = a.x; r.v[1] = a.y; r.v[2] = a.z; r.v[3] = a.w; r.v[4] = b.x; r.v[5] = b.y; r.v[6] = b.z; r.v[7] = b.w;
}
#define JT_X(jt, i) ((jt) + (size_t)(i) * 24)
#define JT_Y(jt, i) ((jt) + (size_t)(i) * 24 + 8)
#define JT_D(jt, i) ((jt) + (size_t)(i) * 24 + 16)

// addition lente (dx == 0 : doublement ou P = -J), identique a slow_add() cote CPU. Renvoie l'indice de saut utilise.
__device__ __noinline__ u32 slow_add(fe& x, fe& y, u32 ji, const u32* jt) {
    fe jx, jy, lam, t, x3, y3, inv;
    ld_fe(jx, JT_X(jt, ji)); ld_fe(jy, JT_Y(jt, ji));
    if (u_equal(x, jx)) {
        if (u_equal(y, jy)) {
            fe n, d; nsqr(n, x); fe_add(t, n, n); fe_add(n, t, n);
            fe_add(d, y, y); fe_inv(inv, d); nmul(lam, n, inv);
        } else {
            ji = NJ;                                            // saut d'evasion n.0 (cf g_E[0] cote CPU)
            ld_fe(jx, JT_X(jt, ji)); ld_fe(jy, JT_Y(jt, ji));
            fe dx; fe_sub(dx, jx, x); fe_inv(inv, dx);
            fe_sub(t, jy, y); nmul(lam, t, inv);
        }
    } else {
        fe dx; fe_sub(dx, jx, x); fe_inv(inv, dx);
        fe_sub(t, jy, y); nmul(lam, t, inv);
    }
    fe l2; nsqr(l2, lam);
    fe_sub(x3, l2, x); fe_sub(x3, x3, jx);
    fe_sub(t, x, x3); nmul(y3, lam, t); fe_sub(y3, y3, y);
    x = x3; y = y3;
    return ji;
}

DV void emit_dp(gdprec_t* outBuf, int outCap, unsigned int* outCount, const KReg& k) {
    unsigned int slot = atomicAdd(outCount, 1u);
    if (slot < (unsigned)outCap) {
        gdprec_t r;
        r.x0 = (u64)k.x.v[0] | ((u64)k.x.v[1] << 32);
        r.x1 = (u64)k.x.v[2] | ((u64)k.x.v[3] << 32);
        r.d = to_g(k.d); r.id = k.id; r.s = (int8_t)k.s;
        outBuf[slot] = r;
    }
}

// =============================== noyau principal ==============================
template <int GB>
__global__ void __launch_bounds__(KG_TPB, KG_MINB)
kang_kernel(KReg* __restrict__ st, u64* __restrict__ rngst, int N, int iters,
            const gpt_t* keys, const uint8_t* solved, int nk, const u32* __restrict__ jt,
            gdprec_t* outBuf, int outCap, unsigned int* outCount,
            unsigned long long* stepsAccum, unsigned long long* tameAccum, u32 dpLimit) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    KReg* mine = st + (size_t)tid * GB;

    u64 rs[4]; for (int k = 0; k < 4; k++) rs[k] = rngst[(size_t)tid * 4 + k];
    KReg K[GB];
    fe DX[GB], PRE[GB];
    u32 JI[GB];                       // indice du saut choisi ; bit 16 = addition lente (dx == 0)
    const fe ONE = u_from32(1);

    for (int j = 0; j < GB; j++) {
        K[j] = mine[j];
        if (K[j].s == 0)              // kangourou "gare" (pas de cle dispo au dernier appel) : on retente
            init_one(rs, K[j], j, keys, solved, nk);
    }

    u64 localSteps = 0, localTame = 0;
    #pragma unroll 1
    for (int it = 0; it < iters; it++) {
        // ---- phase 1 : choix du saut, dx = xJ - x, produits prefixes (Montgomery)
        #pragma unroll 1
        for (int j = 0; j < GB; j++) {
            KReg& k = K[j];
            if (k.s == 0) { DX[j] = ONE; JI[j] = 0x10000u; if (j) PRE[j] = PRE[j - 1]; else PRE[j] = ONE; continue; }
            u32 ji = 0xFFFFFFFFu;
            if (c_sym) {
                const u32 pos = k.hp, xl = k.x.v[0];
                #pragma unroll
                for (int Lc = 2; Lc <= 8; Lc += 2) {
                    if (k.h[(pos - 1u - (u32)Lc) & (HBUF - 1)] == xl) {
                        u32 mn = xl;
                        for (int q = 1; q < Lc; q++) { u32 v = k.h[(pos - 1u - (u32)q) & (HBUF - 1)]; if (v < mn) mn = v; }
                        ji = NJ + (mn >> 7) % NE;
                        for (int h = 0; h < HBUF; h++) k.h[h] = 0u;
                        k.hp = 0; k.h[k.hp++ & (HBUF - 1)] = xl;
                        break;
                    }
                }
            }
            if (ji == 0xFFFFFFFFu) ji = k.x.v[2] & (NJ - 1);
            fe jx; ld_fe(jx, JT_X(jt, ji));
            fe_sub(DX[j], jx, k.x);
            if (u_is_zero(DX[j])) { DX[j] = ONE; ji |= 0x10000u; }
            JI[j] = ji;
            if (j) fe_mul(PRE[j], PRE[j - 1], DX[j]); else PRE[j] = DX[j];
        }
        // ---- phase 2 : UNE inversion pour tout le lot, puis remontee
        fe inv; fe_inv(inv, PRE[GB - 1]);
        #pragma unroll 1
        for (int j = GB - 1; j > 0; j--) { fe t; fe_mul(t, inv, PRE[j - 1]); fe_mul(inv, inv, DX[j]); DX[j] = t; }
        DX[0] = inv;
        // ---- phase 3 : addition affine + distance/symetrie + DP
        u32 tameStep = 0;
        #pragma unroll 1
        for (int j = 0; j < GB; j++) {
            KReg& k = K[j];
            if (k.s == 0) continue;
            u32 ji = JI[j] & 0xFFFFu;
            if (JI[j] & 0x10000u) ji = slow_add(k.x, k.y, ji, jt);
            else {
                fe jx, jy, t, lam, l2, x3, y3;
                ld_fe(jx, JT_X(jt, ji)); ld_fe(jy, JT_Y(jt, ji));
                fe_sub(t, jy, k.y); fe_mul(lam, t, DX[j]);
                fe_sqr(l2, lam);
                fe_sub(x3, l2, k.x); fe_sub(x3, x3, jx);
                fe_sub(t, k.x, x3); fe_mul(y3, lam, t); fe_sub(y3, y3, k.y);
                k.x = x3; k.y = y3;
            }
            { fe jd; ld_fe(jd, JT_D(jt, ji)); sc_add(k.d, k.d, jd); }
            if (c_sym && (k.y.v[0] & 1u)) { fe_neg(k.y, k.y); sc_neg(k.d, k.d); k.s = -k.s; }
            k.h[k.hp++ & (HBUF - 1)] = k.x.v[0];
            k.sn++; k.age++;
            if (k.id == TAME) tameStep++;
            const u64 xlo = (u64)k.x.v[0] | ((u64)k.x.v[1] << 32);
            if ((xlo & c_dpmask) == 0) {
                emit_dp(outBuf, outCap, outCount, k);
                k.sn = 0;
            } else if (k.sn > dpLimit) {
                init_one(rs, k, j, keys, solved, nk);          // bloque dans un cycle long : on repart
            }
        }
        localSteps += GB; localTame += tameStep;
        if ((it & 63) == 0) {
            #pragma unroll 1
            for (int j = 0; j < GB; j++) {
                KReg& k = K[j];
                if (k.s == 0) continue;
                if (k.id == TAME) { if (c_tame_off) init_one(rs, k, j, keys, solved, nk); }
                else if ((k.id < (u32)nk && solved[k.id]) || (c_rotLimit && k.age > c_rotLimit))
                    init_one(rs, k, j, keys, solved, nk);
            }
        }
    }

    for (int j = 0; j < GB; j++) mine[j] = K[j];
    for (int k = 0; k < 4; k++) rngst[(size_t)tid * 4 + k] = rs[k];
    if (localSteps) atomicAdd(stepsAccum, (unsigned long long)localSteps);
    if (localTame)  atomicAdd(tameAccum,  (unsigned long long)localTame);
}

// init : un lancement par indice j (plusieurs courts lancements plutot qu'un seul tres long : evite tout watchdog)
__global__ void init_kernel(KReg* st, u64* rngst, int N, int GB, int j, const gpt_t* keys, const uint8_t* solved, int nk, u64 seedBase) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= N) return;
    u64 rs[4];
    if (j == 0) rng_seed(rs, seedBase + (u64)tid * 0x9E3779B97F4A7C15ULL + 7919ULL);
    else for (int k = 0; k < 4; k++) rs[k] = rngst[(size_t)tid * 4 + k];
    KReg kr;
    for (int i = 0; i < 8; i++) { kr.x.v[i] = 0; kr.y.v[i] = 0; kr.d.v[i] = 0; }
    kr.id = 0; kr.sn = 0; kr.age = 0; kr.hp = 0; kr.s = 0; kr.pad[0] = kr.pad[1] = kr.pad[2] = 0;
    for (int h = 0; h < HBUF; h++) kr.h[h] = 0;
    init_one(rs, kr, j, keys, solved, nk);
    st[(size_t)tid * GB + j] = kr;
    for (int k = 0; k < 4; k++) rngst[(size_t)tid * 4 + k] = rs[k];
}

// ================================ auto-test GPU ================================
// Reference volontairement simple (boucles u64, aucune retenue PTX) pour valider la version rapide.
DV void ref_mul512(u32 t[16], const fe& a, const fe& b) {
    for (int i = 0; i < 16; i++) t[i] = 0;
    for (int i = 0; i < 8; i++) {
        u64 carry = 0;
        for (int j = 0; j < 8; j++) { u64 cur = (u64)a.v[i] * b.v[j] + t[i + j] + carry; t[i + j] = (u32)cur; carry = cur >> 32; }
        t[i + 8] = (u32)carry;
    }
}
DV bool ref_ge_p(const u32 r[8]) {
    for (int i = 7; i >= 2; i--) if (r[i] != 0xFFFFFFFFu) return false;
    if (r[1] != 0xFFFFFFFFu) { if (r[1] != 0xFFFFFFFEu) return false; return r[0] >= P0_; }
    return true;
}
DV void ref_reduce(fe& out, const u32 t[16]) {
    u32 w[18]; for (int i = 0; i < 16; i++) w[i] = t[i]; w[16] = w[17] = 0;
    for (int iter = 0; iter < 6; iter++) {
        u32 nw[18]; u64 carry = 0;
        for (int i = 0; i < 18; i++) {
            u64 cur = (i < 8 ? w[i] : 0u) + carry;
            if (i < 10) cur += (u64)w[8 + i] * 977u;
            if (i >= 1 && i - 1 < 10) cur += w[8 + i - 1];
            nw[i] = (u32)cur; carry = cur >> 32;
        }
        for (int i = 0; i < 18; i++) w[i] = nw[i];
    }
    u32 r[8]; for (int i = 0; i < 8; i++) r[i] = w[i];
    while (ref_ge_p(r)) {       // r -= p  <=>  r += 2^32 + 977 (mod 2^256)
        u64 c = (u64)r[0] + 977u; r[0] = (u32)c; c >>= 32;
        c += (u64)r[1] + 1u; r[1] = (u32)c; c >>= 32;
        for (int i = 2; i < 8; i++) { c += r[i]; r[i] = (u32)c; c >>= 32; }
    }
    for (int i = 0; i < 8; i++) out.v[i] = r[i];
}
DV void ref_mul(fe& r, const fe& a, const fe& b) { u32 t[16]; ref_mul512(t, a, b); ref_reduce(r, t); }
__device__ __constant__ u32 c_p_l[8] = {0xFFFFFC2Fu,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
__device__ __constant__ u32 c_x2g[8] = {0x5C709EE5u,0xABAC09B9u,0x8CEF3CA7u,0x5C778E4Bu,0x95C07CD8u,0x3045406Eu,0x41ED7D6Du,0xC6047F94u};
DV void ref_addsub(fe& r, const fe& a, const fe& b, bool sub) {
    u32 t[16]; for (int i = 0; i < 16; i++) t[i] = 0;
    if (!sub) { u64 c = 0; for (int i = 0; i < 8; i++) { c += (u64)a.v[i] + b.v[i]; t[i] = (u32)c; c >>= 32; } t[8] = (u32)c; ref_reduce(r, t); return; }
    u64 br = 0; u32 d[8];
    for (int i = 0; i < 8; i++) { u64 x = (u64)a.v[i] - b.v[i] - br; d[i] = (u32)x; br = (x >> 63) & 1u; }
    if (br) { u64 c = 0; for (int i = 0; i < 8; i++) { c += (u64)d[i] + c_p_l[i]; d[i] = (u32)c; c >>= 32; } }
    for (int i = 0; i < 8; i++) r.v[i] = d[i];
}
DV bool ref_scalar_ge_n(const u32 r[8]) {
    for (int i = 7; i >= 0; i--) { u32 n = c_nlimbs[i]; if (r[i] != n) return r[i] > n; }
    return true;
}
DV void ref_sc_add(u256& out, const u256& a, const u256& b) {
    u32 r[8]; u64 c = 0;
    for (int i = 0; i < 8; i++) { c += (u64)a.v[i] + b.v[i]; r[i] = (u32)c; c >>= 32; }
    if (c || ref_scalar_ge_n(r)) { u64 br = 0; for (int i = 0; i < 8; i++) { u64 x = (u64)r[i] - c_nlimbs[i] - br; r[i] = (u32)x; br = (x >> 63) & 1u; } }
    for (int i = 0; i < 8; i++) out.v[i] = r[i];
}
DV fe st_rand(u64 s[4], int kind) {
    fe r;
    for (int i = 0; i < 4; i++) { u64 w = rng_next(s); r.v[2 * i] = (u32)w; r.v[2 * i + 1] = (u32)(w >> 32); }
    if (kind == 1) { for (int i = 0; i < 8; i++) r.v[i] = (rng_next(s) & 1) ? 0xFFFFFFFFu : 0u; }      // beaucoup de 0 / 0xFFFFFFFF
    else if (kind == 2) { for (int i = 2; i < 8; i++) r.v[i] = 0xFFFFFFFFu; r.v[1] = (rng_next(s) & 1) ? 0xFFFFFFFFu : 0xFFFFFFFEu; }  // proche de p
    else if (kind == 3) { for (int i = 3; i < 8; i++) r.v[i] = 0u; }                                    // petit
    // ramene dans [0, p) : reduction par soustraction de p tant que necessaire
    u32 t[8]; for (int i = 0; i < 8; i++) t[i] = r.v[i];
    while (ref_ge_p(t)) { u64 c = (u64)t[0] + 977u; t[0] = (u32)c; c >>= 32; c += (u64)t[1] + 1u; t[1] = (u32)c; c >>= 32; for (int i = 2; i < 8; i++) { c += t[i]; t[i] = (u32)c; c >>= 32; } }
    for (int i = 0; i < 8; i++) r.v[i] = t[i];
    return r;
}
__global__ void selftest_kernel(int* fails, int* firstBad, int reps) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    u64 s[4]; rng_seed(s, 0x1234567ULL + tid * 7919ULL);
    int bad = 0, which = 0;
    fe zero = u_zero(), one = u_from32(1), pm1; for (int i = 2; i < 8; i++) pm1.v[i] = 0xFFFFFFFFu; pm1.v[1] = 0xFFFFFFFEu; pm1.v[0] = 0xFFFFFC2Eu;
    for (int rep = 0; rep < reps; rep++) {
        fe a, b, r1, r2;
        int kind = rep & 3;
        a = st_rand(s, kind); b = st_rand(s, (rep >> 2) & 3);
        if (rep == 0) { a = pm1; b = pm1; } if (rep == 1) { a = zero; b = pm1; } if (rep == 2) { a = one; b = pm1; }
        fe_mul(r1, a, b); ref_mul(r2, a, b); if (!u_equal(r1, r2)) { bad++; which = 1; }
        fe_sqr(r1, a);    ref_mul(r2, a, a); if (!u_equal(r1, r2)) { bad++; which = 2; }
        fe_add(r1, a, b); ref_addsub(r2, a, b, false); if (!u_equal(r1, r2)) { bad++; which = 3; }
        fe_sub(r1, a, b); ref_addsub(r2, a, b, true);  if (!u_equal(r1, r2)) { bad++; which = 4; }
        fe_neg(r1, a);    ref_addsub(r2, zero, a, true); if (!u_equal(r1, r2)) { bad++; which = 5; }
        // scalaires modulo n : on reutilise des valeurs < n (a tronque a 255 bits, b idem)
        u256 sa = a, sb = b; sa.v[7] &= 0x7FFFFFFFu; sb.v[7] &= 0x7FFFFFFFu;
        if (rep & 8) { sa = n_val(); u_dec(sa); }     // n - 1
        sc_add(r1, sa, sb); ref_sc_add(r2, sa, sb); if (!u_equal(r1, r2)) { bad++; which = 6; }
        u256 nn; sc_neg(nn, sa); ref_sc_add(r2, nn, sa); if (!u_is_zero(r2)) { bad++; which = 7; }
        if ((rep & 15) == 0 && !u_is_zero(a)) {   // inversion
            fe iv, pr; fe_inv(iv, a); fe_mul(pr, a, iv); if (!u_equal(pr, one)) { bad++; which = 8; }
        }
    }
    if (tid == 0) {     // courbe : 2G, 3G = 2G + G, (n-1)G = -G
        u256 two = u_from32(2), three = u_from32(3);
        GPt g2 = mulG(two);
        for (int i = 0; i < 8; i++) if (g2.x.v[i] != c_x2g[i]) { bad++; which = 9; }
        GPt g3 = pt_add(g2, GEN_pt()), g3b = mulG(three);
        if (!u_equal(g3.x, g3b.x) || !u_equal(g3.y, g3b.y)) { bad++; which = 10; }
        u256 nm1 = n_val(); u_dec(nm1);
        GPt q = mulG(nm1), gn = pt_neg(GEN_pt());
        if (!u_equal(q.x, gn.x) || !u_equal(q.y, gn.y)) { bad++; which = 11; }
    }
    if (bad) { atomicAdd(fails, bad); atomicExch(firstBad, which); }
}

// ============================ cote hote (API C) ================================
struct GpuHandle {
    int device;
    int N, GB, iters, outCap, nk, capThisCall;
    KReg* st; u64* rng; u32* d_jt;
    gpt_t* d_keys; uint8_t* d_solved;
    gdprec_t* d_out; unsigned int* d_outCount;
    unsigned long long *d_steps, *d_tame;
    u32 dpLimit;
    bool pending;
};

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA erreur (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); return nullptr; } } while (0)
#define CKV(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA erreur (%s:%d): %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); } } while (0)

static bool gb_supported(int gb) { return gb == 8 || gb == 16 || gb == 32 || gb == 64; }

static void launch_kang(GpuHandle* h) {
    const int blocks = (h->N + KG_TPB - 1) / KG_TPB;
    #define KARGS h->st, h->rng, h->N, h->iters, h->d_keys, h->d_solved, h->nk, h->d_jt, h->d_out, h->capThisCall, h->d_outCount, h->d_steps, h->d_tame, h->dpLimit
    switch (h->GB) {
        case 8:  KG_LAUNCH(kang_kernel<8>,  blocks, KG_TPB, KARGS); break;
        case 16: KG_LAUNCH(kang_kernel<16>, blocks, KG_TPB, KARGS); break;
        case 32: KG_LAUNCH(kang_kernel<32>, blocks, KG_TPB, KARGS); break;
        default: KG_LAUNCH(kang_kernel<64>, blocks, KG_TPB, KARGS); break;
    }
    #undef KARGS
}

extern "C" int gpu_device_count(void) {
    int n = 0; cudaError_t e = cudaGetDeviceCount(&n);
    return (e == cudaSuccess) ? n : 0;
}
extern "C" void gpu_device_name(int device_id, char* buf, int buflen) {
    if (!buf || buflen <= 0) return;
    cudaDeviceProp p;
    if (cudaGetDeviceProperties(&p, device_id) == cudaSuccess)
        snprintf(buf, buflen, "%s (%d SM, %.1f Go)", p.name, p.multiProcessorCount, p.totalGlobalMem / 1073741824.0);
    else snprintf(buf, buflen, "GPU %d", device_id);
}

// Nombre de blocs (au sens de -gblocks, avec threads_per_block threads chacun) qui remplit EXACTEMENT une
// "vague" de blocs residents sur la carte pour le noyau choisi. Renvoie 0 si indetermine.
extern "C" int gpu_auto_blocks(int device_id, int threads_per_block, int kang_per_thread) {
    if (!gb_supported(kang_per_thread) || threads_per_block < 1) return 0;
    if (cudaSetDevice(device_id) != cudaSuccess) return 0;
    cudaDeviceProp p; if (cudaGetDeviceProperties(&p, device_id) != cudaSuccess) return 0;
    int nb = 0; cudaError_t e;
    switch (kang_per_thread) {
        case 8:  e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kang_kernel<8>,  KG_TPB, 0); break;
        case 16: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kang_kernel<16>, KG_TPB, 0); break;
        case 32: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kang_kernel<32>, KG_TPB, 0); break;
        default: e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&nb, kang_kernel<64>, KG_TPB, 0); break;
    }
    if (e != cudaSuccess || nb < 1) nb = 1;
    long threads = (long)p.multiProcessorCount * nb * KG_TPB;
    return (int)((threads + threads_per_block - 1) / threads_per_block);
}

static bool g_selftest_done[64] = {false};
static bool run_selftest(int device) {
    if (device >= 0 && device < 64 && g_selftest_done[device]) return true;
    int *d_f = nullptr, *d_w = nullptr; int h_f = 0, h_w = 0;
    if (cudaMalloc(&d_f, sizeof(int)) != cudaSuccess || cudaMalloc(&d_w, sizeof(int)) != cudaSuccess) return false;
    cudaMemset(d_f, 0, sizeof(int)); cudaMemset(d_w, 0, sizeof(int));
    KG_LAUNCH(selftest_kernel, 2, 64, d_f, d_w, 256);
    cudaError_t e = cudaDeviceSynchronize();
    cudaMemcpy(&h_f, d_f, sizeof(int), cudaMemcpyDeviceToHost); cudaMemcpy(&h_w, d_w, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_f); cudaFree(d_w);
    if (e != cudaSuccess) { fprintf(stderr, "kangaroo_gpu: auto-test : erreur noyau : %s\n", cudaGetErrorString(e)); return false; }
    if (h_f != 0) {
        fprintf(stderr, "kangaroo_gpu: AUTO-TEST ARITHMETIQUE ECHOUE (%d ecarts, premier type %d : 1=mul 2=carre 3=add 4=sub 5=neg 6=mod-n 7=neg-n 8=inv 9-11=courbe).\n"
                        "               Recompilez avec -DKG_PORTABLE (make portable) et signalez le probleme.\n", h_f, h_w);
        return false;
    }
    if (device >= 0 && device < 64) g_selftest_done[device] = true;
    return true;
}

extern "C" void* gpu_worker_create(const gpu_init_t* cfg) {
    if (cfg->nj != NJ || cfg->ne != NE) {
        fprintf(stderr, "kangaroo_gpu: table de sauts incompatible (nj=%d ne=%d attendu %d/%d)\n", cfg->nj, cfg->ne, NJ, NE);
        return nullptr;
    }
    if (!gb_supported(cfg->kang_per_thread)) {
        fprintf(stderr, "kangaroo_gpu: -gbatch doit valoir 8, 16, 32 ou 64\n");
        return nullptr;
    }
    static const u32 NL[8] = {0xD0364141u,0xBFD25E8Cu,0xAF48A03Bu,0xBAAEDCE6u,0xFFFFFFFEu,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu};
    if (memcmp(&cfg->secp_n, NL, 32) != 0) { fprintf(stderr, "kangaroo_gpu: ordre de courbe inattendu (seul secp256k1 est gere).\n"); return nullptr; }
    CK(cudaSetDevice(cfg->device_id));
    if (!run_selftest(cfg->device_id)) return nullptr;

    GpuHandle* h = new GpuHandle();
    memset(h, 0, sizeof(*h));
    h->device = cfg->device_id;
    h->N = cfg->blocks * cfg->threads_per_block;
    h->GB = cfg->kang_per_thread;
    h->iters = cfg->iters_per_launch < 1 ? 1 : cfg->iters_per_launch;
    h->outCap = cfg->out_capacity < 1 ? 1 : cfg->out_capacity;
    h->nk = cfg->nk;
    h->capThisCall = h->outCap;
    const size_t total = (size_t)h->N * h->GB;

    CK(cudaMalloc(&h->st, total * sizeof(KReg)));
    CK(cudaMalloc(&h->rng, (size_t)h->N * 4 * sizeof(u64)));
    CK(cudaMalloc(&h->d_jt, (NJ + NE) * 96));
    CK(cudaMemcpy(h->d_jt, cfg->jumps, (NJ + NE) * 96, cudaMemcpyHostToDevice));     // gjump_t = x,y,dist = 3 x 32 octets

    CK(cudaMalloc(&h->d_keys, (cfg->nk > 0 ? cfg->nk : 1) * sizeof(gpt_t)));
    if (cfg->nk > 0) CK(cudaMemcpy(h->d_keys, cfg->keys, cfg->nk * sizeof(gpt_t), cudaMemcpyHostToDevice));
    CK(cudaMalloc(&h->d_solved, cfg->nk > 0 ? cfg->nk : 1));
    CK(cudaMemset(h->d_solved, 0, cfg->nk > 0 ? cfg->nk : 1));
    CK(cudaMalloc(&h->d_out, (size_t)h->outCap * sizeof(gdprec_t)));
    CK(cudaMalloc(&h->d_outCount, sizeof(unsigned int)));
    CK(cudaMalloc(&h->d_steps, sizeof(unsigned long long)));
    CK(cudaMalloc(&h->d_tame, sizeof(unsigned long long)));

    int sym = cfg->sym, dpb = cfg->dp_bits, sl = cfg->sl, dt = cfg->do_tame, dw = cfg->do_wild, nk = cfg->nk, toff = 0;
    double tf = cfg->tfrac;
    u64 dpmask = (dpb >= 64) ? ~0ULL : ((1ULL << dpb) - 1);
    CK(cudaMemcpyToSymbol(c_sym, &sym, sizeof(int)));        CK(cudaMemcpyToSymbol(c_dp_bits, &dpb, sizeof(int)));
    CK(cudaMemcpyToSymbol(c_sl, &sl, sizeof(int)));          CK(cudaMemcpyToSymbol(c_do_tame, &dt, sizeof(int)));
    CK(cudaMemcpyToSymbol(c_do_wild, &dw, sizeof(int)));     CK(cudaMemcpyToSymbol(c_nk, &nk, sizeof(int)));
    CK(cudaMemcpyToSymbol(c_tame_off, &toff, sizeof(int)));  CK(cudaMemcpyToSymbol(c_tfrac, &tf, sizeof(double)));
    CK(cudaMemcpyToSymbol(c_dpmask, &dpmask, sizeof(u64)));
    CK(cudaMemcpyToSymbol(c_rotLimit, &cfg->rot_limit, sizeof(u64)));
    CK(cudaMemcpyToSymbol(c_w_l, &cfg->w, 32));
    {   // floor(n/2) attendu = c_nhalf_l ; on verifie la valeur envoyee par l'hote
        static const u32 NH[8] = {0x681B20A0u,0xDFE92F46u,0x57A4501Du,0x5D576E73u,0xFFFFFFFFu,0xFFFFFFFFu,0xFFFFFFFFu,0x7FFFFFFFu};
        if (memcmp(&cfg->nhalf, NH, 32) != 0) { fprintf(stderr, "kangaroo_gpu: floor(n/2) inattendu.\n"); return nullptr; }
    }
    h->dpLimit = (u32)((u64)0xFFFFFFF0ULL < (((u64)64 << (dpb < 24 ? dpb : 24)) + 4096) ? (u64)0xFFFFFFF0ULL : (((u64)64 << (dpb < 24 ? dpb : 24)) + 4096));
    u64 kc0 = 0; CK(cudaMemcpyToSymbol(d_keyctr, &kc0, sizeof(u64)));

    switch (h->GB) {
        case 8:  CKV(cudaFuncSetCacheConfig(kang_kernel<8>,  cudaFuncCachePreferL1)); break;
        case 16: CKV(cudaFuncSetCacheConfig(kang_kernel<16>, cudaFuncCachePreferL1)); break;
        case 32: CKV(cudaFuncSetCacheConfig(kang_kernel<32>, cudaFuncCachePreferL1)); break;
        default: CKV(cudaFuncSetCacheConfig(kang_kernel<64>, cudaFuncCachePreferL1)); break;
    }
    const int ib = (h->N + 127) / 128;
    for (int j = 0; j < h->GB; j++)
        KG_LAUNCH(init_kernel, ib, 128, h->st, h->rng, h->N, h->GB, j, h->d_keys, h->d_solved, h->nk, cfg->seed);
    CK(cudaDeviceSynchronize());
    return h;
}

extern "C" void gpu_worker_sync(void* hv, const uint8_t* solved, int nk, int tame_off) {
    GpuHandle* h = (GpuHandle*)hv;
    cudaSetDevice(h->device);
    if (nk > 0) CKV(cudaMemcpy(h->d_solved, solved, nk * sizeof(uint8_t), cudaMemcpyHostToDevice));
    CKV(cudaMemcpyToSymbol(c_tame_off, &tame_off, sizeof(int)));
}

// Lancement ASYNCHRONE d'un lot (iters_per_launch iterations) : rend la main immediatement.
extern "C" int gpu_worker_launch(void* hv, int out_capacity) {
    GpuHandle* h = (GpuHandle*)hv;
    cudaSetDevice(h->device);
    h->capThisCall = (out_capacity < h->outCap && out_capacity > 0) ? out_capacity : h->outCap;
    CKV(cudaMemset(h->d_outCount, 0, sizeof(unsigned int)));
    CKV(cudaMemset(h->d_steps, 0, sizeof(unsigned long long)));
    CKV(cudaMemset(h->d_tame, 0, sizeof(unsigned long long)));
    launch_kang(h);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { fprintf(stderr, "kangaroo_gpu: echec de lancement du noyau: %s\n", cudaGetErrorString(e)); return -1; }
    h->pending = true;
    return 0;
}

// Attend la fin du lot en cours et recupere ses DP (jusqu'a out_capacity) + les compteurs de sauts.
extern "C" int gpu_worker_fetch(void* hv, gdprec_t* out, int out_capacity, uint64_t* steps, uint64_t* tame_steps) {
    GpuHandle* h = (GpuHandle*)hv;
    cudaSetDevice(h->device);
    if (!h->pending) return 0;
    h->pending = false;
    cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { fprintf(stderr, "kangaroo_gpu: erreur noyau: %s\n", cudaGetErrorString(e)); return 0; }
    unsigned int cnt = 0; CKV(cudaMemcpy(&cnt, h->d_outCount, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    int cap = out_capacity < h->capThisCall ? out_capacity : h->capThisCall;
    int n = (int)(cnt < (unsigned)cap ? cnt : (unsigned)cap);
    if (n > 0) CKV(cudaMemcpy(out, h->d_out, (size_t)n * sizeof(gdprec_t), cudaMemcpyDeviceToHost));
    unsigned long long s = 0, t = 0;
    CKV(cudaMemcpy(&s, h->d_steps, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CKV(cudaMemcpy(&t, h->d_tame, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    if (steps) *steps = s;
    if (tame_steps) *tame_steps = t;
    return n;
}

// Version synchrone (compatibilite) : lancement + attente.
extern "C" int gpu_worker_run(void* hv, gdprec_t* out, int out_capacity, uint64_t* steps, uint64_t* tame_steps) {
    if (gpu_worker_launch(hv, out_capacity) != 0) return 0;
    return gpu_worker_fetch(hv, out, out_capacity, steps, tame_steps);
}

extern "C" void gpu_worker_destroy(void* hv) {
    GpuHandle* h = (GpuHandle*)hv;
    if (!h) return;
    cudaSetDevice(h->device);
    cudaDeviceSynchronize();
    cudaFree(h->st); cudaFree(h->rng); cudaFree(h->d_jt);
    cudaFree(h->d_keys); cudaFree(h->d_solved); cudaFree(h->d_out); cudaFree(h->d_outCount);
    cudaFree(h->d_steps); cudaFree(h->d_tame);
    delete h;
}
