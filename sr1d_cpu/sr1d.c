#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define ADIABATIC_GAMMA (5.0 / 3.0)
#define min2(a, b) (a) < (b) ? (a) : (b)
#define max2(a, b) (a) > (b) ? (a) : (b)

#ifdef SINGLE
typedef float real;
#define square_root sqrtf
#define absolute_value fabsf
#else
typedef double real;
#define square_root sqrt
#define absolute_value fabs
#endif

void conserved_to_primitive(const real *cons, real *prim)
{
    const real newton_iter_max = 50;
    const real error_tolerance = 1e-12 * cons[0];
    const real gm              = ADIABATIC_GAMMA;
    const real m               = cons[0];
    const real tau             = cons[3];
    const real ss              = cons[1] * cons[1] + cons[2] * cons[2];
    int iteration              = 0;
    real p                     = prim[3];
    real w0;

    while (1) {
        const real et = tau + p + m;
        const real b2 = min2(ss / et / et, 1.0 - 1e-10);
        const real w2 = 1.0 / (1.0 - b2);
        const real w  = square_root(w2);
        const real e  = (tau + m * (1.0 - w) + p * (1.0 - w2)) / (m * w);
        const real d  = m / w;
        const real h  = 1.0 + e + p / d;
        const real a2 = gm * p / (d * h);
        const real f  = d * e * (gm - 1.0) - p;
        const real g  = b2 * a2 - 1.0;

        p -= f / g;

        if (absolute_value(f) < error_tolerance || iteration == newton_iter_max) {
            w0 = w;
            break;
        }
        iteration += 1;
    }

    prim[0] = m / w0;
    prim[1] = w0 * cons[1] / (tau + m + p);
    prim[2] = w0 * cons[2] / (tau + m + p);
    prim[3] = p;
}

real primitive_to_gamma_beta_squared(const real *prim)
{
    const real u1 = prim[1];
    const real u2 = prim[2];
    return u1 * u1 + u2 * u2;
}

real primitive_to_lorentz_factor(const real *prim)
{
    return square_root(1.0 + primitive_to_gamma_beta_squared(prim));
}

real primitive_to_gamma_beta_component(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

real primitive_to_beta_component(const real *prim, int direction)
{
    const real w = primitive_to_lorentz_factor(prim);

    switch (direction)
    {
        case 0: return prim[1] / w;
        case 1: return prim[2] / w;
        default: return 0.0;
    }
}

real primitive_to_enthalpy_density(const real* prim)
{
    const real rho = prim[0];
    const real pre = prim[3];
    return rho + pre * (1.0 + 1.0 / (ADIABATIC_GAMMA - 1.0));
}

void primitive_to_conserved(const real *prim, real *cons)
{
    const real rho = prim[0];
    const real u1 = prim[1];
    const real u2 = prim[2];
    const real pre = prim[3];

    const real w = primitive_to_lorentz_factor(prim);
    const real h = primitive_to_enthalpy_density(prim) / rho;
    const real m = rho * w;

    cons[0] = m;
    cons[1] = m * h * u1;
    cons[2] = m * h * u2;
    cons[3] = m * (h * w - 1.0) - pre;
}

void primitive_to_flux_vector(const real *prim, real *flux, int direction)
{
    const real vn = primitive_to_beta_component(prim, direction);
    const real pre = prim[3];
    real cons[4];
    primitive_to_conserved(prim, cons);

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pre * (direction == 0);
    flux[2] = vn * cons[2] + pre * (direction == 1);
    flux[3] = vn * cons[3] + pre * vn;
}

real primitive_to_sound_speed_squared(const real *prim)
{
    const real pre = prim[3];
    const real rho_h = primitive_to_enthalpy_density(prim);
    return ADIABATIC_GAMMA * pre / rho_h;
}

void primitive_to_outer_wavespeeds(const real *prim, real *wavespeeds, int direction)
{
    const real a2 = primitive_to_sound_speed_squared(prim);
    const real un = primitive_to_gamma_beta_component(prim, direction);
    const real uu = primitive_to_gamma_beta_squared(prim);
    const real vv = uu / (1.0 + uu);
    const real v2 = un * un / (1.0 + uu);
    const real vn = square_root(v2);
    const real k0 = square_root(a2 * (1.0 - vv) * (1.0 - vv * a2 - v2 * (1.0 - a2)));

    wavespeeds[0] = (vn * (1.0 - a2) - k0) / (1.0 - vv * a2);
    wavespeeds[1] = (vn * (1.0 - a2) + k0) / (1.0 - vv * a2);
}

void riemann_hlle(const real *pl, const real *pr, real *flux, int direction)
{
    real ul[4];
    real ur[4];
    real fl[4];
    real fr[4];
    real al[2];
    real ar[2];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
    primitive_to_flux_vector(pl, fl, direction);
    primitive_to_flux_vector(pr, fr, direction);
    primitive_to_outer_wavespeeds(pl, al, direction);
    primitive_to_outer_wavespeeds(pr, ar, direction);

    const real am = min2(0.0, min2(al[0], ar[0]));
    const real ap = max2(0.0, max2(al[1], ar[1]));

    for (int i = 0; i < 4; ++i)
    {
        flux[i] = (fl[i] * ap - fr[i] * am - (ul[i] - ur[i]) * ap * am) / (ap - am);
    }
}

void initial_primitive(real *primitive, int num_zones, real x0, real x1)
{
    real dx = (x1 - x0) / num_zones;

    for (int i = 0; i < num_zones; ++i)
    {
        real x = (i + 0.5) * dx;
        real *prim = &primitive[i * 4];

        if (x < 0.5 * (x0 + x1))
        {
            prim[0] = 1.0;
            prim[1] = 0.0;
            prim[2] = 0.0;
            prim[3] = 1.0;
        }
        else
        {
            prim[0] = 0.1;
            prim[1] = 0.0;
            prim[2] = 0.0;
            prim[3] = 0.125;
        }
    }
}

struct UpdateStruct
{
    int num_zones;
    real x0;
    real x1;
    real *primitive;
    real *conserved;
    real *flux;
};

struct UpdateStruct update_struct_new(int num_zones, real x0, real x1)
{
    struct UpdateStruct update;
    update.num_zones = num_zones;
    update.x0 = x0;
    update.x1 = x1;

    update.primitive = (real*) malloc(num_zones * 4 * sizeof(real));
    update.conserved = (real*) malloc(num_zones * 4 * sizeof(real));
    update.flux = (real*) malloc((num_zones + 1) * 4 * sizeof(real));

    return update;
}

void update_struct_del(struct UpdateStruct update)
{
    free(update.primitive);
    free(update.conserved);
    free(update.flux);
}

void update_struct_set_primitive(struct UpdateStruct update, const real *primitive_host)
{
    real *conserved_host = (real*) malloc(update.num_zones * 4 * sizeof(real));

    for (int i = 0; i < update.num_zones; ++i)
    {
        const real *prim = &primitive_host[4 * i];
        /* */ real *cons = &conserved_host[4 * i];
        primitive_to_conserved(prim, cons);
    }

    memcpy(
        update.primitive,
        primitive_host,
        update.num_zones * 4 * sizeof(real));

    memcpy(
        update.conserved,
        conserved_host,
        update.num_zones * 4 * sizeof(real));

    free(conserved_host);
}

void update_struct_get_primitive(struct UpdateStruct update, real *primitive_host)
{
    memcpy(primitive_host,
        update.primitive,
        update.num_zones * 4 * sizeof(real));
}

void update_struct_do_compute_flux(struct UpdateStruct update)
{
    for (int i = 0; i < update.num_zones + 1; ++i)
    {
        int il = i - 1;
        int ir = i;

        if (il == -1)
            il += 1;

        if (ir == update.num_zones)
            ir -= 1;

        const real *pl = &update.primitive[4 * il];
        const real *pr = &update.primitive[4 * ir];
        real *flux = &update.flux[4 * i];
        riemann_hlle(pl, pr, flux, 0);
    }
}

void update_struct_do_advance_cons(struct UpdateStruct update, real dt)
{
    for (int i = 0; i < update.num_zones; ++i)
    {
        const real dx = (update.x1 - update.x0) / update.num_zones;
        const real *fl = &update.flux[4 * (i + 0)];
        const real *fr = &update.flux[4 * (i + 1)];
        real *cons = &update.conserved[4 * i];
        real *prim = &update.primitive[4 * i];

        for (int q = 0; q < 4; ++q)
        {
            cons[q] -= (fr[q] - fl[q]) * dt / dx;
        }
        conserved_to_primitive(cons, prim);
    }
}

int main()
{
    const int num_zones = 1 << 16;
    const int fold = 100;
    const real x0 = 0.0;
    const real x1 = 1.0;
    const real dx = (x1 - x0) / num_zones;

    real *primitive = (real*) malloc(num_zones * 4 * sizeof(real));
    struct UpdateStruct update = update_struct_new(num_zones, x0, x1);

    initial_primitive(primitive, num_zones, x0, x1);
    update_struct_set_primitive(update, primitive);

    int iteration = 0;
    real time = 0.0;
    real dt = dx * 0.1;

    while (time < 0.1)
    {
        clock_t start = clock();

        for (int i = 0; i < fold; ++i)
        {
            update_struct_do_compute_flux(update);
            update_struct_do_advance_cons(update, dt);

            time += dt;
            iteration += 1;
        }
        clock_t end = clock();

        real seconds = ((real) (end - start)) / CLOCKS_PER_SEC;
        real mzps = (num_zones / 1e6) / seconds * fold;
        printf("[%d] t=%.3e Mzps=%.2f\n", iteration, time, mzps);
    }

    update_struct_get_primitive(update, primitive);
    update_struct_del(update);

    FILE* outfile = fopen("sr1d.dat", "w");

    for (int i = 0; i < num_zones; ++i)
    {
        real *prim = &primitive[i * 4];
        real x = (i + 0.5) * dx;
        fprintf(outfile, "%f %f %f %f\n", x, prim[0], prim[1], prim[3]);
    }
    fclose(outfile);
    free(primitive);

    return 0;
}
