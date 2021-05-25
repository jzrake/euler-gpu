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
#define power powf
#else
typedef double real;
#define square_root sqrt
#define power pow
#endif

void conserved_to_primitive(const real *cons, real *prim)
{
    const real rho = cons[0];
    const real px = cons[1];
    const real py = cons[2];
    const real energy = cons[3];

    const real vx = px / rho;
    const real vy = py / rho;
    const real kinetic_energy = 0.5 * rho * (vx * vx + vy * vy);
    const real thermal_energy = energy - kinetic_energy;
    const real pressure = thermal_energy * (ADIABATIC_GAMMA - 1.0);

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = pressure;
}

void primitive_to_conserved(const real *prim, real *cons)
{
    const real rho = prim[0];
    const real vx = prim[1];
    const real vy = prim[2];
    const real pressure = prim[3];

    const real px = vx * rho;
    const real py = vy * rho;
    const real kinetic_energy = 0.5 * rho * (vx * vx + vy * vy);
    const real thermal_energy = pressure / (ADIABATIC_GAMMA - 1.0);

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
    cons[3] = kinetic_energy + thermal_energy;
}

real primitive_to_velocity_component(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

void primitive_to_flux_vector(const real *prim, real *flux, int direction)
{
    const real vn = primitive_to_velocity_component(prim, direction);
    const real pressure = prim[3];
    real cons[4];
    primitive_to_conserved(prim, cons);

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
    flux[3] = vn * cons[3] + pressure * vn;
}

real primitive_to_sound_speed_squared(const real *prim)
{
    const real rho = prim[0];
    const real pressure = prim[3];
    return ADIABATIC_GAMMA * pressure / rho;
}

void primitive_to_outer_wavespeeds(const real *prim, real *wavespeeds, int direction)
{
    const real cs = square_root(primitive_to_sound_speed_squared(prim));
    const real vn = primitive_to_velocity_component(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
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

void initial_primitive(real *primitive, int ni, int nj, real x0, real x1, real y0, real y1)
{
    real dx = (x1 - x0) / ni;
    real dy = (y1 - y0) / nj;

    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            real x = (i + 0.5) * dx;
            real y = (j + 0.5) * dy;
            real *prim = &primitive[4 * (i * nj + j)];
            real r2 = power(x - 0.5, 2) + power(y - 0.5, 2);

            if (square_root(r2) < 0.125)
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
}

struct UpdateStruct
{
    int ni;
    int nj;
    real x0;
    real x1;
    real y0;
    real y1;
    real *primitive;
    real *conserved;
    real *flux_i;
    real *flux_j;
};

struct UpdateStruct update_struct_new(int ni, int nj, real x0, real x1, real y0, real y1)
{
    struct UpdateStruct update;
    update.ni = ni;
    update.nj = nj;
    update.x0 = x0;
    update.x1 = x1;
    update.y0 = y0;
    update.y1 = y1;

    update.primitive = (real*) malloc(ni * nj * 4 * sizeof(real));
    update.conserved = (real*) malloc(ni * nj * 4 * sizeof(real));
    update.flux_i = (real*) malloc((ni + 1) * nj * 4 * sizeof(real));
    update.flux_j = (real*) malloc(ni * (nj + 1) * 4 * sizeof(real));

    return update;
}

void update_struct_del(struct UpdateStruct update)
{
    free(update.primitive);
    free(update.conserved);
    free(update.flux_i);
    free(update.flux_j);
}

void update_struct_set_primitive(struct UpdateStruct update, const real *primitive_host)
{
    int ni = update.ni;
    int nj = update.nj;
    int num_zones = ni * nj;
    real *conserved_host = (real*) malloc(num_zones * 4 * sizeof(real));

    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            const real *prim = &primitive_host[4 * (i * nj + j)];
            /* */ real *cons = &conserved_host[4 * (i * nj + j)];
            primitive_to_conserved(prim, cons);
        }
    }

    memcpy(
        update.primitive,
        primitive_host,
        num_zones * 4 * sizeof(real)
    );

    memcpy(
        update.conserved,
        conserved_host,
        num_zones * 4 * sizeof(real)
    );
    free(conserved_host);
}

void update_struct_get_primitive(struct UpdateStruct update, real *primitive_host)
{
    int num_zones = update.ni * update.nj;
    memcpy(primitive_host,
        update.primitive,
        num_zones * 4 * sizeof(real)
    );
}

void update_struct_do_compute_flux(struct UpdateStruct update)
{
    int ni = update.ni;
    int nj = update.nj;

    for (int i = 0; i < ni + 1; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            int il = i - 1;
            int ir = i;

            if (il == -1)
                il += 1;

            if (ir == ni)
                ir -= 1;

            const real *pl = &update.primitive[4 * (il * nj + j)];
            const real *pr = &update.primitive[4 * (ir * nj + j)];

            real *flux = &update.flux_i[4 * (i * nj + j)];
            riemann_hlle(pl, pr, flux, 0);
        }
    }

    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj + 1; ++j)
        {
            int jl = j - 1;
            int jr = j;

            if (jl == -1)
                jl += 1;

            if (jr == nj)
                jr -= 1;

            const real *pl = &update.primitive[4 * (i * nj + jl)];
            const real *pr = &update.primitive[4 * (i * nj + jr)];

            real *flux = &update.flux_j[4 * (i * nj + j)];
            riemann_hlle(pl, pr, flux, 1);
        }
    }
}

void update_struct_do_advance_cons(struct UpdateStruct update, real dt)
{
    int ni = update.ni;
    int nj = update.nj;
    const real dx = (update.x1 - update.x0) / update.ni;
    const real dy = (update.y1 - update.y0) / update.nj;

    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            const real *fli = &update.flux_i[4 * ((i + 0) * nj + j)];
            const real *fri = &update.flux_i[4 * ((i + 1) * nj + j)];
            const real *flj = &update.flux_j[4 * (i * nj + (j + 0))];
            const real *frj = &update.flux_j[4 * (i * nj + (j + 1))];

            real *cons = &update.conserved[4 * (i * nj + j)];
            real *prim = &update.primitive[4 * (i * nj + j)];

            for (int q = 0; q < 4; ++q)
            {
                cons[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
            }
            conserved_to_primitive(cons, prim);
        }
    }
}

int main()
{
    const int ni = 1024;
    const int nj = 1024;
    const int fold = 10;
    const real x0 = 0.0;
    const real x1 = 1.0;
    const real y0 = 0.0;
    const real y1 = 1.0;
    const real dx = (x1 - x0) / ni;
    const real dy = (y1 - y0) / nj;

    real *primitive = (real*) malloc(ni * nj * 4 * sizeof(real));
    struct UpdateStruct update = update_struct_new(ni, nj, x0, x1, y0, y1);

    initial_primitive(primitive, ni, nj, x0, x1, y0, y1);
    update_struct_set_primitive(update, primitive);

    int iteration = 0;
    real time = 0.0;
    real dt = min2(dx, dy) * 0.05;

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
        real mzps = (ni * nj / 1e6) / seconds * fold;
        printf("[%d] t=%.3e Mzps=%.2f\n", iteration, time, mzps);
    }

    update_struct_get_primitive(update, primitive);
    update_struct_del(update);

    FILE* outfile = fopen("euler2d.dat", "w");

    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            real *prim = &primitive[4 * (i * nj + j)];
            real x = (i + 0.5) * dx;
            real y = (j + 0.5) * dy;
            fprintf(outfile, "%f %f %f %f %f %f\n", x, y, prim[0], prim[1], prim[2], prim[3]);
        }
    }
    fclose(outfile);
    free(primitive);
    return 0;
}
