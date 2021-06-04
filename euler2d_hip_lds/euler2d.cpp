#include <stdio.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <hip/hip_runtime.h>

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

__device__ void conserved_to_primitive(const real *cons, real *prim)
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

__device__ __host__ void primitive_to_conserved(const real *prim, real *cons)
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

__device__ real primitive_to_velocity_component(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

__device__ void primitive_to_flux_vector(const real *prim, real *flux, int direction)
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

__device__ real primitive_to_sound_speed_squared(const real *prim)
{
    const real rho = prim[0];
    const real pressure = prim[3];
    return ADIABATIC_GAMMA * pressure / rho;
}

__device__ void primitive_to_outer_wavespeeds(const real *prim, real *wavespeeds, int direction)
{
    const real cs = square_root(primitive_to_sound_speed_squared(prim));
    const real vn = primitive_to_velocity_component(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

__device__ void riemann_hlle(const real *pl, const real *pr, real *flux, int direction)
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

    hipMalloc(&update.primitive, ni * nj * 4 * sizeof(real));
    hipMalloc(&update.conserved, ni * nj * 4 * sizeof(real));

    return update;
}

void update_struct_del(struct UpdateStruct update)
{
    hipFree(update.primitive);
    hipFree(update.conserved);
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

    hipMemcpy(
        update.primitive,
        primitive_host,
        num_zones * 4 * sizeof(real),
        hipMemcpyHostToDevice
    );

    hipMemcpy(
        update.conserved,
        conserved_host,
        num_zones * 4 * sizeof(real),
        hipMemcpyHostToDevice
    );
    free(conserved_host);
}

void update_struct_get_primitive(struct UpdateStruct update, real *primitive_host)
{
    int num_zones = update.ni * update.nj;
    hipMemcpy(primitive_host,
        update.primitive,
        num_zones * 4 * sizeof(real),
        hipMemcpyDeviceToHost
    );
}

__global__ void update_struct_do_advance_cons(struct UpdateStruct update, real dt)
{
    const real dx = (update.x1 - update.x0) / update.ni;
    const real dy = (update.y1 - update.y0) / update.nj;
    int num_guard = 1;

    extern __shared__ real shared_prim[];

    // Have four index spaces:
    //
    // - lt: local thread index (in-block)
    // - gt: global thread index
    // - gm: global memory index
    // - lm: lds memory index

    int ni_lt = blockDim.x;
    int nj_lt = blockDim.y;
    int ni_gt = gridDim.x * blockDim.x;
    int nj_gt = gridDim.y * blockDim.y;
    int ni_gm = update.ni;
    int nj_gm = update.nj;
    int ni_lm = blockDim.x + 2 * num_guard;
    int nj_lm = blockDim.y + 2 * num_guard;

    int si_gm = 4 * nj_gm;
    int sj_gm = 4;
    int si_lm = 4 * nj_lm;
    int sj_lm = 4;

    int i_lt = threadIdx.x;
    int j_lt = threadIdx.y;
    int i_gt = threadIdx.x + blockIdx.x * blockDim.x;
    int j_gt = threadIdx.y + blockIdx.y * blockDim.y;

    if (i_gt >= update.ni || j_gt >= update.nj)
    {
        return;
    }

    {
        int i_lm = i_lt;
        int j_lm = j_lt;
        int i_gm = i_gt - num_guard;
        int j_gm = j_gt - num_guard;

        if (i_gm < 0)
            i_gm = 0;
        if (j_gm < 0)
            j_gm = 0;
        if (i_gm == update.ni)
            i_gm = update.ni - 1;
        if (j_gm == update.nj)
            j_gm = update.nj - 1;

        for (int q = 0; q < 4; ++q)
        {
            shared_prim     [i_lm * si_lm + j_lm * sj_lm + q] =
            update.primitive[i_gm * si_gm + j_gm * sj_gm + q];
        }
    }
    if (i_lt < 2 * num_guard)
    {
        int i_lm = i_lt + ni_lt;
        int j_lm = j_lt;
        int i_gm = i_gt - num_guard + ni_lt;
        int j_gm = j_gt - num_guard;

        if (i_gm < 0)
            i_gm = 0;
        if (j_gm < 0)
            j_gm = 0;
        if (i_gm == update.ni)
            i_gm = update.ni - 1;
        if (j_gm == update.nj)
            j_gm = update.nj - 1;

        for (int q = 0; q < 4; ++q)
        {
            shared_prim     [i_lm * si_lm + j_lm * sj_lm + q] =
            update.primitive[i_gm * si_gm + j_gm * sj_gm + q];
        }
    }
    if (j_lt < 2 * num_guard)
    {
        int i_lm = i_lt;
        int j_lm = j_lt + nj_lt;
        int i_gm = i_gt - num_guard;
        int j_gm = j_gt - num_guard + nj_lt;

        if (i_gm < 0)
            i_gm = 0;
        if (j_gm < 0)
            j_gm = 0;
        if (i_gm == update.ni)
            i_gm = update.ni - 1;
        if (j_gm == update.nj)
            j_gm = update.nj - 1;

        for (int q = 0; q < 4; ++q)
        {
            shared_prim     [i_lm * si_lm + j_lm * sj_lm + q] =
            update.primitive[i_gm * si_gm + j_gm * sj_gm + q];
        }
    }
    if (i_lt < 2 * num_guard && j_lt < 2 * num_guard)
    {
        int i_lm = i_lt + ni_lt;
        int j_lm = j_lt + nj_lt;
        int i_gm = i_gt - num_guard + ni_lt;
        int j_gm = j_gt - num_guard + nj_lt;

        if (i_gm < 0)
            i_gm = 0;
        if (j_gm < 0)
            j_gm = 0;
        if (i_gm == update.ni)
            i_gm = update.ni - 1;
        if (j_gm == update.nj)
            j_gm = update.nj - 1;

        for (int q = 0; q < 4; ++q)
        {
            shared_prim     [i_lm * si_lm + j_lm * sj_lm + q] =
            update.primitive[i_gm * si_gm + j_gm * sj_gm + q];
        }
    }
    __syncthreads();

    int i_gm = i_gt;
    int j_gm = j_gt;
    int i_lm = i_lt + num_guard;
    int j_lm = j_lt + num_guard;

    int il = i_lm - 1;
    int ir = i_lm + 1;
    int jl = j_lm - 1;
    int jr = j_lm + 1;

    const real *pli = &shared_prim[il * si_lm + j_lm * sj_lm];
    const real *pri = &shared_prim[ir * si_lm + j_lm * sj_lm];
    const real *plj = &shared_prim[i_lm * si_lm + jl * sj_lm];
    const real *prj = &shared_prim[i_lm * si_lm + jr * sj_lm];
    real *prim = &shared_prim     [i_lm * si_lm + j_lm * sj_lm];
    real *cons = &update.conserved[i_gm * si_gm + j_gm * sj_gm];

    real fli[4];
    real fri[4];
    real flj[4];
    real frj[4];
    riemann_hlle(pli, prim, fli, 0);
    riemann_hlle(prim, pri, fri, 0);
    riemann_hlle(plj, prim, flj, 1);
    riemann_hlle(prim, prj, frj, 1);

    for (int q = 0; q < 4; ++q)
    {
        cons[q] -= ((fri[q] - fli[q]) / dx + (frj[q] - flj[q]) / dy) * dt;
    }
    conserved_to_primitive(cons, prim);

    for (int q = 0; q < 4; ++q)
    {
        update.primitive[i_gm * si_gm + j_gm * sj_gm + q] = prim[q];
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

    int thread_per_dim_i = 16;
    int thread_per_dim_j = 16;
    int blocks_per_dim_i = (ni + thread_per_dim_i - 1) / thread_per_dim_i;
    int blocks_per_dim_j = (nj + thread_per_dim_j - 1) / thread_per_dim_j;
    dim3 group_size = dim3(blocks_per_dim_i, blocks_per_dim_j, 1);
    dim3 block_size = dim3(thread_per_dim_i, thread_per_dim_j, 1);
    int shared_memory = (block_size.x + 2) * (block_size.y + 2) * 4 * sizeof(real);

    while (time < 0.1)
    {
        clock_t start = clock();

        for (int i = 0; i < fold; ++i)
        {
            update_struct_do_advance_cons<<<group_size, block_size, shared_memory>>>(update, dt);

            time += dt;
            iteration += 1;
        }
        hipDeviceSynchronize();
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
