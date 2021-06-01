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

__device__ void conserved_to_primitive(const real *cons, real *prim)
{
    const real rho = cons[0];
    const real px = cons[1];
    const real py = cons[2];
    const real pz = cons[3];
    const real energy = cons[4];

    const real vx = px / rho;
    const real vy = py / rho;
    const real vz = pz / rho;
    const real kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    const real thermal_energy = energy - kinetic_energy;
    const real pressure = thermal_energy * (ADIABATIC_GAMMA - 1.0);

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = vz;
    prim[4] = pressure;
}

__device__ __host__ void primitive_to_conserved(const real *prim, real *cons)
{
    const real rho = prim[0];
    const real vx = prim[1];
    const real vy = prim[2];
    const real vz = prim[3];
    const real pressure = prim[4];

    const real px = vx * rho;
    const real py = vy * rho;
    const real pz = vz * rho;
    const real kinetic_energy = 0.5 * rho * (vx * vx + vy * vy + vz * vz);
    const real thermal_energy = pressure / (ADIABATIC_GAMMA - 1.0);

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
    cons[3] = pz;
    cons[4] = kinetic_energy + thermal_energy;
}

__device__ real primitive_to_velocity_component(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        case 2: return prim[3];
        default: return 0.0;
    }
}

__device__ void primitive_to_flux_vector(const real *prim, real *flux, int direction)
{
    const real vn = primitive_to_velocity_component(prim, direction);
    const real pressure = prim[4];
    real cons[5];
    primitive_to_conserved(prim, cons);

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
    flux[3] = vn * cons[3] + pressure * (direction == 2);
    flux[4] = vn * cons[4] + pressure * vn;
}

__device__ real primitive_to_sound_speed_squared(const real *prim)
{
    const real rho = prim[0];
    const real pressure = prim[4];
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
    real ul[5];
    real ur[5];
    real fl[5];
    real fr[5];
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

    for (int i = 0; i < 5; ++i)
    {
        flux[i] = (fl[i] * ap - fr[i] * am - (ul[i] - ur[i]) * ap * am) / (ap - am);
    }
}

void initial_primitive(
    real *primitive,
    int ni,
    int nj,
    int nk,
    real x0,
    real x1,
    real y0,
    real y1,
    real z0,
    real z1)
{
    real dx = (x1 - x0) / ni;
    real dy = (y1 - y0) / nj;
    real dz = (z1 - z0) / nk;

    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            for (int k = 0; k < nk; ++k)
            {
                real x = (i + 0.5) * dx;
                real y = (j + 0.5) * dy;
                real z = (k + 0.5) * dz;
                real *prim = &primitive[5 * (i * nj * nk + j * nk + k)];
                real r2 = power(x - 0.5, 2) + power(y - 0.5, 2) + power(z - 0.5, 2);

                if (square_root(r2) < 0.125)
                {
                    prim[0] = 1.0;
                    prim[1] = 0.0;
                    prim[2] = 0.0;
                    prim[3] = 0.0;
                    prim[4] = 1.0;
                }
                else
                {
                    prim[0] = 0.1;
                    prim[1] = 0.0;
                    prim[2] = 0.0;
                    prim[3] = 0.0;
                    prim[4] = 0.125;
                }
            }
        }
    }
}

struct UpdateStruct
{
    int ni;
    int nj;
    int nk;
    real x0;
    real x1;
    real y0;
    real y1;
    real z0;
    real z1;
    real *primitive;
    real *conserved;
    real *flux_i;
    real *flux_j;
    real *flux_k;
};

struct UpdateStruct update_struct_new(
    int ni,
    int nj,
    int nk,
    real x0,
    real x1,
    real y0,
    real y1,
    real z0,
    real z1)
{
    struct UpdateStruct update;
    update.ni = ni;
    update.nj = nj;
    update.nk = nk;
    update.x0 = x0;
    update.x1 = x1;
    update.y0 = y0;
    update.y1 = y1;
    update.z0 = z0;
    update.z1 = z1;

    cudaMalloc(&update.primitive, ni * nj * nk * 5 * sizeof(real));
    cudaMalloc(&update.conserved, ni * nj * nk * 5 * sizeof(real));
    cudaMalloc(&update.flux_i, (ni + 1) * nj * nk * 5 * sizeof(real));
    cudaMalloc(&update.flux_j, ni * (nj + 1) * nk * 5 * sizeof(real));
    cudaMalloc(&update.flux_k, ni * nj * (nk + 1) * 5 * sizeof(real));

    return update;
}

void update_struct_del(struct UpdateStruct update)
{
    cudaFree(update.primitive);
    cudaFree(update.conserved);
    cudaFree(update.flux_i);
    cudaFree(update.flux_j);
    cudaFree(update.flux_k);
}

void update_struct_set_primitive(struct UpdateStruct update, const real *primitive_host)
{
    int ni = update.ni;
    int nj = update.nj;
    int nk = update.nk;
    int num_zones = ni * nj * nk;
    real *conserved_host = (real*) malloc(num_zones * 5 * sizeof(real));

    for (int i = 0; i < ni; ++i)
    {
        for (int j = 0; j < nj; ++j)
        {
            for (int k = 0; k < nk; ++k)
            {
                const real *prim = &primitive_host[5 * (i * nj * nk + j * nk + k)];
                /* */ real *cons = &conserved_host[5 * (i * nj * nk + j * nk + k)];
                primitive_to_conserved(prim, cons);
            }
        }
    }

    cudaMemcpy(
        update.primitive,
        primitive_host,
        num_zones * 5 * sizeof(real),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        update.conserved,
        conserved_host,
        num_zones * 5 * sizeof(real),
        cudaMemcpyHostToDevice
    );
    free(conserved_host);
}

void update_struct_get_primitive(struct UpdateStruct update, real *primitive_host)
{
    int num_zones = update.ni * update.nj * update.nk;
    cudaMemcpy(primitive_host,
        update.primitive,
        num_zones * 5 * sizeof(real),
        cudaMemcpyDeviceToHost
    );
}

__global__ void update_struct_do_compute_flux(struct UpdateStruct update)
{
    int ni = update.ni;
    int nj = update.nj;
    int nk = update.nk;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;

    if (i < ni + 1 && j < nj && k < nk)
    {
        int il = i - 1;
        int ir = i;

        if (il == -1)
            il += 1;

        if (ir == ni)
            ir -= 1;

        const real *pl = &update.primitive[5 * (il * nj * nk + j * nk + k)];
        const real *pr = &update.primitive[5 * (ir * nj * nk + j * nk + k)];

        real *flux = &update.flux_i[5 * (i * nj * nk + j * nk + k)];
        riemann_hlle(pl, pr, flux, 0);
    }

    if (j < nj + 1 && k < nk && i < ni)
    {
        int jl = j - 1;
        int jr = j;

        if (jl == -1)
            jl += 1;

        if (jr == nj)
            jr -= 1;

        const real *pl = &update.primitive[5 * (i * nj * nk + jl * nk + k)];
        const real *pr = &update.primitive[5 * (i * nj * nk + jr * nk + k)];

        real *flux = &update.flux_j[5 * (i * nj * nk + j * nk + k)];
        riemann_hlle(pl, pr, flux, 1);
    }

    if (k < nk + 1 && i < ni && j < nj)
    {
        int kl = k - 1;
        int kr = k;

        if (kl == -1)
            kl += 1;

        if (kr == nk)
            kr -= 1;

        const real *pl = &update.primitive[5 * (i * nj * nk + j * nk + kl)];
        const real *pr = &update.primitive[5 * (i * nj * nk + j * nk + kr)];

        real *flux = &update.flux_k[5 * (i * nj * nk + j * nk + k)];
        riemann_hlle(pl, pr, flux, 2);
    }
}

__global__ void update_struct_do_advance_cons(struct UpdateStruct update, real dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z * blockDim.z + threadIdx.z;
    int ni = update.ni;
    int nj = update.nj;
    int nk = update.nk;
    const real dx = (update.x1 - update.x0) / update.ni;
    const real dy = (update.y1 - update.y0) / update.nj;
    const real dz = (update.z1 - update.z0) / update.nk;

    if (i < ni && j < nj && k < nk)
    {
        const real *fli = &update.flux_i[5 * ((i + 0) * nj * nk + j * nk + k)];
        const real *fri = &update.flux_i[5 * ((i + 1) * nj * nk + j * nk + k)];
        const real *flj = &update.flux_j[5 * (i * nj * nk + (j + 0) * nk + k)];
        const real *frj = &update.flux_j[5 * (i * nj * nk + (j + 1) * nk + k)];
        const real *flk = &update.flux_k[5 * (i * nj * nk + j * nk + (k + 0))];
        const real *frk = &update.flux_k[5 * (i * nj * nk + j * nk + (k + 1))];

        real *cons = &update.conserved[5 * (i * nj * nk + j * nk + k)];
        real *prim = &update.primitive[5 * (i * nj * nk + j * nk + k)];

        for (int q = 0; q < 5; ++q)
        {
            cons[q] -= (
                (fri[q] - fli[q]) / dx +
                (frj[q] - flj[q]) / dy +
                (frk[q] - flk[q]) / dz) * dt;
        }
        conserved_to_primitive(cons, prim);
    }
}

int main()
{
    const int ni = 128;
    const int nj = 128;
    const int nk = 128;
    const int fold = 10;
    const real x0 = 0.0;
    const real x1 = 1.0;
    const real y0 = 0.0;
    const real y1 = 1.0;
    const real z0 = 0.0;
    const real z1 = 1.0;
    const real dx = (x1 - x0) / ni;
    const real dy = (y1 - y0) / nj;
    const real dz = (z1 - z0) / nk;

    real *primitive = (real*) malloc(ni * nj * nk * 5 * sizeof(real));
    struct UpdateStruct update = update_struct_new(ni, nj, nk, x0, x1, y0, y1, z0, z1);

    initial_primitive(primitive, ni, nj, nk, x0, x1, y0, y1, z0, z1);
    update_struct_set_primitive(update, primitive);

    int iteration = 0;
    real time = 0.0;
    real dt = min2(min2(dx, dy), dz) * 0.05;

    int thread_per_dim_i = 4;
    int thread_per_dim_j = 4;
    int thread_per_dim_k = 4;
    int blocks_per_dim_i = (ni + thread_per_dim_i - 1) / thread_per_dim_i;
    int blocks_per_dim_j = (nj + thread_per_dim_j - 1) / thread_per_dim_j;
    int blocks_per_dim_k = (nk + thread_per_dim_k - 1) / thread_per_dim_k;
    dim3 group_size = dim3(blocks_per_dim_i, blocks_per_dim_j, blocks_per_dim_k);
    dim3 block_size = dim3(thread_per_dim_i, thread_per_dim_j, thread_per_dim_k);

    while (time < 0.2)
    {
        clock_t start = clock();

        for (int i = 0; i < fold; ++i)
        {
            update_struct_do_compute_flux<<<group_size, block_size>>>(update);
            update_struct_do_advance_cons<<<group_size, block_size>>>(update, dt);

            time += dt;
            iteration += 1;
            cudaDeviceSynchronize();
        }
        clock_t end = clock();

        real seconds = ((real) (end - start)) / CLOCKS_PER_SEC;
        real mzps = (ni * nj * nk / 1e6) / seconds * fold;
        printf("[%d] t=%.3e Mzps=%.2f\n", iteration, time, mzps);
    }

    update_struct_get_primitive(update, primitive);
    update_struct_del(update);

    FILE* outfile = fopen("euler3d.bin", "wb");
    fwrite(primitive, sizeof(real), 5 * ni * nj * nk, outfile);
    fclose(outfile);
    free(primitive);
    return 0;
}
