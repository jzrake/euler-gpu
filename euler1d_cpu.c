#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>




// ============================================================================
#ifndef __NVCC__
#define __host__
#define __device__

void *compute_malloc(size_t count)
{
    return malloc(count);
}

void compute_free(void* ptr)
{
    free(ptr);
}

void compute_memcpy_host_to_device(void* dst, const void* src, size_t count)
{
    memcpy(dst, src, count);
}

void compute_memcpy_device_to_host(void* dst, const void* src, size_t count)
{
    memcpy(dst, src, count);
}

// ============================================================================
#else

void *compute_malloc(size_t count)
{
    void *ptr;
    cudaMalloc(&ptr, count);
    return ptr;
}

void compute_free(void* ptr)
{
    cudaFree(ptr);
}

void compute_memcpy_host_to_device(void* dst, const void* src, size_t count)
{
    cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
}

void compute_memcpy_device_to_host(void* dst, const void* src, size_t count)
{
    cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
}

#endif




// ============================================================================
#define ADIABATIC_GAMMA (5.0 / 3.0)
#define max2(a, b) (a) > (b) ? (a) : (b)
#define min2(a, b) (a) < (b) ? (a) : (b)
#define max3(a, b, c) max2(a, max2(b, c))
#define min3(a, b, c) min2(a, min2(b, c))
typedef double real;




// ============================================================================
__host__ __device__ void conserved_to_primitive(const real *cons, real *prim)
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

__host__ __device__ void primitive_to_conserved(const real *prim, real *cons)
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

__host__ __device__ real primitive_to_velocity_component(const real *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

__host__ __device__ void primitive_to_flux_vector(const real *prim, real *flux, int direction)
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

__host__ __device__ real primitive_to_sound_speed_squared(const real *prim)
{
    const real rho = prim[0];
    const real pressure = prim[3];
    return ADIABATIC_GAMMA * pressure / rho;
}

__host__ __device__ void primitive_to_outer_wavespeeds(const real *prim, real *wavespeeds, int direction)
{
    const real cs = sqrt(primitive_to_sound_speed_squared(prim));
    const real vn = primitive_to_velocity_component(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

__host__ __device__ void riemann_hlle(const real *pl, const real *pr, real *flux, int direction)
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

    const real am = min3(0.0, al[0], ar[0]);
    const real ap = max3(0.0, al[1], ar[1]);

    for (int i = 0; i < 4; ++i)
    {
        flux[i] = (fl[i] * ap - fr[i] * am - (ul[i] - ur[i]) * ap * am) / (ap - am);
    }
}




// ============================================================================
struct UpdateStruct
{
    int num_zones;
    int block_size;
    real x0;
    real x1;
    real *primitive;
    real *conserved;
    real *flux;
};

__device__ void do_flux(struct UpdateStruct update, int i)
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

__device__ void do_advance_cons(struct UpdateStruct update, int i, real dt)
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




// ============================================================================
struct UpdateStruct update_struct_new(int num_zones, int block_size, real x0, real x1)
{
    struct UpdateStruct update;
    update.num_zones = num_zones;
    update.block_size = block_size;
    update.x0 = x0;
    update.x1 = x1;

    update.primitive = (double*) compute_malloc(num_zones * 4 * sizeof(real));
    update.conserved = (double*) compute_malloc(num_zones * 4 * sizeof(real));
    update.flux = (double*) compute_malloc((num_zones + 1) * 4 * sizeof(real));

    return update;
}

void update_struct_del(struct UpdateStruct update)
{
    compute_free(update.primitive);
    compute_free(update.conserved);
    compute_free(update.flux);
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

    compute_memcpy_host_to_device(
        update.primitive,
        primitive_host,
        update.num_zones * 4 * sizeof(real)
    );

    compute_memcpy_host_to_device(
        update.conserved,
        conserved_host,
        update.num_zones * 4 * sizeof(real)
    );
    free(conserved_host);
}

void update_struct_get_primitive(struct UpdateStruct update, real *primitive_host)
{
    compute_memcpy_device_to_host(primitive_host,
        update.primitive,
        update.num_zones * 4 * sizeof(real)
    );
}

#ifndef __NVCC__

void update_struct_do_compute_flux(struct UpdateStruct update)
{
    for (int i = 0; i < update.num_zones + 1; ++i)
    {
        do_flux(update, i);
    }
}

void update_struct_do_advance_cons(struct UpdateStruct update, real dt)
{
    for (int i = 0; i < update.num_zones + 1; ++i)
    {
        do_advance_cons(update, i, dt);
    }
}

#else

__global__ void cuda_do_compute_flux(struct UpdateStruct update)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < update.num_zones + 1)
    {
        do_flux(update, i);
    }
}

__global__ void cuda_do_advance_cons(struct UpdateStruct update, real dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < update.num_zones)
    {
        do_advance_cons(update, i, dt);
    }
}

void update_struct_do_compute_flux(struct UpdateStruct update)
{
    int num_blocks = update.num_zones / update.block_size + 1;
    cuda_do_compute_flux<<<num_blocks, update.block_size>>>(update);
}

void update_struct_do_advance_cons(struct UpdateStruct update, real dt)
{
    int num_blocks = update.num_zones / update.block_size + 0;
    cuda_do_advance_cons<<<num_blocks, update.block_size>>>(update, dt);
}

#endif




// ============================================================================
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




// ============================================================================
int main()
{
    const int num_zones = 1 << 14;
    const int block_size = 32;
    const int fold = 100;
    const real x0 = 0.0;
    const real x1 = 1.0;
    const real dx = (x1 - x0) / num_zones;

    real *primitive = (real*) malloc(num_zones * 4 * sizeof(real));
    struct UpdateStruct update = update_struct_new(num_zones, block_size, x0, x1);

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

    FILE* outfile = fopen("euler1d.dat", "w");

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
