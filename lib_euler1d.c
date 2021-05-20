#include <math.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>




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
