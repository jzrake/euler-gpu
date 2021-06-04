#include <stdio.h>
#include <time.h>

#define ADIABATIC_GAMMA (5.0 / 3.0)

typedef double real;

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
    const real cs = sqrt(primitive_to_sound_speed_squared(prim));
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

    const real am = min(0.0, min(al[0], ar[0]));
    const real ap = max(0.0, max(al[1], ar[1]));

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
};

struct UpdateStruct update_struct_new(int num_zones, real x0, real x1)
{
    struct UpdateStruct update;
    update.num_zones = num_zones;
    update.x0 = x0;
    update.x1 = x1;

    cudaMalloc(&update.primitive, num_zones * 4 * sizeof(real));
    cudaMalloc(&update.conserved, num_zones * 4 * sizeof(real));

    return update;
}

void update_struct_del(struct UpdateStruct update)
{
    cudaFree(update.primitive);
    cudaFree(update.conserved);
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

    cudaMemcpy(
        update.primitive,
        primitive_host,
        update.num_zones * 4 * sizeof(real),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        update.conserved,
        conserved_host,
        update.num_zones * 4 * sizeof(real),
        cudaMemcpyHostToDevice
    );
    free(conserved_host);
}

void update_struct_get_primitive(struct UpdateStruct update, real *primitive_host)
{
    cudaMemcpy(primitive_host,
        update.primitive,
        update.num_zones * 4 * sizeof(real),
        cudaMemcpyDeviceToHost
    );
}

__global__ void update_struct_do_advance_cons(UpdateStruct update, real dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= update.num_zones)
        return;

    int il = i - 1;
    int ic = i;
    int ir = i + 1;

    if (il == -1)
        il += 1;

    if (ir == update.num_zones)
        ir -= 1;

    const real dx = (update.x1 - update.x0) / update.num_zones;
    real *pl = &update.primitive[4 * il];
    real *pc = &update.primitive[4 * ic];
    real *pr = &update.primitive[4 * ir];

    real fl[4];
    real fr[4];
    riemann_hlle(pl, pc, fl, 0);
    riemann_hlle(pc, pr, fr, 0);

    real *cons = &update.conserved[4 * ic];

    for (int q = 0; q < 4; ++q)
    {
        cons[q] -= (fr[q] - fl[q]) * dt / dx;
    }
    conserved_to_primitive(cons, pc);
}

int main()
{
    const int num_zones = 1 << 24;
    const int block_size = 64;
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
            update_struct_do_advance_cons<<<num_zones / block_size, block_size>>>(update, dt);
            time += dt;
            iteration += 1;
        }
        cudaDeviceSynchronize();
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

    cudaError_t error = cudaGetLastError();

    if (error)
    {
        printf("%s\n", cudaGetErrorString(error));
    }
    return 0;
}
