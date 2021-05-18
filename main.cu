#include <stdio.h>
#include <time.h>

#define ADIABATIC_GAMMA (5.0 / 3.0)
#define max(a, b) (a) > (b) ? (a) : (b)
#define min(a, b) (a) < (b) ? (a) : (b)
#define max3(a, b, c) max(a, max(b, c))
#define min3(a, b, c) min(a, min(b, c))

__device__ void conserved_to_primitive(const double *cons, double *prim)
{
    const double rho = cons[0];
    const double px = cons[1];
    const double py = cons[2];
    const double energy = cons[3];

    const double vx = px / rho;
    const double vy = py / rho;
    const double kinetic_energy = 0.5 * rho * (vx * vx + vy * vy);
    const double thermal_energy = energy - kinetic_energy;
    const double pressure = thermal_energy * (ADIABATIC_GAMMA - 1.0);

    prim[0] = rho;
    prim[1] = vx;
    prim[2] = vy;
    prim[3] = pressure;
}

__device__ __host__ void primitive_to_conserved(const double *prim, double *cons)
{
    const double rho = prim[0];
    const double vx = prim[1];
    const double vy = prim[2];
    const double pressure = prim[3];

    const double px = vx * rho;
    const double py = vy * rho;
    const double kinetic_energy = 0.5 * rho * (vx * vx + vy * vy);
    const double thermal_energy = pressure / (ADIABATIC_GAMMA - 1.0);

    cons[0] = rho;
    cons[1] = px;
    cons[2] = py;
    cons[3] = kinetic_energy + thermal_energy;
}

__device__ double primitive_to_velocity_component(const double *prim, int direction)
{
    switch (direction)
    {
        case 0: return prim[1];
        case 1: return prim[2];
        default: return 0.0;
    }
}

__device__ void primitive_to_flux_vector(const double *prim, double *flux, int direction)
{
    const double vn = primitive_to_velocity_component(prim, direction);
    const double pressure = prim[3];
    double cons[4];
    primitive_to_conserved(prim, cons);

    flux[0] = vn * cons[0];
    flux[1] = vn * cons[1] + pressure * (direction == 0);
    flux[2] = vn * cons[2] + pressure * (direction == 1);
    flux[3] = vn * cons[3] + pressure * vn;
}

__device__ double primitive_to_sound_speed_squared(const double *prim)
{
    const double rho = prim[0];
    const double pressure = prim[3];
    return ADIABATIC_GAMMA * pressure / rho;
}

__device__ void primitive_to_outer_wavespeeds(const double *prim, double *wavespeeds, int direction)
{
    const double cs = sqrt(primitive_to_sound_speed_squared(prim));
    const double vn = primitive_to_velocity_component(prim, direction);
    wavespeeds[0] = vn - cs;
    wavespeeds[1] = vn + cs;
}

__device__ void riemann_hlle(const double *pl, const double *pr, double *flux, int direction)
{
    double ul[4];
    double ur[4];
    double fl[4];
    double fr[4];
    double al[2];
    double ar[2];

    primitive_to_conserved(pl, ul);
    primitive_to_conserved(pr, ur);
    primitive_to_flux_vector(pl, fl, direction);
    primitive_to_flux_vector(pr, fr, direction);
    primitive_to_outer_wavespeeds(pl, al, direction);
    primitive_to_outer_wavespeeds(pr, ar, direction);

    const double am = min3(al[0], ar[0], 0.0);
    const double ap = max3(al[1], ar[1], 0.0);

    for (int i = 0; i < 4; ++i)
    {
        flux[i] = (fl[i] * ap - fr[i] * am - (ul[i] - ur[i]) * ap * am) / (ap - am);
    }
}

void initial_primitive(double *primitive, int num_zones, double x0, double x1)
{
    double dx = (x1 - x0) / num_zones;

    for (int i = 0; i < num_zones; ++i)
    {
        double x = (i + 0.5) * dx;
        double *prim = &primitive[i * 4];

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
    double x0;
    double x1;
    double *primitive;
    double *conserved;
    double *flux;
};

struct UpdateStruct update_struct_new(int num_zones, double x0, double x1)
{
    struct UpdateStruct update;
    update.num_zones = num_zones;
    update.x0 = x0;
    update.x1 = x1;

    cudaMalloc(&update.primitive, num_zones * 4 * sizeof(double));
    cudaMalloc(&update.conserved, num_zones * 4 * sizeof(double));
    cudaMalloc(&update.flux, (num_zones + 1) * 4 * sizeof(double));

    return update;
}

void update_struct_del(struct UpdateStruct update)
{
    cudaFree(update.primitive);
    cudaFree(update.conserved);
    cudaFree(update.flux);
}

void update_struct_set_primitive(struct UpdateStruct update, const double *primitive_host)
{
    double *conserved_host = (double*) malloc(update.num_zones * 4 * sizeof(double));

    for (int i = 0; i < update.num_zones; ++i)
    {
        const double *prim = &primitive_host[4 * i];
        /* */ double *cons = &conserved_host[4 * i];
        primitive_to_conserved(prim, cons);
    }

    cudaMemcpy(
        update.primitive,
        primitive_host,
        update.num_zones * 4 * sizeof(double),
        cudaMemcpyHostToDevice
    );

    cudaMemcpy(
        update.conserved,
        conserved_host,
        update.num_zones * 4 * sizeof(double),
        cudaMemcpyHostToDevice
    );
    free(conserved_host);
}

void update_struct_get_primitive(struct UpdateStruct update, double *primitive_host)
{
    cudaMemcpy(primitive_host,
        update.primitive,
        update.num_zones * 4 * sizeof(double),
        cudaMemcpyDeviceToHost
    );
}

__global__ void update_struct_do_compute_flux(UpdateStruct update)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int il = i - 1;
    int ir = i;

    if (i > update.num_zones)
    {
        return;
    }

    if (il == -1)
        il += 1;

    if (ir == update.num_zones)
        ir -= 1;

    const double *pl = &update.primitive[4 * il];
    const double *pr = &update.primitive[4 * ir];
    double *flux = &update.flux[4 * i];
    riemann_hlle(pl, pr, flux, 0);
}

__global__ void update_struct_do_advance_cons(UpdateStruct update, double dt)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= update.num_zones)
    {
        return;
    }
    const double dx = (update.x1 - update.x0) / update.num_zones;
    const double *fl = &update.flux[4 * (i + 0)];
    const double *fr = &update.flux[4 * (i + 1)];
    double *cons = &update.conserved[4 * i];

    for (int q = 0; q < 4; ++q)
    {
        cons[q] -= (fr[q] - fl[q]) * dt / dx;
    }
}

__global__ void update_struct_do_cons_to_prim(UpdateStruct update)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i >= update.num_zones)
    {
        return;
    }
    const double *cons = &update.conserved[4 * i];
    /* */ double *prim = &update.primitive[4 * i];
    conserved_to_primitive(cons, prim);
}

int main()
{
    const int num_zones = 1 << 14;
    const int block_size = 256;
    const int fold = 10;
    const double x0 = 0.0;
    const double x1 = 1.0;
    const double dx = (x1 - x0) / num_zones;

    double *primitive = (double*) malloc(num_zones * 4 * sizeof(double));
    struct UpdateStruct update = update_struct_new(num_zones, x0, x1);

    initial_primitive(primitive, num_zones, x0, x1);
    update_struct_set_primitive(update, primitive);

    int iteration = 0;
    double time = 0.0;
    double dt = dx * 0.1;

    while (time < 0.2)
    {
        clock_t start = clock();

        for (int i = 0; i < fold; ++i)
        {
            update_struct_do_compute_flux<<<num_zones / block_size + 1, block_size>>>(update);
            update_struct_do_advance_cons<<<num_zones / block_size + 0, block_size>>>(update, dt);
            update_struct_do_cons_to_prim<<<num_zones / block_size + 0, block_size>>>(update);

            time += dt;
            iteration += 1;
        }
        clock_t end = clock();

        double seconds = ((double) (end - start)) / CLOCKS_PER_SEC;
        double mzps = (num_zones / 1e6) / seconds * fold;
        printf("[%d] t=%.4f Mzps=%.2f\n", iteration, time, mzps);
    }

    update_struct_get_primitive(update, primitive);
    update_struct_del(update);

    FILE* outfile = fopen("euler.dat", "w");

    for (int i = 0; i < num_zones; ++i)
    {
        double *prim = &primitive[i * 4];
        double x = (i + 0.5) * dx;
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
