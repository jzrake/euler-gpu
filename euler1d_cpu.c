#include <time.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "lib_euler1d.c"




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
int main()
{
    const int num_zones = 1 << 12;
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
