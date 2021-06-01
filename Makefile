default:
	$(MAKE) -C euler1d_cpu
	$(MAKE) -C euler1d_gpu
	$(MAKE) -C euler1d_uni
	$(MAKE) -C euler2d_cpu
	$(MAKE) -C euler2d_gpu
	$(MAKE) -C euler3d_gpu
	$(MAKE) -C sr1d_cpu
	$(MAKE) -C sr1d_gpu

clean:
	$(MAKE) -C euler1d_cpu clean
	$(MAKE) -C euler1d_gpu clean
	$(MAKE) -C euler1d_uni clean
	$(MAKE) -C euler2d_cpu clean
	$(MAKE) -C euler2d_gpu clean
	$(MAKE) -C euler3d_gpu clean
	$(MAKE) -C sr1d_cpu clean
	$(MAKE) -C sr1d_gpu clean
