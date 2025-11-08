def set_config_batch(config,total_samples_per_epoch, total_batch_size, per_gpu_capacity=1):
    #  Samples per epoch
    config.train.total_samples_per_epoch = total_samples_per_epoch  #256
    # config.train.num_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))
    
    assert config.train.total_samples_per_epoch%config.train.num_gpus==0, "total_samples_per_epoch must be divisible by num_gpus"
    config.train.samples_per_epoch_per_gpu = config.train.total_samples_per_epoch//config.train.num_gpus # 64
    
    #  Total batch size
    config.train.total_batch_size = total_batch_size  #128
    assert config.train.total_batch_size%config.train.num_gpus==0, "total_batch_size must be divisible by num_gpus"
    config.train.batch_size_per_gpu = config.train.total_batch_size//config.train.num_gpus  # 32
    config.train.batch_size_per_gpu_available = per_gpu_capacity    # 4
    assert config.train.batch_size_per_gpu%config.train.batch_size_per_gpu_available==0, "batch_size_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.gradient_accumulation_steps = config.train.batch_size_per_gpu//config.train.batch_size_per_gpu_available # 8
    
    assert config.train.samples_per_epoch_per_gpu%config.train.batch_size_per_gpu_available==0, "samples_per_epoch_per_gpu must be divisible by batch_size_per_gpu_available"
    config.train.data_loader_iterations  = config.train.samples_per_epoch_per_gpu//config.train.batch_size_per_gpu_available  # 16
    return config