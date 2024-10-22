JobBatchName            = "attention replace"
universe                = vanilla

# Artefact
Requirements            = (Target.HasCHTCStaging == true)
executable              = run.sh
transfer_input_files    = models.py,main.py,datasets.py,train.py,utils.py,loss.py,environment.yml,cifar-100-python.tar.gz
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT

# Logging
stream_output           = true
output = $(Cluster)_$(Process).out
error  = $(Cluster)_$(Process).err

# Compute resources
request_cpus            = 4
request_memory          = 16GB
request_disk            = 20GB

# Extra GPU settings
request_gpus            = 1
Requirements            = (Target.CUDADriverVersion >= 10.1)
+WantGPULab             = true
# change to true if *not* using staging for checkpoints and interested in accessing GPUs beyond CHTC
+WantFlocking           = false
+WantGlidein            = false
+GPUJobLength           = "short"

# Runs
queue 1

