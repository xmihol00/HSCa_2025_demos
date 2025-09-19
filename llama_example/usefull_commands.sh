# reserve GPU with 48GB memory and some data transfer from the matylda5 data server
qlogin -q long.q -l gpu=1,mem_free=48G,gpu_ram=48G,matylda5=5

# login to a specific GPU node
qlogin -q long.q@supergpu19

# check the GPU utilization
watch -n 0.1 nvidia-smi
