# multivariate-GAN
Public repository for article "Multivariate emulation of convective-scale numerical weather predictions with generative adversarial networks: a proof-of-concept"


*gan_horovod* : contains up-to-date GAN training logics, networks architectures and data pipelining, interfacing horovod API.

*gan_std* : is the original library containing complete training logics, networks architectures and data pipelining but no interface for multi-GPU.

*metrics4arome* : contains the implementations of the many metrics used to compare GAN and PEARO outputs, together with short snippets to test them. Includes spectral analysis, Wasserstein distances implementations and scattering transform analysis.


# Requirements :
torch v>=1.7, numpy, horovod (for gan_horovod only, see https://horovod.readthedocs.io/en/stable/ for docs), kymatio (latest, see https://www.kymat.io/)


# how to use :

*gan_std* : python3 main.py --data_dir 'my_data_dir' --output_dir 'my_output_dir'
*gan_horovod* , with e.g 4 GPUs : horovodrun -np 4 -H localhost:4 main.py -data_dir 'my_data_dir' --output_dir 'my_output_dir'. Note that the batch size used here is a *per-GPU* batch size.
