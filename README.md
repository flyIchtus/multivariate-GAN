# multivariate-GAN
Public repository for article "Multivariate emulation of convective-scale numerical weather predictions with generative adversarial networks: a proof-of-concept"


*gan_horovod* : contains up-to-date GAN training logics, networks architectures and data pipelining, interfacing horovod API.

*gan_std* : is the original library (not up-to-date) containing basic training logics and no interface for multi-GPU.

*metrics4arome* : contains the implementations of the many metrics used to compare GAN and PEARO outputs, together with short snippets to test them. Includes spectral analysis, Wasserstein distances implementations and scattering transform analysis.


# Requirements :
torch, numpy, horovod, kymatio
