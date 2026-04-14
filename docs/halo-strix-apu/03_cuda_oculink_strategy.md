# 03 — RTX 4060 Ti over OCuLink: When It Helps and When It Does Not

## 1. The question

You asked whether it would make sense to connect an **RTX 4060 Ti 16GB** through **OCuLink** and use it only for the operations where CUDA is stronger than ROCm.

The short answer is:

> **Usually no for per-op offload.**
> **Potentially yes for whole-workload offload.**

## 2. Why per-op offload is usually the wrong idea

The bandwidth hierarchy is the key.

### OCuLink
A common OCuLink eGPU setup is effectively **PCIe 4.0 x4**, which vendors commonly advertise as **up to 64 Gbps**.  
Example product references:  
<https://www.adt.link/product/F9GV4.html>  
<https://store.minisforum.com/products/minisforum-deg2-oculink-egpu-dock>

That is roughly:

- **8 GB/s theoretical raw bandwidth**

### RTX 4060 Ti 16GB
NVIDIA’s official page lists the RTX 4060 Ti with:

- **16 GB or 8 GB GDDR6**
- **128-bit memory interface**
- **4th generation Tensor Cores**
- **PCIe Gen 4**

Source: NVIDIA official specs  
<https://www.nvidia.com/en-eu/geforce/graphics-cards/40-series/rtx-4060-4060ti/>

A board partner spec page makes the lane usage and memory speed explicit:

- **PCIe Gen 4 x16 (uses x8)**
- **18 Gbps memory speed**
- **128-bit memory bus**

Source: MSI spec page  
<https://www.msi.com/Graphics-Card/GeForce-RTX-4060-Ti-GAMING-X-16G/Specification>

That implies a local VRAM bandwidth of roughly:

- `18 Gbps × 128 / 8 = 288 GB/s`

### Practical interpretation
An ~8 GB/s transport link is tiny compared with ~288 GB/s device-local VRAM bandwidth.

So if you do this pattern:

1. compute a few layers on Halo Strix
2. ship activations to the 4060 Ti
3. run a faster CUDA op
4. ship activations back

you are asking a very slow interconnect to sit inside the critical path of each step.

That is normally a losing pattern.

## 3. A useful back-of-the-envelope latency check

At an ideal **8 GB/s**:

- **1 GB** transfer ≈ **125 ms** one way
- **4 GB** transfer ≈ **500 ms** one way

Real throughput will usually be lower.

This means a CUDA-only optimization that saves, say, 10–30 ms on a specific op is not enough if you had to move hundreds of MB or GB to use it.

## 4. When the 4060 Ti does make sense

The 4060 Ti can still be useful in two cases.

### Case A: the whole model runs on the 4060 Ti
This is the best CUDA use case. You avoid ping-pong and keep:

- parameters
- activations
- optimizer state
- gradients

on one device.

### Case B: a coarse stage runs on the 4060 Ti
This only makes sense if the boundary crossings are rare and the compute block on CUDA is large enough to amortize the copy cost.

Examples of coarse offload that *might* be viable:
- an entire inference workload
- an entire training run for a custom architecture
- a very large isolated stage that runs for long enough before data must return

## 5. What not to do

Avoid:
- sending attention to CUDA and the rest to ROCm
- sending only matmuls to CUDA
- per-layer or per-block device switching
- fine-grained hybrid scheduling across AMD + NVIDIA over OCuLink

That style of setup usually loses to simply picking one device for the whole workload.

## 6. Why this matters specifically for your custom models

For your standard fused Llama-like path, Halo Strix is already delivering about **40k tok/s**, which is very respectable for the platform.

The 4060 Ti question is more relevant for your **custom architectures**, where the problem is not only raw FLOPs but also:

- weaker backend maturity on gfx1151 for some paths
- worse fusion
- more irregular execution

In those cases, CUDA may genuinely map the whole model better.

But again, the useful comparison is:

- **entire custom model on Halo Strix**
versus
- **entire custom model on 4060 Ti**

not “some ops here, some ops there.”

## 7. The real strategic use of the 4060 Ti

The strongest use of the 4060 Ti in your setup is:

> a separate execution target for workloads that map poorly to gfx1151

That means:
- benchmark the **whole custom model** on the 4060 Ti
- compare full end-to-end tok/s
- compare stability and ease of optimization
- only keep the 4060 Ti in the picture if it wins at the **whole-workload** level

## 8. Bottom-line recommendation

### Good idea
- run the **entire custom architecture** on the 4060 Ti and compare against the entire model on Halo Strix

### Bad idea
- use the 4060 Ti as a “special matmul helper” while the APU GPU handles the rest

## References

- NVIDIA RTX 4060 Ti official specs:  
  <https://www.nvidia.com/en-eu/geforce/graphics-cards/40-series/rtx-4060-4060ti/>
- MSI RTX 4060 Ti 16G specification page:  
  <https://www.msi.com/Graphics-Card/GeForce-RTX-4060-Ti-GAMING-X-16G/Specification>
- ADT-Link OCuLink eGPU adapter example:  
  <https://www.adt.link/product/F9GV4.html>
- Minisforum OCuLink eGPU dock example:  
  <https://store.minisforum.com/products/minisforum-deg2-oculink-egpu-dock>
