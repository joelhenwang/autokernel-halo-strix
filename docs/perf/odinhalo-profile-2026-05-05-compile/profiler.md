# OdinHalo deep profile — 2026-05-05 (compile_zones)

**Config:** OdinHalo (57.6M params) | batch=16 | block=256 | fused AdamW | fp16 autocast | mode=compile_zones | warmup=25 | measured=5

**Total GPU time measured:** 5456.4 μs across 8790 op calls over 3 active profiler steps.


## Categorized breakdown

| Category | Self CUDA μs | % of wall | Top ops |
|----------|-------------:|----------:|---------|
| other | 1,918 | 35.1% | `ProfilerStep*`, `## Call CompiledFxGraph fd4ufndp56zs5zzf263fct7ihngr4edrw2gwdprjqwgdpiuuwedh ##`, `Memset (Device)` |
| copy | 1,145 | 21.0% | `triton_poi_fused__to_copy_mul_transpose_view_8`, `triton_poi_fused__to_copy_mul_transpose_view_8`, `aten::copy_` |
| matmul | 1,113 | 20.4% | `aten::mm`, `Cijk_Ailk_Bjlk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB2_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB3072_LBSPPM0_LPA16_LPB16_LPM0_LRVW16_LWPMn1_MIAV1_MIWT2_3_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB3_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS0_TLDSM1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1`, `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA3072_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW16_LWPMn1_MIAV1_MIWT3_3_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA3_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSM1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1` |
| elementwise | 1,054 | 19.3% | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul> >(int, at::native::CUDAFunctor_add<float>, std::array<char*, 3ul>)`, `aten::add_`, `aten::embedding_dense_backward` |
| conv | 97 | 1.8% | `void causal_conv1d_channellast_bwd_kernel<Causal_conv1d_channellast_bwd_kernel_traits<128, 3, 128, false, true, float, float>, false, false, false>(ConvParamsBwd)`, `DaoAILab::_causal_conv1d_bwd_cpp`, `void causal_conv1d_channellast_fwd_kernel<Causal_conv1d_channellast_fwd_kernel_traits<128, 3, 64, true, float, float>, false>(ConvParamsBase)` |
| optimizer | 79 | 1.5% | `Optimizer.step#AdamW.step`, `aten::_fused_adamw_`, `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListMetadata<1>, at::native::(anonymous namespace)::UnaryOpFunctor<float, 1, 1, 0>, at::native::_amp_foreach_non_finite_check_and_unscale_cuda_(c10::ArrayRef<at::Tensor>, at::Tensor&, at::Tensor const&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::native::(anonymous namespace)::TensorListMetadata<1>, at::native::(anonymous namespace)::UnaryOpFunctor<float, 1, 1, 0>, at::native::_amp_foreach_non_finite_check_and_unscale_cuda_(c10::ArrayRef<at::Tensor>, at::Tensor&, at::Tensor const&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})` |
| attention | 49 | 0.9% | `void at::native::(anonymous namespace)::cunn_SoftMaxBackward<8, c10::Half, float, c10::Half, at::native::(anonymous namespace)::LogSoftMaxBackwardEpilogue>(c10::Half*, c10::Half const*, c10::Half const*, long)`, `aten::_log_softmax_backward_data`, `void at::native::(anonymous namespace)::cunn_SoftMaxForward<8, c10::Half, float, c10::Half, at::native::(anonymous namespace)::LogSoftMaxForwardEpilogue>(c10::Half*, c10::Half const*, int)` |
| loss | 0 | 0.0% | `void at::native::(anonymous namespace)::nll_loss_forward_reduce_cuda_kernel_2d<float, float, long>(float*, float*, float const*, long const*, float const*, bool, long, long, long, long)`, `aten::nll_loss_forward`, `void at::native::(anonymous namespace)::nll_loss_backward_reduce_cuda_kernel_2d<float, long>(float*, float const*, long const*, float const*, float const*, bool, int, int, long, long)` |

## Top 40 ops by self CUDA time

| # | Name | Category | Self CUDA μs | % | Calls | μs/call |
|--:|------|----------|-------------:|--:|------:|--------:|
| 1 | `ProfilerStep*` | other | 859.7 | 15.75% | 3 | 286.55 |
| 2 | `aten::mm` | matmul | 465.0 | 8.52% | 744 | 0.63 |
| 3 | `## Call CompiledFxGraph fd4ufndp56zs5zzf263fct7ihngr4edrw2gwdprjqwgdpiuuwedh ##` | other | 424.9 | 7.79% | 30 | 14.16 |
| 4 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<fl` | elementwise | 274.6 | 5.03% | 512 | 0.54 |
| 5 | `aten::add_` | elementwise | 266.9 | 4.89% | 402 | 0.66 |
| 6 | `triton_poi_fused__to_copy_mul_transpose_view_8` | copy | 249.6 | 4.57% | 40 | 6.24 |
| 7 | `triton_poi_fused__to_copy_mul_transpose_view_8` | copy | 248.5 | 4.56% | 30 | 8.28 |
| 8 | `aten::copy_` | copy | 242.2 | 4.44% | 129 | 1.88 |
| 9 | `Cijk_Ailk_Bjlk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_LDSB0_AFC1_` | matmul | 240.5 | 4.41% | 244 | 0.99 |
| 10 | `aten::embedding_dense_backward` | elementwise | 222.0 | 4.07% | 3 | 73.99 |
| 11 | `Memset (Device)` | other | 221.7 | 4.06% | 16 | 13.85 |
| 12 | `Memcpy HtoD (Host -> Device)` | copy | 215.5 | 3.95% | 6 | 35.92 |
| 13 | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1_SN_LDSB0_AFC1_` | matmul | 163.6 | 3.00% | 288 | 0.57 |
| 14 | `## Call CompiledFxGraph fr2ikae2bjuc5xuug5myzvpfdybdxh3nl7buyi3v6v5dxpagzsre ##` | other | 86.0 | 1.58% | 15 | 5.73 |
| 15 | `## Call CompiledFxGraph fjpkvbcht67kzb2dqndhjwyle5eupgdtgimcclmaubxi23qukdsm ##` | other | 78.4 | 1.44% | 30 | 2.61 |
| 16 | `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x80x64_MI16x16x1_SN_LDSB1_AFC1` | matmul | 54.9 | 1.01% | 56 | 0.98 |
| 17 | `void causal_conv1d_channellast_bwd_kernel<Causal_conv1d_channellast_bwd_kernel_t` | conv | 47.9 | 0.88% | 60 | 0.80 |
| 18 | `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC` | matmul | 42.7 | 0.78% | 123 | 0.35 |
| 19 | `## Call CompiledFxGraph fysj6tdjm4a3gy5nydtbprsuz3io4xbmnc6nspymbcjizju2deao ##` | other | 36.4 | 0.67% | 15 | 2.42 |
| 20 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(a` | matmul | 36.2 | 0.66% | 12 | 3.02 |
| 21 | `DaoAILab::_causal_conv1d_bwd_cpp` | conv | 35.9 | 0.66% | 45 | 0.80 |
| 22 | `## Call CompiledFxGraph fganol4fnn6cjz6lpju2emxwve5z54rzkjtju3xgviib6w2rtkgv ##` | other | 31.5 | 0.58% | 30 | 1.05 |
| 23 | `aten::mul` | elementwise | 29.9 | 0.55% | 228 | 0.13 |
| 24 | `Optimizer.step#AdamW.step` | optimizer | 27.2 | 0.50% | 3 | 9.07 |
| 25 | `aten::_fused_adamw_` | optimizer | 27.2 | 0.50% | 3 | 9.06 |
| 26 | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_LDSB0_AFC1_` | matmul | 26.8 | 0.49% | 8 | 3.35 |
| 27 | `## Call CompiledFxGraph fvezn3ufrspmnzwvtt4upxh5ugqvd66owcn3bu4u5fyfch46juo3 ##` | other | 26.8 | 0.49% | 3 | 8.94 |
| 28 | `## Call CompiledFxGraph fphiggvqsuffntgmrcgsegkyar46oehvxemmjwkmmzkawmocccfd ##` | matmul | 25.9 | 0.48% | 3 | 8.64 |
| 29 | `## Call CompiledFxGraph f4xeeufzyql6wnjamwtkiktt6j2sfkwifawf23gmiqtr7pm5paou ##` | other | 25.6 | 0.47% | 3 | 8.54 |
| 30 | `triton_poi_fused__unsafe_view_cat_mul_silu_silu_backward_split_view_2` | elementwise | 24.9 | 0.46% | 48 | 0.52 |
| 31 | `void at::native::vectorized_elementwise_kernel<8, at::native::BUnaryFunctor<c10:` | elementwise | 18.7 | 0.34% | 8 | 2.34 |
| 32 | `triton_poi_fused__unsafe_view_cat_mul_silu_silu_backward_split_view_2` | elementwise | 18.7 | 0.34% | 36 | 0.52 |
| 33 | `void at::native::vectorized_elementwise_kernel<8, at::native::AUnaryFunctor<c10:` | elementwise | 18.4 | 0.34% | 8 | 2.30 |
| 34 | `void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kerne` | copy | 17.5 | 0.32% | 47 | 0.37 |
| 35 | `void at::native::vectorized_elementwise_kernel<4, at::native::float16tofloat32_c` | copy | 15.4 | 0.28% | 31 | 0.50 |
| 36 | `void at::native::(anonymous namespace)::cunn_SoftMaxBackward<8, c10::Half, float` | attention | 15.1 | 0.28% | 4 | 3.77 |
| 37 | `Cijk_SH_BiasS_HAS_ScaleAlphaVec_PostGSU2_VW4` | matmul | 14.7 | 0.27% | 136 | 0.11 |
| 38 | `## Call CompiledFxGraph f3dqpltvpdtn7f73vnlhp43vxcuah2sma4pwdv7ccorecwbuscag ##` | other | 14.7 | 0.27% | 15 | 0.98 |
| 39 | `triton_poi_fused__unsafe_view_mul_silu_split_5` | elementwise | 14.3 | 0.26% | 47 | 0.30 |
| 40 | `aten::div` | elementwise | 14.3 | 0.26% | 18 | 0.79 |

## Sanity check

Category totals sum: 5,456 μs
Raw event total: 5,456 μs
Delta: 0.00% (must be < 5%)
