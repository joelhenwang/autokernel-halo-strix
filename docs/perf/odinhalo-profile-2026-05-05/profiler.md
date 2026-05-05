# OdinHalo deep profile — 2026-05-05

**Config:** OdinHalo (57.6M params) | batch=16 | block=256 | fused AdamW | fp16 autocast | warmup=25 | measured=5

**Total GPU time measured:** 6166.8 μs across 21974 op calls over 3 active profiler steps.


## Categorized breakdown

| Category | Self CUDA μs | % of wall | Top ops |
|----------|-------------:|----------:|---------|
| elementwise | 2,514 | 40.8% | `aten::mul`, `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul> >(int, at::native::BinaryFunctor<float, float, float, at::native::binary_internal::MulFunctor<float> >, std::array<char*, 3ul>)`, `aten::add_` |
| other | 1,550 | 25.1% | `ProfilerStep*`, `Memset (Device)`, `aten::sum` |
| matmul | 1,061 | 17.2% | `aten::mm`, `Cijk_Ailk_Bjlk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB2_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA2048_LBSPPB3072_LBSPPM0_LPA16_LPB16_LPM0_LRVW16_LWPMn1_MIAV1_MIWT2_3_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA1_NLCB3_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS0_TLDSM1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1`, `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1_SN_LDSB0_AFC1_AG0_AFEM1_AFEM1_ASEM1_CLR1_CADS0_DTLA0_DTLB0_DTVA0_DTVB0_DTVSM0_DPLB0_EPS0_ELFLR0_EMLLn1_FDSI0_GRPM1_GRVWA8_GRVWB8_GSUAMB_GLS0_ISA1151_IU1_K1_LDSTI0_LBSPPA3072_LBSPPB128_LBSPPM0_LPA16_LPB16_LPM0_LRVW16_LWPMn1_MIAV1_MIWT3_3_MO40_MGRIPM1_NTn1_NTA0_NTB0_NTC0_NTD0_NTM0_NEPBS0_NLCA3_NLCB1_ONLL1_PGR2_PLR1_PKA0_SGROB0_SIA3_SS1_SPO0_SRVW0_SSO0_SVW1_SK0_SKFTR0_SKXCCM0_SGRO0_TIN0_TLDS1_TLDSM1_ULSGRO0_USL1_UIOFGRO0_UPLRP0_USFGROn1_VSn1_VWA1_VWB1_WSGRA0_WSGRB0_WS32_WG32_4_1` |
| copy | 808 | 13.1% | `aten::copy_`, `Memcpy HtoD (Host -> Device)`, `void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul> >(int, at::native::float16_copy_kernel_cuda(at::TensorIteratorBase&)::{lambda(float)#1}, std::array<char*, 2ul>)` |
| conv | 104 | 1.7% | `void causal_conv1d_channellast_bwd_kernel<Causal_conv1d_channellast_bwd_kernel_traits<128, 3, 128, false, true, float, float>, false, false, false>(ConvParamsBwd)`, `DaoAILab::_causal_conv1d_bwd_cpp`, `void causal_conv1d_channellast_fwd_kernel<Causal_conv1d_channellast_fwd_kernel_traits<128, 3, 64, true, float, float>, false>(ConvParamsBase)` |
| optimizer | 80 | 1.3% | `Optimizer.step#AdamW.step`, `aten::_fused_adamw_`, `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(anonymous namespace)::TensorListMetadata<1>, at::native::(anonymous namespace)::UnaryOpFunctor<float, 1, 1, 0>, at::native::_amp_foreach_non_finite_check_and_unscale_cuda_(c10::ArrayRef<at::Tensor>, at::Tensor&, at::Tensor const&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1}>(at::native::(anonymous namespace)::TensorListMetadata<1>, at::native::(anonymous namespace)::UnaryOpFunctor<float, 1, 1, 0>, at::native::_amp_foreach_non_finite_check_and_unscale_cuda_(c10::ArrayRef<at::Tensor>, at::Tensor&, at::Tensor const&)::{lambda()#1}::operator()() const::{lambda()#2}::operator()() const::{lambda(float)#1})` |
| attention | 49 | 0.8% | `void at::native::(anonymous namespace)::cunn_SoftMaxBackward<8, c10::Half, float, c10::Half, at::native::(anonymous namespace)::LogSoftMaxBackwardEpilogue>(c10::Half*, c10::Half const*, c10::Half const*, long)`, `aten::_log_softmax_backward_data`, `void at::native::(anonymous namespace)::cunn_SoftMaxForward<8, c10::Half, float, c10::Half, at::native::(anonymous namespace)::LogSoftMaxForwardEpilogue>(c10::Half*, c10::Half const*, int)` |
| loss | 0 | 0.0% | `void at::native::(anonymous namespace)::nll_loss_forward_reduce_cuda_kernel_2d<float, float, long>(float*, float*, float const*, long const*, float const*, bool, long, long, long, long)`, `aten::nll_loss_forward`, `void at::native::(anonymous namespace)::nll_loss_backward_reduce_cuda_kernel_2d<float, long>(float*, float const*, long const*, float const*, float const*, bool, int, int, long, long)` |

## Top 40 ops by self CUDA time

| # | Name | Category | Self CUDA μs | % | Calls | μs/call |
|--:|------|----------|-------------:|--:|------:|--------:|
| 1 | `ProfilerStep*` | other | 1,109.2 | 17.99% | 3 | 369.74 |
| 2 | `aten::mul` | elementwise | 526.8 | 8.54% | 1785 | 0.30 |
| 3 | `aten::mm` | matmul | 451.3 | 7.32% | 744 | 0.61 |
| 4 | `void at::native::vectorized_elementwise_kernel<4, at::native::BinaryFunctor<floa` | elementwise | 432.0 | 7.01% | 736 | 0.59 |
| 5 | `aten::add_` | elementwise | 383.1 | 6.21% | 561 | 0.68 |
| 6 | `void at::native::vectorized_elementwise_kernel<4, at::native::CUDAFunctor_add<fl` | elementwise | 377.5 | 6.12% | 480 | 0.79 |
| 7 | `aten::copy_` | copy | 372.6 | 6.04% | 1323 | 0.28 |
| 8 | `aten::embedding_dense_backward` | elementwise | 296.4 | 4.81% | 3 | 98.80 |
| 9 | `Memset (Device)` | other | 296.1 | 4.80% | 16 | 18.51 |
| 10 | `Memcpy HtoD (Host -> Device)` | copy | 289.0 | 4.69% | 6 | 48.17 |
| 11 | `Cijk_Ailk_Bjlk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_LDSB0_AFC1_` | matmul | 231.1 | 3.75% | 244 | 0.95 |
| 12 | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT96x96x32_MI16x16x1_SN_LDSB0_AFC1_` | matmul | 158.5 | 2.57% | 288 | 0.55 |
| 13 | `aten::sum` | other | 64.1 | 1.04% | 375 | 0.17 |
| 14 | `void at::native::reduce_kernel<128, 4, at::native::ReduceOp<float, at::native::f` | elementwise | 60.0 | 0.97% | 208 | 0.29 |
| 15 | `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x80x64_MI16x16x1_SN_LDSB1_AFC1` | matmul | 57.5 | 0.93% | 58 | 0.99 |
| 16 | `void causal_conv1d_channellast_bwd_kernel<Causal_conv1d_channellast_bwd_kernel_t` | conv | 51.5 | 0.84% | 60 | 0.86 |
| 17 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel` | elementwise | 48.9 | 0.79% | 183 | 0.27 |
| 18 | `void at::native::vectorized_elementwise_kernel<4, at::native::float16_copy_kerne` | copy | 46.9 | 0.76% | 704 | 0.07 |
| 19 | `Cijk_Alik_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT128x128x32_MI16x16x1_SN_LDSB0_AFC` | matmul | 43.2 | 0.70% | 129 | 0.34 |
| 20 | `DaoAILab::_causal_conv1d_bwd_cpp` | conv | 38.7 | 0.63% | 45 | 0.86 |
| 21 | `void at::native::(anonymous namespace)::CatArrayBatchedCopy_contig<at::native::(` | copy | 38.6 | 0.63% | 140 | 0.28 |
| 22 | `void at::native::(anonymous namespace)::multi_tensor_apply_kernel<at::native::(a` | matmul | 36.2 | 0.59% | 12 | 3.02 |
| 23 | `void at::native::vectorized_elementwise_kernel<4, at::native::float16tofloat32_c` | copy | 32.2 | 0.52% | 483 | 0.07 |
| 24 | `void at::native::elementwise_kernel_manual_unroll<128, 4, at::native::gpu_kernel` | elementwise | 31.7 | 0.51% | 703 | 0.05 |
| 25 | `aten::cat` | other | 31.6 | 0.51% | 126 | 0.25 |
| 26 | `Optimizer.step#AdamW.step` | optimizer | 27.2 | 0.44% | 3 | 9.08 |
| 27 | `aten::_fused_adamw_` | optimizer | 27.2 | 0.44% | 3 | 9.07 |
| 28 | `Cijk_Ailk_Bljk_HHS_BH_Bias_HA_S_SAV_UserArgs_MT64x96x32_MI16x16x1_SN_LDSB0_AFC1_` | matmul | 25.9 | 0.42% | 8 | 3.24 |
| 29 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel` | elementwise | 24.9 | 0.40% | 72 | 0.35 |
| 30 | `void at::native::vectorized_elementwise_kernel<8, at::native::BinaryFunctor<c10:` | elementwise | 23.7 | 0.38% | 87 | 0.27 |
| 31 | `void at::native::vectorized_elementwise_kernel<8, at::native::CUDAFunctor_add<c1` | elementwise | 23.3 | 0.38% | 455 | 0.05 |
| 32 | `void at::native::reduce_kernel<512, 1, at::native::ReduceOp<float, at::native::f` | elementwise | 22.4 | 0.36% | 232 | 0.10 |
| 33 | `aten::add` | elementwise | 21.9 | 0.36% | 444 | 0.05 |
| 34 | `aten::div` | elementwise | 21.1 | 0.34% | 252 | 0.08 |
| 35 | `Memcpy DtoD (Device -> Device)` | copy | 20.0 | 0.32% | 200 | 0.10 |
| 36 | `aten::silu_backward` | elementwise | 18.7 | 0.30% | 54 | 0.35 |
| 37 | `void at::native::vectorized_elementwise_kernel<8, at::native::BUnaryFunctor<c10:` | elementwise | 18.6 | 0.30% | 8 | 2.33 |
| 38 | `void at::native::vectorized_elementwise_kernel<8, at::native::AUnaryFunctor<c10:` | elementwise | 18.4 | 0.30% | 8 | 2.30 |
| 39 | `void at::native::(anonymous namespace)::cunn_SoftMaxBackward<8, c10::Half, float` | attention | 15.1 | 0.25% | 4 | 3.78 |
| 40 | `void at::native::elementwise_kernel_manual_unroll<128, 8, at::native::gpu_kernel` | elementwise | 14.6 | 0.24% | 58 | 0.25 |

## Sanity check

Category totals sum: 6,167 μs
Raw event total: 6,167 μs
Delta: 0.00% (must be < 5%)
