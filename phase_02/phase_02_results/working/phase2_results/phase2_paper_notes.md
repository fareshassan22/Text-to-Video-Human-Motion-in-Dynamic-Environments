
# Phase 2 Paper Notes

## Weakness Analysis Summary

Phase 1 showed strong semantic alignment according to CLIP-SIM, but temporal consistency remained the main weakness. The complex and dynamic scenes produced lower SSIM and visible frame-to-frame instability. This motivated Phase 2 to focus on reducing flicker while preserving prompt alignment.

## Proposed Enhancements

### Enhancement A: Prompt Conditioning and CFG

The first enhancement improves semantic conditioning using a richer prompt, a negative prompt, and a higher classifier-free guidance scale. CFG changes the denoising direction by increasing the difference between conditional and unconditional predictions, which can improve text-video alignment.

### Enhancement B: Temporal Stabilization

The second enhancement reduces temporal instability by lowering SVD motion/noise parameters, then applying a temporal low-pass filter and frame interpolation. The filter follows:

`I_smooth[t] = alpha * I[t] + (1 - alpha) * I_smooth[t-1]`

This suppresses high-frequency frame flicker while preserving most of the original motion.

## Ablation Results

| Variant | CLIP-SIM | Mean SSIM | Mean Frame Diff | Duration (s) | Temporal | Semantic |
|---|---:|---:|---:|---:|---|---|
| baseline | 34.220 | 0.705 | 19.999 | 3.57 | moderate | good |
| enhancement_A_prompt_cfg | 33.801 | 0.806 | 10.172 | 3.57 | moderate | good |
| enhancement_B_temporal | 34.951 | 0.957 | 4.496 | 4.08 | good | good |
| combined_A_B | 33.646 | 0.947 | 3.954 | 4.08 | good | good |

## Interpretation

Compared with the baseline, the combined enhancement changed CLIP-SIM by -0.573 and mean SSIM by 0.242. The final video duration is 4.08 seconds, satisfying the Phase 2 minimum duration requirement of 4 seconds.

## Limitations

Temporal smoothing can reduce flicker, but excessive smoothing may blur fast motion or reduce motion realism. Therefore, the final interpretation should include visual inspection in addition to SSIM, PSNR, and CLIP-SIM.

## Conclusion

Phase 2 implemented and evaluated two targeted enhancements: semantic prompt conditioning and temporal stabilization. The ablation study directly compares the baseline, each individual enhancement, and the combined system, providing quantitative and qualitative evidence for the final paper.
