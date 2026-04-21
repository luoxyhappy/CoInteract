# CoInteract: Spatially-Structured Co-Generation for Interactive Human-Object Video Synthesis



<p align="center">
  <a href="https://luoxyhappy.github.io/">Xiangyang Luo</a><sup>1,2*</sup>, Xiaozhe Xin<sup>2*✉</sup>, Tao Feng<sup>1</sup>, <a href="https://guoxu1233.github.io/homepage/">Xu Guo</a><sup>1</sup>, Meiguang Jin<sup>2</sup>, Junfeng Ma<sup>2</sup>
  <br>
  <sup>1</sup> Tsinghua University &nbsp; <sup>2</sup> Alibaba Group
  <br>
  <sup>*</sup> Equal contribution &nbsp; <sup>✉</sup> Corresponding author
</p>

<p align="center">
  <a href="https://arxiv.org/abs/xxxx.xxxxx"><img src="https://img.shields.io/badge/arXiv-Paper-red"></a>
  <a href="https://huggingface.co/your-username/CoInteract"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-Model-yellow"></a>
  <a href="https://xinxiaozhe12345.github.io/CoInteract_Project/"><img src="https://img.shields.io/badge/Project-Page-blue"></a>
</p>

## Demo
<p align="center">
  <video src="https://github.com/user-attachments/assets/fe59a768-38e8-4403-bd13-155662c628d6" controls width="100%"></video>
</p>

## Roadmap

| Stage | Status | Description |
|-------|--------|-------------|
| 1 | 🔜 | Release inference code and model weights (within one week) |
| 2 | 🔜 | Release training code |
| 3 | 📋 | Add pose control support |

## Highlights

CoInteract enables high-quality **speech-driven human-object interaction video synthesis** with fine-grained spatial control. It supports diverse generation modes including video generation, unified generation, and interactive generation.

<p align="center">
  <img src="assets/teaser.jpg" width="100%">
</p>

Key contributions:

- **Human-Aware Mixture-of-Experts (MoE)**: A spatial routing mechanism that dynamically dispatches tokens to specialized expert networks (hand expert + face expert), supervised by GT bounding boxes during training and fully automatic at inference.
- **Spatially-Structured Co-Generation**: Joint training of RGB video and HOI depth maps provides structural guidance for realistic interactions, without requiring depth input at inference time.
- **Audio Face Mask**: Constrains the audio cross-attention residual to the face region, preventing audio signals from leaking into hand/object areas and improving lip-sync quality.

<p align="center">
  <img src="assets/pipeline.jpg" width="70%">
</p>

## Citation

```bibtex
soooooon
```

<!-- ## Acknowledgments

- [DiffSynth-Studio](https://github.com/modelscope/DiffSynth-Studio)
- [Wan2.2](https://github.com/Wan-Video/Wan2.2)
- [chinese-wav2vec2-large](https://huggingface.co/TencentGameMate/chinese-wav2vec2-large) -->
