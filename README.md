# TimeColor: Flexible Reference Colorization via Temporal Concatenation

<a href="https://bconstantine.github.io/TimeColor"><img src="https://img.shields.io/static/v1?label=Project&message=Website&color=blue"></a>
<a href="https://arxiv.org/abs/2601.00296"><img src="https://img.shields.io/badge/arXiv-2601.00296-b31b1b.svg"></a>
<a href="https://www.apache.org/licenses/LICENSE-2.0.txt"><img src="https://img.shields.io/badge/License-Apache-yellow"></a>

> <a href="https://bconstantine.github.io/TimeColor/">**TimeColor: Flexible Reference Colorization via Temporal Concatenation**</a>

**Please see our [demo page](https://bconstantine.github.io/TimeColor) for results**

</p>


## Requirements

TimeColor is implemented using PyTorch. Training were performed on 6 NVIDIA A40 GPUs (FSDP + Gradient Accumulation 2), and inference were performed on 1 NVIDIA A40 GPU (VAE Slicing + Tiling, CPU Offload). 

## Setup
```bash
git clone https://github.com/bconstantine/TimeColor.git
cd TimeColor
```

## Environment
We use separate `conda` environments for data preprocessing and inference.

For sketch_preprocessing, please use the following environment:
```bash
conda create -n timecolor_sketch_processing.yml
conda activate timecolor_sketch_processing
```

To set up the inference environment, use:

```bash
conda env create -f timecolor.yml
conda activate timecolor
#separately install flash-attn to avoid environment's pip dependencies conflict
pip install flash-attn==2.7.4.post1 --no-build-isolation
#patch custom attention alternative
./scripts/patch_diffusers.sh
#install xDiT if you want to use cfg parallelism with flash attn
pip install "xfuser[flash-attn]" 
```

## Dataset
We used the SAKUGA Dataset as the basis for our training, validation, and test datasets. This dataset can be found on GitHub [here](https://github.com/KytraScript/SakugaDataset). Training and testing are done on the training and testing split of the SAKUGA respectively. Shout out to the authors for providing large scale dataset, which is a vital part of our model creation!

## Sketch Generation
To run the sketch generation, first download the `netG_A_latest` model, available from the [InformativeDrawings](https://github.com/carolineec/informative-drawings) repository. After putting those weights into the `checkpoint/sketch_generation/` folder, sketch generation can be run as follows:
```bash
conda activate timecolor_sketch_processing
./scripts/sketch_generation.sh
```
Specify your input and output folder inside the shell script. The script will convert all nested mp4 in the input folder, and output them in the output folder, following the same nested relative path.

## Checkpoints
Our full weights can be downloaded [here](https://cloud.tsinghua.edu.cn/d/f99e8c98d9e144ddbaa4/). Paste the `model_weights/` folder into `checkpoint/TimeColor-final/`
### ⚠️ Scope & Variability Limitation
TimeColor is a research model trained with constrained compute and development resources for sketch-guided video colorization. As a generative system, outputs are stochastic and may vary across inputs and runs. To explore different outcomes, please adjust seed, steps, prompt, guidance_scale (CFG), and sketch/reference settings.

## Inference
Our custom inference script is based on [`finetrainers`](https://github.com/a-r-r-o-w/finetrainers) and ['xDiT'](https://github.com/xdit-project/xDiT), using `CogVideoX-5B` as our base model. To run inference, run the provided script, we give two alternatives, using xDiT (parallel GPU, faster speed) or using single GPU

```bash
#xDiT inference
conda activate timecolor
./scripts/colorize_cfgparallel.sh

#regular inference
conda activate timecolor
./scripts/colorize_singlegpu.sh
```

You may configure GPU use by modifying each `.sh` file. 

We have tested our model performace up to 49 frame counts, the default window for the CogVideoX-5B base. Following the base model, inputs are expected at 480x720 HxW and frame count must be `4N+1, N>3`. 

#### Input format
For input representation, please see the example input json (e.g. `./examples/inference_samples_xdit.json`)
```jsonc
{
  "data": [
    //single reference data case
    {
      "caption": "sample caption",
      "sketch_control_path": "sketch input mp4", //relative to repo directory
      "reference_paths": [
        //list of reference images, relative to repo directory
        "ref image path"
      ],

      //feel free to tweak this number
      "num_inference_steps": 50,
      "num_frames": 45,
      "frame_rate": 15,

      //feel free to tweak this number
      "height": 480,
      "width": 720,

      //output name, relative to repo root
      "custom_output_name": "output name"
    }, 
    //multi-reference data case
    {
      "caption": "sample caption",
      "sketch_control_path": "sketch input mp4", //relative to repo directory
      "reference_paths": [
        //list of reference images, relative to repo directory
        "ref image path",
        "should be more than one"
      ],
      //identity mask, associate which reference to target
      "identity_mask_npz_path": "./examples/multiref/case1/mask.npz",


      //feel free to tweak this number
      "num_inference_steps": 50,
      "num_frames": 45,
      "frame_rate": 15,

      //feel free to tweak this number
      "height": 480,
      "width": 720,

      //output name, relative to repo root
      "custom_output_name": "output name"
    },

  ]
}
```
Identity masks are stored as NumPy .npz files containing a **3D array** of shape **(T, H, W)** (frame count× height pixels × width pixels). The file must include an array under the key **mask**. Each pixel value is a 0-indexed reference ID in {0, …, R−1}, indicating which reference the pixel is associated with.

## 🤝 Acknowledgements

We would like to express our gratitude to the following open-source projects that have been instrumental during our development process:

- [CogVideo](https://github.com/THUDM/CogVideo): An open source video generation framework by THUKEG, which we use as our DiT base model. 
- [finetrainers](https://github.com/a-r-r-o-w/finetrainers): A Memory-optimized training library for diffusion models. Our whole training and inference architecture use finetrainers repository as its base. 
- [SAKUGA-42M](https://github.com/KytraScript/SakugaDataset): A large-scale animation dataset. We use SAKUGA dataset for our training and evaluation processes. 
- [SAM2](https://github.com/facebookresearch/sam2): A mask propagator model. We use SAM2 for our automated mutlireference dataset generation pipeline to track main subjects throughout the video. 
- [GroundingDINO](https://github.com/IDEA-Research/GroundingDINO): A Text-Grounded Object Detection model. We use GroundingDINO for our automated mutlireference dataset generation pipeline to detect all main subjects present in the scene grounded on text input. 
- [InternVL3](https://github.com/OpenGVLab/InternVL/): A multimodal large-language model (MLLM). We use InternVL3 for our automated mutlireference dataset generation pipeline to detect all main subjects. 
- [xDiT](https://github.com/xdit-project/xDiT): A Scalable Inference Engine for Diffusion Transformers (DiTs) with parallelism. We use xDiT as a variant of our inference script that supports CFG parallelism. 

Special thanks to the contributors of these work for their hard work and dedication!

## Citation
If you find this work useful, please consider giving a star and citing it!
```bibtex
@misc{sadihin2026timecolorflexiblereferencecolorization,
      title={TimeColor: Flexible Reference Colorization via Temporal Concatenation}, 
      author={Bryan Constantine Sadihin and Yihao Meng and Michael Hua Wang and Matteo Jiahao Chen and Hang Su},
      year={2026},
      eprint={2601.00296},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2601.00296}, 
}
```
