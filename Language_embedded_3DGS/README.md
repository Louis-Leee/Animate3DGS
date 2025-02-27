# Language-embedded 3DGS


# Installation
The installation is similar to [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting).

Then install the dependencies:
```bash
conda env create --file environment.yml
conda activate gaussian_splatting
```
In default, we use the public ViT-H model for SAM. You can download the pre-trained model from [here](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth) and put it under ./third_party/segment-anything/sam_ckpt.

Since we need the pre-trained 3D-GS model for mask scales extraction, the first step is to train the 3D Gaussians:

## Pre-train the 3D Gaussians
We inherit all attributes from 3DGS, more information about training the Gaussians can be found in their repo.
```bash
python train_scene.py -s <path to COLMAP or NeRF Synthetic dataset>
```

## Prepare data
Then, to get the sam_masks and corresponding mask scales, run the following command:
```bash
python extract_segment_everything_masks.py --image_root <path to the scene data> --sam_checkpoint_path <path to the pre-trained SAM model> --downsample <1/2/4/8>
python get_scale.py --image_root <path to the scene data> --model_path <path to the pre-trained 3DGS model>
```
Note that sometimes the downsample is essential due to the limited GPU memory.

If you want to try the open-vocabulary segmentation, extract the CLIP features first:
```bash
python get_clip_features.py --image_root <path to the scene data>
```

## Train 3D Gaussian Segmentation Features
```bash
python train_contrastive_feature.py -m <path to the pre-trained 3DGS model> --iterations 10000 --num_sampled_rays 1000
```

## 3D Segmentation
Currently our 3DGS provides an interactive GUI (saga_gui.py) implemented with dearpygui and a jupyter-notebook (prompt_segmenting.ipynb). To run the GUI:
```bash
python saga_gui.py --model_path <path to the pre-trained 3DGS model>
```
Temporarily, open-vocabulary segmentation is only implemented in the jupyter notebook. Please refer to prompt_segmenting.ipynb for detailed instructions.


## Rendering
After saving segmentation results in the interactive GUI or running the scripts in prompt_segmenting.ipynb, the bitmap of the Gaussians will be saved in ``./segmentation_res/your_name.pt`` (you can set the name by yourself). To render the segmentation results on training views (get the segmented object by removing the background), run the following command:
```bash
python render.py -m <path to the pre-trained 3DGS model> --precomputed_mask <path to the segmentation results> --target scene --segment
```

To get the 2D rendered masks, run the following command:
```bash
python render.py -m <path to the pre-trained 3DGS model> --precomputed_mask <path to the segmentation results> --target seg
```

You can also render the pre-trained 3DGS model without segmentation:
```bash
python render.py -m <path to the pre-trained 3DGS model> --target scene
```


## Acknowledgement
The implementation of saga refers to [GARField](https://github.com/chungmin99/garfield.git), [OmniSeg3D](https://github.com/OceanYing/OmniSeg3D-GS), [Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [SAGA](https://github.com/Jumpat/SegAnyGAussians), and we sincerely thank them for their contributions to the community.
