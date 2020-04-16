# MMDM: Multi-frame and Multi-scale for Image Demoiréing
This repository is for MMDM introduced in the following paper

Shuai Liu, Chenghua Li, Nan Nan, Ziyao Zong, Ruixia Song, "MMDM: Multi-frame and Multi-scale for Image Demoiréing", CVPRW 2020.

## Real Image Denoising
### Train
TODO

### Test
    ```bash
    # For rawRGB
    # TODO
    
    # For sRGB
    python main.py --data_test MyImage --scale 2 --model RCAN --n_resgroups 10 --n_resblocks 20 --n_feats 64 --pre_train ../model/RCAN_BIX2.pt --test_only --save_results --chop --save 'RCAN' --testpath ../LR/LRBI --testset Set5
    ```

## Demoireing
### Train
TODO

### Test
    ```bash
    # For single frame input.
    # TODO
    
    # For multi-frame inputs (Burst)
    TODO
    ```
    
## Citation
If you find the code helpful in your resarch or work, please cite the following papers.
```
@inproceedings{Liu2020MMDM,
	title={{MMDM}: Multi-frame and Multi-scale for Image Demoiréing},
	author={Liu, Shuai and Li, Chenghua and Nan, Nan and Zong, Ziyao and Song, Ruixia},
	booktitle={The IEEE Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
	year={2020}
}
```
