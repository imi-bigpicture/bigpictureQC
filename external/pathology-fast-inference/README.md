# pathology-fast-inference
High-speed multiprocessing inference for Whole Slide Images
# gan inference call example
```
python3.6 -u <path-to-package>/pathology-fast-inference/scripts/applygan_multiproc.py 
--model_path "<path-to-cyclegan-training-output>/checkpoint" 
--param_file_path "<path-to-cyclegan-parameter-file>.yaml" 
--input_wsi_path "<path-to-input-wsis>/*.tif" 
--mask_wsi_path "<path-to-mask-files>/{image}.tif" 
--output_wsi_path "<path-to-output>/{image}.tif" 
--normalizer "default" 
--tile_size 1024 
--read_spacing <input-spacing-wsi> 
--write_spacing <output-spacing-wsi> 
--gpu_count 1 
--batch_size 3 
--readers 5 
--writers 5 
--work_directory "/home/user/work" 
--a2b 
--cache_directory "/home/user/cache"

python3.6 -u <path-to-package>/pathology-fast-inference/scripts/applygan_multiproc.py 
--model_path "<path-to-model>/backgroundsegmentation.net" 
--input_wsi_path "<path-to-input-wsis>/*.tif"
--output_wsi_path "<path-to-output>/{image}.tif" 
--normalizer "default" 
--tile_size 512
--read_spacing 0.5
--write_spacing 0.5
--gpu_count 1
--batch_size 8 
--readers 5
--writers 5 
--work_directory "/home/user/work" 
--cache_directory "/home/user/cache"
```
