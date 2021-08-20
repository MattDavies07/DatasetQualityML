# DatasetQualityML
Code as part of Newcastle University MSc dissertation, looking into analysing the importance of data quality by its impact on model performance. Analysis carried out by looking at the DRIVE dataset and a U-Net model for segmentation.

The code for this project was adpated from the DRIVE U-Net implementation by: https://github.com/nikhilroxtomar/Retina-Blood-Vessel-Segmentation-in-PyTorch/tree/main/UNET

## U-Net
The U-Net model and code can be found in the `model` folder. This has been updated and adapted to fit the purposes of the project.

## Experiment Structure
All experiments were run using Google Colab and the GPU it provides.
- Firstly run `data_aug.py` to set up the images for the training and test set. Within `data_aug.py` setting can be changed to apply the gradient, intensity and image quality changes to the dataset. All data to be used for the run can be found in `exp_data/`.
- Then run `train.py` found within the `model` folder. This will produce a tensorbaord run file which can be found in `model/runs/{RUN}`
- Finally running `test.py` from the `model` folder, will test the trained model against the test set. Results can be found in `model/results/{RESULT_START_DATE_TIME}`. This will include the performance metrics for each image as well as for the whole set in `results.csv`, as well as the predicted images as well as an image showing the input image, ground truth and predicted image side by side.

A standard run, using the orgininal data and ground truth masks/annotations can be found in the `exp_data`, `runs` and `results` folders mentioned.
