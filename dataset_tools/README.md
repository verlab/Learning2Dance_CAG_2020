<link rel="stylesheet" type="../docs/assets/style.css" media="all" href="URL" />

<img align="left" width="auto" height="75" src="../docs/assets/ufmg.png">
<img align="right" width="auto" height="75" src="../docs/assets/verlab.png">
<br/>
<br/>
<br/>
<br/>
<hr>

<h1 align="center"> <b>Learning to Dance - Dataset Tools</b></h1>


### This folder contains all the functions used in the production of Learning2Dance dataset. 

- If you want to create your own dataset with similar characteristics or add styles to this dataset, follow through the 10 steps bellow. 
> NOTE: It's recommended to iterate through steps 3-9 twice to produce better results.

- **To generate metadata for running l2d model on original dataset, jump to [step 8](##8.-Generate-metadata-file-to-PyTorch-dataloader).** 

## 1. Download videos from YouTube

>NOTE: You'll need [youtube-dl](https://youtube-dl.org/) to download youtube videos. Make sure that you can run it from the terminal. The API is updated from time to time, so if you have problems running it, try ```pip install --upgrade youtube_dl```.

You will need to create a folder `$TXTS_PATH` that contains one .txt file for each style you want to download, and put on each line of the .txt files the YouTube video url.
```
${TXTS_PATH}
|-- style1.txt
|-- style2.txt
|-- ...
```

To download the videos and create dataset structure run the command bellow, you need to specify the output path `$DATASET_ROOT`.

```bash
python3 download_dataset.py --input_txts $TXTS_PATH/style1.txt $TXTS_PATH/style2.txt ... --output_path $DATASET_ROOT
```

The resulting dataset structure will be similar to this:

```
${DATASET_ROOT}
|-- style1
    |-- I_0
        |-- I_0.mp4
    |-- I_1
    |-- ...
|-- style2
    |-- I_0
    |-- I_1
    |-- ...
|-- ...
```

## 2. Standardize video FPS and audio sample rate(SR)

To standardize all videos to standard `$FPS` and `$SR`, run:

```bash
python3 standardize_fps_sr.py --dataset_path $DATASET_ROOT --sample_rate $SR --fps $FPS
```

## 3. Run OpenPose pose extractor

The default pose extractor used in the paper and in the dataset is OpenPose with the 25 joints configuration. Instructions to compile and run the demo code can be found at the [OpenPose Official Repository](https://github.com/CMU-Perceptual-Computing-Lab/openpose). 

You can also try this [Docker Image with OpenPose and Python API support](https://hub.docker.com/r/cwaffles/openpose).

>NOTE: We provide the script that we used in our process of creation, but you can use other configurations of OpenPose or the Python API. The only requisite is that the dataset tree is maintained.

To use OpenPose demo and extract poses from all videos, run:

```bash
./run_openpose.sh $DATASET_ROOT
```

The resulting dataset tree will be:

```
${DATASET_ROOT}
|-- style1
    |-- I_0
        |-- I_0.mp4
        |-- openpose
            |-- json
            |-- rendered_poses
    |-- I_1
    |-- ...
|-- style2
    |-- I_0
    |-- I_1
    |-- ...
|-- ...
```

## 4. Filter the results from OpenPose

We developed our own person tracker and filter for low confidence poses. This step is needed to ensure data quality and code compatibility in the next steps.

Run:

```bash
python3 filter_openpose.py --dataset_path $DATASET_ROOT
```

## 5. Manually select videos with good poses

Probably the videos will have miss detections from OpenPose, we need to filter these out. 

First run `render_debug.py` to visualize OpenPose skeletons and the corresponding .json file:

```bash
python3 render_debug.py --dataset_path $DATASET_ROOT
```

Then put an .txt file inside every video folder with the same name of the video. 

Example:

```
${DATASET_ROOT}
|-- style1
    |-- I_0
        |-- I_0.mp4
        |-- I_0.txt
        |-- openpose
            |-- json
            |-- rendered_poses
    |-- I_1
    |-- ...
|-- style2
    |-- I_0
    |-- I_1
    |-- ...
|-- ...
```

Specify on every line of each file an interval separeted by a comma. Example:

```
${I_0.txt}
100, 234
249, 500
```

```bash
python3 cut_videos.py --dataset_path $DATASET_ROOT
```

## 6. Extract audio from videos

To extract the audio files from videos, run:

```bash
python3 extract_audios.py --dataset_path $DATASET_ROOT
```

## 7. Preprocess data

```bash
python3 preprocess.py --dataset_path $DATASET_ROOT
```

## 7.1 (Optional) Data augmentation

> NOTE: The pose/audio data augmentation parameters are hard-coded, if you desire to change the procedure you will need to change `data_augmentation.py` code.

To augment only pose data, run:
```bash
python3 data_augmentation.py --dataset_path $DATASET_ROOT --pose
```
To augment pose and audio data use the `audio` flag:
```bash
python3 data_augmentation.py --dataset_path $DATASET_ROOT --pose --audio
```

## 8. Generate metadata file to PyTorch dataloader

The metadata file maps all the samples informations to the PyTorch dataloader. You can change the size of the samples, the size of stride between them and use or not augmented data by varying the metadata file. 

To run without data_augmentation:

```bash
python3 create_metadata.py --dataset_path $DATASET_ROOT --sample_size $SAMPLE_SIZE --stride $STRIDE
```

To run with data_augmentation(needs step 7.1, available at original l2d dataset):

```bash
python3 create_metadata.py --dataset_path $DATASET_ROOT --sample_size $SAMPLE_SIZE --stride $STRIDE --data_aug
```

## 9. Visualize results

To render 10 random videos with audio:
```bash
python3 render_data.py --test_dataset --dataset_path $DATASET_ROOT --metadata_path $METADATA_PATH 
```
To rendered specific video:
```bash
python3 render_data.py --render_video $VIDEO_PATH
```

To render all videos from dataset, run:
```bash
./render_dataset.sh $DATASET_ROOT
```