# gi-fishing
GI Fishing Frames

## Requirements
- Numpy
- Matplotlib
- pandas
- tqdm
- OpenCV-Python

## Usage
- Create Conda environment:
```
conda env create -f environment.yml
conda activate frames
```

- Export your video to frames using:
```
video_to_frames.sh <video_filename>
```

- Place your frame data in `data/pufferfish-struggle`

- Run `main.py`
