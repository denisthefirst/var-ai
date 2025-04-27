# Video Collection

The video collection script `collect.py` downloads all specified video
clips inside the `video-collection.yaml` file in the projects root
directory.

## Why use this script?

The script has no real value on the other than speeding up the data
gathering process. It was born out off convinience, because downloading
a whole soccer game from youtube manually (2 x 45 min) takes ages. Then
loading the games into a video editing programm and cutting out the
necessary sections seemt to be to tedious.

The collection scritp only downloads the neccessary sections from within
a video and thus saving on resources and time (script takes ~3 min for
22 videos depending on the internet speed and video length). Specifing
which clips to download also helps in keeping the dataset homogenuis
between the developers.

The script only downloads a video once, meaning if something is added to
the collection already downloaded videos are not downloaded again. To
download a video again, just remove it or change its filenname.

## Prerequisites

The script uses `yt-dlp` to download the clips from youtube and `PyYAML`
for working with `.yaml` files.

Install the dependencies:

``` sh
pip install -r requirements.txt
```

If neccessary create a `data/raw` directory in the projects root:

``` sh
mkdir -p data/raw
```

## How to

As already mentioned, the video collection is specified inside the
`video-collection.yaml` file.

Here is an exmaple video-collection with subsections:

``` yaml
- title: "uruguay-vs-ghana-fifa-wc-2010"
  sections:
  - title: "top" 
    url: "https://www.youtube.com/watch?v=f5M9mzcjZ_8&t=7890s" 
    duration: 15
  - title: "side" 
    url: "https://www.youtube.com/watch?v=f5M9mzcjZ_8&t=7905s" 
    duration: 6
  - title: "front" 
    url: "https://www.youtube.com/watch?v=f5M9mzcjZ_8&t=7911s" 
    duration: 6
```

This collection tells the script to download 3 video clips. As a result,
the follwing files are stored to the `data/raw` directory:

``` txt
data/raw
├── uruguay-vs-ghana-fifa-wc-2010-front.mp4
├── uruguay-vs-ghana-fifa-wc-2010-side.mp4
└── uruguay-vs-ghana-fifa-wc-2010-top.mp4
```

All clips from share the same "main" title and then an individual
section title. If both title and section title are the same on different
instances a counter is prefixed to the filename, preventing the script
from overwriting files with the same name.

After adding videos to the collection, run the script:

``` sh
python src/collecting/collect.py
```
