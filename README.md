# Iron Man AR Builder

A webcam-based AR-style block builder made with Python, OpenCV, and MediaPipe.

## Features

- Hand tracking through your webcam
- Pinch to create blocks
- Pinch and drag to move blocks
- Optional grid snapping
- Adjustable block size
- Delete individual blocks
- Clear all blocks
- Automatic MediaPipe hand-landmarker model download

## Controls

- Pinch: create or grab a block
- `G`: toggle grid
- `+` / `-`: change block size
- `D`: delete hovered block
- `C`: clear all blocks
- `Q`: quit

## Setup

1. Install Python.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the project:

```bash
python ar_builder.py
```

The hand tracking model will download automatically the first time the program runs.

## Requirements

- Python
- Webcam
- OpenCV
- MediaPipe
