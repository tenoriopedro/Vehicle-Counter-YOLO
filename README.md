# Traffic Flow Counter with YOLOv8 🚗📈

## 📌 Features

- 📦 Vehicle detection and tracking using **YOLOv8**
- ➕ Counting cars, motorcycles, and trucks
- ↔️ Separate counting by direction (inbound and outbound)
- 📹 Generates video with overlaid results
- ✅ Test video included for immediate use

---

## 🎥 Test Video Included

The project includes a test video named `track_video_car01.mp4` inside the `test_files/` folder.  
This allows you to test the system immediately without searching for external videos.

---

## 🚀 How to Download and Run

### 1. Clone the repository:

```bash
git clone https://github.com/your-username/traffic-flow-counter-yolov8.git
cd traffic-flow-counter-yolov8
```

### 2. Install dependencies (recommended to create and activate a virtual environment):

```bash

python -m venv venv

.\venv\Scripts\activate.ps1

```

### 3. Run the main video processing script:

```bash

python compile_video.py

```

### 4. Run the script to generate and display the final video:

```bash

python show_results.py

```

- The final output video will be saved in the `result_files/` folder as `video_countingCar_result01.mp4`