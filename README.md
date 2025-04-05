# License Plate Recognition (LPR) System for Edge Devices

## Overview

This project implements a real-time License Plate Recognition (LPR) system specifically optimized for resource-constrained edge devices like the NVIDIA Jetson Nano. It utilizes street-side cameras as video input and employs a pipeline of YOLO for vehicle and license plate detection, and CRNN for Optical Character Recognition (OCR) of the license plate text.

To overcome the performance limitations of running standard PyTorch YOLO models on the Jetson Nano, this project leverages **TensorRTX** and **PyCUDA** for significant inference speedups. Additionally, a **Kalman filter-based tracking** system is integrated to track individual vehicles across frames, associating each vehicle with its detected license plate.

A user-friendly **PyQt5 interface** is included, providing live video streaming and a search functionality for recorded vehicles and their license plates. Captured car and motorcycle images are saved in the `Record` directory.

### Real-time Recognition and Tracking Demo

You can watch the demonstration video showcasing the real-time model detection within the PyQt interface and the historical record search functionality:

![LPR Demo](https://github.com/Peter-1119/RealtimeEdgeLPR/blob/main/Assets/python%202025-04-05%2023-08-50.mp4)

## Key Features

* **Optimized for Edge Devices:** Designed and optimized for low-power, resource-constrained devices like the Jetson Nano.
* **High-Performance Inference:** Utilizes TensorRTX and PyCUDA to accelerate YOLO model inference, achieving smooth real-time performance on the Jetson Nano.
* **Accurate License Plate Detection:** Employs YOLO for robust detection of vehicles and license plates in various conditions.
* **Reliable Character Recognition:** Integrates CRNN for accurate Optical Character Recognition (OCR) of license plate text.
* **Vehicle Tracking:** Implements a Kalman filter-based tracking system to maintain the identity of vehicles across video frames.
* **License Plate Association:** Links detected license plates to their corresponding tracked vehicles.
* **User-Friendly Interface:** Features a PyQt5 graphical user interface for live video streaming and searching recorded data.
* **Image Recording:** Automatically saves captured images of cars and motorcycles with detected license plates in the `Record` directory.

## Directory Structure

```
LPR_interface/
├── Assets/              # Demo GIFs, images, and video thumbnails
├── models/
│   ├── *.py             # Model definition scripts (e.g., for CRNN)
│   └── weights/
│       ├── yolo/        # TensorRTX engine weights for YOLO
│       └── crnn/        # PyTorch weights for CRNN
├── ObjectTracker/       # Tracking algorithm script (likely Kalman filter)
├── Record/              # Directory to store captured car and motorcycle images
├── videos/              # Test video files for YOLO and CRNN
├── WorkWidget/          # PyQt5 interface script
├── main.py              # Main application entry point
└── README.md
```

## Getting Started

### Prerequisites

* **NVIDIA Jetson Nano:** This project is primarily designed and tested for the Jetson Nano.
* **JetPack SDK:** Ensure the NVIDIA JetPack SDK is installed on your Jetson Nano, including CUDA, cuDNN, and TensorRT.
* **TensorRTX:** You will need to build and potentially adapt TensorRTX for your specific YOLO model. Refer to the TensorRTX documentation for instructions.
* **PyCUDA:** Install PyCUDA for interacting with the GPU.
* **PyTorch:** Install PyTorch and torchvision.
* **PyQt5:** Install PyQt5 for the graphical user interface.
* **Other Dependencies:** Install any other required Python libraries (e.g., OpenCV).

### Installation

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/Peter-1119/RealtimeEdgeLPR.git
    cd LPR_interface
    ```

2.  **Download Weights:**
    * Place your pre-trained PyTorch weights for the CRNN model in the `models/weights/crnn/` directory.
    * Build the TensorRT engine for your YOLO model using TensorRTX and place the resulting engine file(s) in the `models/weights/yolo/` directory. Follow the TensorRTX documentation for the specific steps involved in building the engine.

3.  **Install Python Dependencies:**
    ```bash
    pip install -r requirements.txt  # If a requirements.txt file is available
    # Otherwise, install the necessary libraries individually:
    pip install torch numpy
    pip install pycuda
    pip install pyqt5 opencv-python
    ```

### Usage

1.  **Run the Main Application:**
    ```bash
    python main.py
    ```
    This will launch the PyQt5 interface.

2.  **Interface Overview:**
    * **Live Stream:** The interface should display a live video stream from your connected camera.
    * **Object Detection:** Detected vehicles and license plates will be highlighted in the video feed.
    * **License Plate Recognition:** The recognized license plate text will be displayed alongside the detected license plate.
    * **Tracking:** Vehicles will be assigned unique IDs and tracked across frames.
    * **Search Functionality:** You should be able to search for recorded vehicles based on their license plate number or other relevant criteria.
    * **Recordings:** Captured images of cars and motorcycles with their detected license plates will be saved in the `Record` directory, organized as needed.

3.  **Testing with Video Files (Optional):**
    You can place test video files in the `videos/` directory and potentially configure the `main.py` or interface to process these videos instead of a live camera feed for testing purposes.

## Project Structure Details

* **`Assets/`:** Contains static assets like demo GIFs showcasing the project's functionality, example images, and video thumbnails for potential documentation or presentation.
* **`models/`:** Houses the Python scripts defining the model architectures (e.g., for the CRNN) and the `weights/` subdirectory containing the trained model parameters.
    * **`weights/yolo/`:** This directory is crucial and should contain the optimized TensorRT engine file(s) for your YOLO model.
    * **`weights/crnn/`:** This directory should contain the pre-trained PyTorch weight file(s) for your CRNN model.
* **`ObjectTracker/`:** Contains the Python script implementing the vehicle tracking algorithm, likely utilizing a Kalman filter for state estimation and prediction.
* **`Record/`:** This directory will automatically store captured images of detected cars and motorcycles, potentially organized by date or license plate number for easier management.
* **`videos/`:** This directory is intended for storing test video files that can be used to evaluate the performance of the YOLO and CRNN models without relying on a live camera feed.
* **`WorkWidget/`:** This directory (or likely a file named `WorkWidget.py`) contains the Python script defining the PyQt5 widgets and logic for the user interface, including the live stream display and search functionality.
* **`main.py`:** This is the main entry point of the application. It likely initializes the camera, loads the models (YOLO TensorRT engine and CRNN PyTorch model), sets up the tracking system, integrates with the PyQt5 interface, and orchestrates the entire LPR pipeline.
* **`README.md`:** This file provides an overview of the project, its features, setup instructions, and directory structure.

## Optimization and Performance

The core of this project's optimization lies in the use of **TensorRTX** and **PyCUDA** for accelerating the YOLO model inference on the Jetson Nano. TensorRT optimizes the neural network graph for the target hardware, leading to significant reductions in latency and increases in throughput. PyCUDA provides a Python interface for interacting directly with the NVIDIA GPU, enabling efficient memory management and kernel execution.

The integration of a **Kalman filter** not only enables vehicle tracking but can also improve the robustness of license plate recognition by associating multiple detections of the same vehicle over time.

## Potential Future Enhancements

* **Improved Tracking Algorithms:** Explore more advanced tracking algorithms for handling occlusions and complex traffic scenarios.
* **License Plate Normalization:** Implement techniques to normalize license plate images (e.g., perspective correction) before OCR to improve accuracy.
* **Cloud Integration:** Consider integrating with cloud services for data storage, analytics, or remote monitoring.
* **Expand Vehicle Type Recognition:** Extend the object detection capabilities to classify different types of vehicles (e.g., cars, trucks, motorcycles) more accurately.
* **Refined User Interface:** Add more advanced search filters, data visualization, and configuration options to the PyQt5 interface.

## Contributing

[Optional: Add information about how others can contribute to your project.]

## License

[Optional: Add license information for your project.]

---

This `README.md` provides a comprehensive overview of your LPR project. Remember to replace `<your_repository_url>` with the actual URL of your repository and fill in any optional sections as needed. Good luck with your project!
