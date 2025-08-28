# Camera Calibration (OpenCV + Gradio)

A complete camera calibration system using OpenCV with an interactive Gradio interface. This system computes camera intrinsic parameters (focal length, principal point, distortion coefficients) from chessboard calibration images.

## 🚀 Features

- **Interactive Gradio UI** for easy calibration workflow
- **Modular Design** with organized helper classes
- **3D Visualization** of camera poses and chessboard
- **Undistortion Preview** showing before/after comparison
- **Sample Image Overlays** with coordinate axes
- **JSON Export** of calibration results
- **Google Colab Compatible** with automatic setup

## 📋 Setup

### Google Colab (Recommended)
1. Open the notebook in Google Colab
2. Run the first cell to install packages automatically
3. All dependencies will be installed via `%pip install`

### Local Installation
```bash
# Clone or download the repository
git clone <your-repository-url>
cd camera-calibration

# Install dependencies
pip install opencv-python numpy matplotlib plotly gradio pillow
pip install pytransform3d  # Optional for advanced 3D visualization

# Launch Jupyter notebook
jupyter notebook camera_calibration.ipynb
```

## 🎯 How to Use

### Step 1: Prepare Calibration Pattern
1. Download and print the OpenCV chessboard pattern: [pattern.png](https://docs.opencv.org/4.x/pattern.png)
2. Mount it on a flat, rigid surface (cardboard, clipboard, etc.)
3. Ensure the pattern is flat without wrinkles or bends

### Step 2: Capture Images
1. Take 15-20 images of the chessboard pattern
2. Vary the angles, distances, and positions
3. Include tilted views and corner positions
4. Ensure the entire pattern is visible in each image
5. Save images as `.jpg` or `.jpeg` files

**Tips for good calibration:**
- Cover the entire image area with different pattern positions
- Include images with the pattern at various depths
- Avoid motion blur - keep the camera steady
- Use good lighting conditions

### Step 3: Run Calibration
1. **In the notebook UI:**
   - Upload your `.jpeg` images using the file uploader
   - Images are automatically saved to `/content/images/` (Colab) or `./images/` (local)
   
2. **Set pattern parameters:**
   - **Pattern Width**: Number of inner corners horizontally (default: 9)
   - **Pattern Height**: Number of inner corners vertically (default: 6)
   - **Square Size**: Physical size of each square in meters (default: 0.025m = 25mm)
   
3. **Process Images**: Click "📋 Process Images" to detect chessboard corners
   - Status will show how many images had successful corner detection
   - Need at least 10 successful detections for calibration
   
4. **Run Calibration**: Click "🎯 Run Calibration" to compute camera parameters

### Step 4: View Results
The system provides comprehensive results:

1. **Calibration Parameters:**
   - Camera matrix (K) with focal lengths (fx, fy) and principal point (cx, cy)
   - Distortion coefficients
   - Reprojection error in pixels
   
2. **3D Visualization:**
   - Interactive plot showing camera poses relative to the chessboard
   - Red points represent the chessboard corners
   - Blue diamonds represent camera positions
   
3. **Sample Images:**
   - Up to 3 sample images with coordinate axes overlaid
   - X-axis (red), Y-axis (green), Z-axis (blue)
   
4. **Undistortion Preview:**
   - Side-by-side comparison of original vs. undistorted image
   - Shows the effect of lens distortion correction

## 📁 Output Files

The calibration results are saved to `calibration.json` with the following structure:

```json
{
    \"camera_matrix\": [[fx, 0, cx], [0, fy, cy], [0, 0, 1]],
    \"distortion_coefficients\": [k1, k2, p1, p2, k3],
    \"rotation_vectors\": [...],
    \"translation_vectors\": [...],
    \"image_width\": 1920,
    \"image_height\": 1080
}
```

## 🏗️ Code Structure

The system is organized into modular helper classes:

### Core Classes
- **`Calib`**: Main calibration orchestrator and workflow management
- **`Board`**: Chessboard pattern model and corner detection
- **`Cam`**: Camera mathematics, calibration algorithms, and error computation
- **`Img`**: Image processing utilities (loading, resizing, undistortion)
- **`Overlay`**: Visualization overlays (axes, corners, annotations)
- **`Render`**: 3D plotting and interactive visualization
- **`IO`**: File operations and calibration data persistence

### Design Principles
- **Stateless Methods**: All methods use `@staticmethod` for portability
- **Separation of Concerns**: UI logic separated from calibration algorithms
- **Modular Architecture**: Easy to extend and modify individual components
- **Type Hints**: Full type annotations for better code clarity

## 🔧 Parameters

### Pattern Parameters
- **Pattern Size**: Internal corners (not squares)
  - Standard OpenCV pattern: 9×6 inner corners
  - Total squares: 10×7
- **Square Size**: Physical dimension in meters
  - Measure carefully for accurate results
  - Common sizes: 20mm, 25mm, 30mm

### Calibration Requirements
- **Minimum Images**: 10 successful detections required
- **Recommended**: 15-20 images for better accuracy
- **Coverage**: Pattern should appear across entire image area
- **Variety**: Different angles, distances, and orientations

## 📊 Understanding Results

### Camera Matrix (K)
```
K = [fx  0  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```
- **fx, fy**: Focal lengths in pixels
- **cx, cy**: Principal point (optical center)

### Distortion Coefficients
- **k1, k2, k3**: Radial distortion coefficients
- **p1, p2**: Tangential distortion coefficients

### Reprojection Error
- Average pixel error between detected and projected corners
- **Good**: < 1.0 pixel
- **Acceptable**: < 2.0 pixels
- **Poor**: > 3.0 pixels

## 🐛 Troubleshooting

### Common Issues

1. **\"Need at least 10 successful detections\"**
   - Ensure chessboard pattern is clearly visible
   - Check lighting conditions
   - Verify pattern parameters match your printed pattern

2. **High reprojection error**
   - Add more images with better coverage
   - Ensure pattern is perfectly flat
   - Check for motion blur in images

3. **Gradio interface not loading**
   - Restart the notebook kernel
   - Check internet connection (for Colab)
   - Try running locally if Colab issues persist

4. **Import errors**
   - Ensure all packages are installed: `%pip install opencv-python numpy matplotlib plotly gradio pillow`
   - Restart runtime after package installation (in Colab)

### Tips for Better Results

1. **Pattern Quality**
   - Print on high-quality paper
   - Mount on rigid surface
   - Avoid shadows and reflections

2. **Image Capture**
   - Use manual focus if possible
   - Ensure sharp, well-lit images
   - Cover all areas of the image sensor

3. **Pattern Positioning**
   - Include edge and corner positions
   - Vary distance and angle
   - Fill 20-60% of image with pattern

## 📚 References

- [OpenCV Camera Calibration Tutorial](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html)
- [OpenCV Calibration Pattern](https://docs.opencv.org/4.x/pattern.png)
- [Pinhole Camera Model Notes](https://github.com/ribeiro-computer-vision/pinhole_camera_model/tree/main)

## 📄 License

This project is provided for educational purposes. Please refer to individual package licenses for OpenCV, Gradio, and other dependencies.

## 🤝 Contributing

Feel free to submit issues, suggestions, or improvements to enhance the calibration system.

---

**Note**: This system is designed for educational and research purposes. For production applications, consider additional calibration validation and error analysis.