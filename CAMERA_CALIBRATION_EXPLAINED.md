# Camera Calibration: Mathematical Theory and Code Implementation

## Table of Contents
1. [Introduction](#introduction)
2. [Mathematical Theory](#mathematical-theory)
3. [Calibration Process](#calibration-process)
4. [Code Implementation](#code-implementation)
5. [Error Analysis](#error-analysis)
6. [Practical Applications](#practical-applications)

---

## Introduction

Camera calibration is the process of determining the intrinsic and extrinsic parameters of a camera. This allows us to:
- Correct lens distortion
- Convert between pixel coordinates and real-world coordinates
- Enable accurate 3D measurements from 2D images
- Perform stereo vision and augmented reality applications

---

## Mathematical Theory

### 1. Pinhole Camera Model

The pinhole camera model is the foundation of camera calibration. It describes how 3D world points are projected onto a 2D image plane.

#### Basic Projection Equation:
```
s * [u]   [fx  0  cx] [X]
    [v] = [ 0 fy  cy] [Y]
    [1]   [ 0  0   1] [Z]
```

Where:
- `(X, Y, Z)` = 3D world coordinates
- `(u, v)` = 2D image pixel coordinates  
- `s` = scaling factor
- `fx, fy` = focal lengths in pixel units
- `cx, cy` = principal point (optical center)

#### Homogeneous Coordinates:
We use homogeneous coordinates to represent points:
- 2D point: `[u, v, 1]ᵀ`
- 3D point: `[X, Y, Z, 1]ᵀ`

This allows us to represent projections as matrix multiplications.

### 2. Camera Matrix (Intrinsic Parameters)

The camera matrix K contains intrinsic parameters specific to each camera:

```
K = [fx  0  cx]
    [ 0 fy  cy]
    [ 0  0   1]
```

#### Parameters Explained:
- **fx, fy**: Focal lengths in pixels
  - `fx = f * mx` (focal length × pixel density in x)
  - `fy = f * my` (focal length × pixel density in y)
  - If pixels are square: `fx = fy`

- **cx, cy**: Principal point coordinates
  - Usually close to image center: `(width/2, height/2)`
  - Represents where optical axis intersects image plane

### 3. Lens Distortion

Real cameras have lens distortion that deviates from the pinhole model.

#### Radial Distortion:
Caused by lens shape - straight lines appear curved.

**Mathematical Model:**
```
x_distorted = x_undistorted * (1 + k1*r² + k2*r⁴ + k3*r⁶)
y_distorted = y_undistorted * (1 + k1*r² + k2*r⁴ + k3*r⁶)
```

Where:
- `r² = x² + y²` (distance from center)
- `k1, k2, k3` = radial distortion coefficients

#### Tangential Distortion:
Caused by lens not being perfectly parallel to image plane.

**Mathematical Model:**
```
x_distorted = x_undistorted + [2*p1*x*y + p2*(r² + 2*x²)]
y_distorted = y_undistorted + [p1*(r² + 2*y²) + 2*p2*x*y]
```

Where:
- `p1, p2` = tangential distortion coefficients

#### Complete Distortion Model:
```
Distortion Coefficients = [k1, k2, p1, p2, k3]
```

### 4. Extrinsic Parameters

Extrinsic parameters describe camera position and orientation in world coordinates.

#### Rotation and Translation:
For each calibration image, we have:
- **Rotation vector (rvec)**: 3D vector representing rotation axis and angle
- **Translation vector (tvec)**: 3D vector representing camera position

#### Transformation Equations:
```
[X_camera]   [R11 R12 R13] [X_world]   [tx]
[Y_camera] = [R21 R22 R23] [Y_world] + [ty]
[Z_camera]   [R31 R32 R33] [Z_world]   [tz]
```

Where R is the 3×3 rotation matrix derived from the rotation vector.

### 5. Complete Projection Pipeline

The full transformation from 3D world coordinates to 2D image pixels:

```
1. World → Camera coordinates: Apply rotation R and translation t
2. Camera → Normalized coordinates: Divide by Z (perspective projection)
3. Apply distortion: Add radial and tangential distortion
4. Normalized → Pixel coordinates: Apply camera matrix K
```

**Mathematical Expression:**
```
[u]   [fx  0  cx] [x_distorted]
[v] = [ 0 fy  cy] [y_distorted]
[1]   [ 0  0   1] [     1     ]

where (x_distorted, y_distorted) comes from applying distortion to normalized coordinates
```

---

## Calibration Process

### 1. Calibration Pattern (Chessboard)

We use a chessboard because:
- Corners are easily detectable
- Known geometry (square spacing)
- Multiple constraint equations per image

#### Object Points:
For a 9×6 inner corner chessboard with 25mm squares:
```python
# 3D coordinates in pattern coordinate system
object_points = []
for i in range(6):  # height
    for j in range(9):  # width
        object_points.append([j*0.025, i*0.025, 0.0])  # Z=0 (planar)
```

#### Image Points:
Detected corner locations in pixel coordinates:
```python
# 2D coordinates in image
ret, corners = cv2.findChessboardCorners(image, (9, 6))
if ret:
    # Refine corner locations to sub-pixel accuracy
    corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
    image_points.append(corners)
```

### 2. Calibration Equations

For each detected corner, we have one constraint equation:

```
s_i * [u_i]   [fx  0  cx] [R | t] [X_i]
      [v_i] = [ 0 fy  cy]        [Y_i]
      [ 1 ]   [ 0  0   1]        [Z_i]
                                 [ 1 ]
```

With N images and M corners per image, we have 2×N×M equations for the unknown parameters.

### 3. Parameter Estimation

OpenCV uses **iterative optimization** (Levenberg-Marquardt algorithm) to minimize:

```
Σ Σ ||m_ij - projection(M_i, K, D, R_j, t_j)||²
i j
```

Where:
- `m_ij` = detected corner j in image i
- `M_i` = 3D coordinates of corner j
- `K` = camera matrix
- `D` = distortion coefficients [k1, k2, p1, p2, k3]
- `R_j, t_j` = rotation and translation for image j

---

## Code Implementation

### 1. Board Class - Pattern Definition

```python
class Board:
    def __init__(self, pattern_size: Tuple[int, int], square_size: float):
        """
        pattern_size: (width, height) of inner corners - NOT squares!
        square_size: physical size of each square in meters
        """
        self.pattern_size = pattern_size  # e.g., (9, 6)
        self.square_size = square_size    # e.g., 0.025 (25mm)
        self.objp = self._create_object_points()
    
    def _create_object_points(self) -> np.ndarray:
        """Generate 3D coordinates of chessboard corners"""
        # Create grid of 3D points
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        
        # Fill X,Y coordinates (Z=0 for planar pattern)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 
                               0:self.pattern_size[1]].T.reshape(-1, 2)
        
        # Scale by square size to get real-world coordinates
        objp *= self.square_size
        return objp
```

**Mathematical Explanation:**
- `np.mgrid` creates a coordinate grid
- For (9,6) pattern: creates 54 points from (0,0,0) to (8,5,0)
- Scaling by `square_size` converts to real-world units (meters)

### 2. Corner Detection

```python
@staticmethod
def find_corners(image: np.ndarray, pattern_size: Tuple[int, int]):
    """Find chessboard corners with sub-pixel accuracy"""
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Find corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
    
    if ret:
        # Refine to sub-pixel accuracy
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    
    return ret, corners
```

**Algorithm Explanation:**
1. **Initial Detection**: Uses image gradients and connectivity analysis
2. **Sub-pixel Refinement**: Fits parabolic surface to corner neighborhood
3. **Termination Criteria**: Stops when change < 0.001 or after 30 iterations

### 3. Camera Calibration

```python
@staticmethod
def calibrate_camera(object_points, image_points, image_size):
    """Perform camera calibration using OpenCV"""
    
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,  # List of 3D points for each image
        image_points,   # List of 2D points for each image  
        image_size,     # Image dimensions (width, height)
        None,           # Initial camera matrix (auto-initialize)
        None            # Initial distortion coeffs (auto-initialize)
    )
    
    return ret, camera_matrix, dist_coeffs, rvecs, tvecs
```

**What OpenCV Does Internally:**
1. **Initialize parameters**: Estimate focal length from image size
2. **Iterative optimization**: Minimize reprojection error using Levenberg-Marquardt
3. **Return results**: Optimized parameters and per-image poses

### 4. Reprojection Error Calculation

```python
@staticmethod
def compute_reprojection_error(object_points, image_points, 
                              camera_matrix, dist_coeffs, rvecs, tvecs):
    """Calculate average reprojection error"""
    
    total_error = 0
    total_points = 0
    
    for i in range(len(object_points)):
        # Project 3D points to 2D using calibrated parameters
        projected_points, _ = cv2.projectPoints(
            object_points[i], rvecs[i], tvecs[i], 
            camera_matrix, dist_coeffs
        )
        
        # Calculate L2 distance between detected and projected points
        error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2)
        
        # Normalize by number of points
        error = error / len(projected_points)
        
        total_error += error
        total_points += 1
    
    return total_error / total_points  # Average error in pixels
```

**Mathematical Explanation:**
```
Reprojection Error = (1/N) * Σ ||detected_point_i - projected_point_i||²

Where projected_point_i = K * distort(R*P_i + t)
- K: camera matrix
- distort(): applies lens distortion  
- R, t: rotation and translation
- P_i: 3D object point
```

### 5. Undistortion Process

```python
@staticmethod
def undistort_image(image, camera_matrix, dist_coeffs):
    """Remove lens distortion from image"""
    
    return cv2.undistort(image, camera_matrix, dist_coeffs, 
                        None, camera_matrix)
```

**Undistortion Mathematics:**
1. **For each output pixel (u, v)**:
   - Convert to normalized coordinates: `x = (u - cx)/fx, y = (v - cy)/fy`
   - Apply inverse distortion model to find original distorted coordinates
   - Map back to input image coordinates
   - Interpolate pixel value

### 6. 3D Visualization - Camera Poses

```python
@staticmethod
def plot_camera_poses_plotly(rvecs, tvecs, board_size, square_size):
    """Visualize camera positions relative to chessboard"""
    
    fig = go.Figure()
    
    # Plot chessboard corners as red points
    board_points = []
    for i in range(board_size[1] + 1):  # Include border
        for j in range(board_size[0] + 1):
            board_points.append([j * square_size, i * square_size, 0])
    
    board_points = np.array(board_points)
    fig.add_trace(go.Scatter3d(
        x=board_points[:, 0], y=board_points[:, 1], z=board_points[:, 2],
        mode='markers', marker=dict(size=3, color='red'),
        name='Chessboard', opacity=0.6
    ))
    
    # Plot camera positions as blue diamonds
    camera_positions = []
    for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
        # Convert rotation vector to matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Camera position in world coordinates
        # World point = -R^T * tvec
        camera_pos = -R.T @ tvec.flatten()
        camera_positions.append(camera_pos)
    
    camera_positions = np.array(camera_positions)
    fig.add_trace(go.Scatter3d(
        x=camera_positions[:, 0], y=camera_positions[:, 1], z=camera_positions[:, 2],
        mode='markers', marker=dict(size=8, color='blue', symbol='diamond'),
        name='Cameras'
    ))
    
    return fig
```

**Mathematical Explanation of Camera Position:**
- `rvec, tvec` represent transformation from world to camera coordinates
- Camera position in world coordinates: `C = -R^T * t`
- This is because: `P_camera = R * P_world + t`, so `P_world = R^T * (P_camera - t)`
- When `P_camera = [0,0,0]` (camera origin), `P_world = -R^T * t`

### 7. Coordinate Axes Overlay

```python
@staticmethod
def draw_axes(image, camera_matrix, dist_coeffs, rvec, tvec, length=0.1):
    """Draw 3D coordinate axes on image"""
    
    # Define 3D axis points (origin + unit vectors)
    axes_points = np.float32([
        [0, 0, 0],      # Origin
        [length, 0, 0], # X-axis (red)
        [0, length, 0], # Y-axis (green) 
        [0, 0, -length] # Z-axis (blue, negative for right-hand rule)
    ]).reshape(-1, 3)
    
    # Project 3D points to image plane
    imgpts, _ = cv2.projectPoints(axes_points, rvec, tvec, 
                                 camera_matrix, dist_coeffs)
    imgpts = np.int32(imgpts).reshape(-1, 2)
    
    # Draw lines from origin to each axis endpoint
    img_with_axes = image.copy()
    corner = tuple(imgpts[0].ravel().astype(int))  # Origin
    
    # X-axis: Red line
    img_with_axes = cv2.line(img_with_axes, corner, 
                           tuple(imgpts[1].ravel().astype(int)), (0,0,255), 5)
    # Y-axis: Green line  
    img_with_axes = cv2.line(img_with_axes, corner,
                           tuple(imgpts[2].ravel().astype(int)), (0,255,0), 5)
    # Z-axis: Blue line
    img_with_axes = cv2.line(img_with_axes, corner,
                           tuple(imgpts[3].ravel().astype(int)), (255,0,0), 5)
    
    return img_with_axes
```

**Projection Mathematics:**
```
3D Point → 2D Image Point

1. World coordinates: [X, Y, Z]ᵀ
2. Camera coordinates: R * [X, Y, Z]ᵀ + t  
3. Normalized coordinates: [x, y] = [X_cam/Z_cam, Y_cam/Z_cam]
4. Apply distortion: [x_dist, y_dist] = distortion_model(x, y)
5. Pixel coordinates: [u, v] = [fx*x_dist + cx, fy*y_dist + cy]
```

---

## Error Analysis

### 1. Reprojection Error

**Good Calibration Metrics:**
- **< 1.0 pixel**: Excellent calibration
- **1.0-2.0 pixels**: Good calibration  
- **> 3.0 pixels**: Poor calibration (check images/pattern)

### 2. Error Sources

**Systematic Errors:**
- Inaccurate pattern dimensions
- Pattern not perfectly planar
- Poor image quality (blur, lighting)

**Random Errors:**
- Corner detection noise
- Camera sensor noise
- Numerical precision limits

### 3. Improving Calibration

**Best Practices:**
1. **More images**: 15-20+ from different angles
2. **Full coverage**: Pattern in all image regions
3. **Varied distances**: Near and far positions
4. **Quality images**: Sharp, well-lit, no motion blur
5. **Accurate pattern**: Precisely measured square size

---

## Practical Applications

### 1. Image Rectification

Remove lens distortion for accurate measurements:
```python
# Undistort image
undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

# Now straight lines in real world appear straight in image
```

### 2. 3D Reconstruction

Convert image points to real-world coordinates:
```python
# Given: image point (u, v), camera parameters, and depth Z
# Calculate: real-world point (X, Y, Z)

# Step 1: Convert to normalized coordinates
x_norm = (u - cx) / fx
y_norm = (v - cy) / fy

# Step 2: Scale by depth
X = x_norm * Z  
Y = y_norm * Z
# Z is given or estimated from stereo/other sensors
```

### 3. Augmented Reality

Place virtual objects in real scenes:
```python
# Define virtual object corners in 3D
object_3d = np.array([[0,0,0], [0.1,0,0], [0.1,0.1,0], [0,0.1,0]], dtype=np.float32)

# Project to image coordinates
object_2d, _ = cv2.projectPoints(object_3d, rvec, tvec, 
                               camera_matrix, dist_coeffs)

# Draw virtual object on image
cv2.fillPoly(image, [object_2d.astype(int)], (0, 255, 0))
```

### 4. Distance Measurement

Measure real-world distances in images:
```python
# Given two image points and known depth plane
def measure_distance(point1, point2, depth, camera_matrix):
    fx, fy = camera_matrix[0,0], camera_matrix[1,1]
    cx, cy = camera_matrix[0,2], camera_matrix[1,2]
    
    # Convert to real-world coordinates
    x1 = (point1[0] - cx) * depth / fx
    y1 = (point1[1] - cy) * depth / fy
    
    x2 = (point2[0] - cx) * depth / fx  
    y2 = (point2[1] - cy) * depth / fy
    
    # Calculate Euclidean distance
    distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
    return distance
```

---

## Summary

Camera calibration involves:

1. **Mathematical Model**: Pinhole camera + lens distortion
2. **Calibration Pattern**: Known 3D geometry (chessboard)
3. **Parameter Estimation**: Minimize reprojection error
4. **Validation**: Check reprojection error < 1-2 pixels
5. **Application**: Undistortion, 3D reconstruction, AR

The key insight is that by observing a known 3D pattern from multiple viewpoints, we can solve for both the camera's internal properties (intrinsics) and its pose for each view (extrinsics). This enables accurate computer vision applications.

**Final Calibration Result:**
- **Camera Matrix K**: Focal lengths and principal point
- **Distortion Coefficients**: 5 parameters for lens correction  
- **Reprojection Error**: Quality metric (< 1 pixel is excellent)
- **Extrinsics**: Camera poses for each calibration image

This mathematical foundation enables all modern computer vision applications requiring accurate camera models!