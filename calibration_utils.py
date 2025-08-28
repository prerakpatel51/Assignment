import cv2
import numpy as np
import json
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from typing import List, Tuple, Optional
import glob


class IO:
    """File and device utilities"""
    
    @staticmethod
    def create_directory(path: str) -> None:
        """Create directory if it doesn't exist"""
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def list_images(directory: str, extension: str = "*.jpg") -> List[str]:
        """List all images in directory with given extension"""
        return glob.glob(os.path.join(directory, extension))
    
    @staticmethod
    def save_calibration_results(filename: str, camera_matrix: np.ndarray, 
                               dist_coeffs: np.ndarray, rvecs: List[np.ndarray], 
                               tvecs: List[np.ndarray], image_size: Tuple[int, int]) -> None:
        """Save calibration results to JSON file"""
        results = {
            'camera_matrix': camera_matrix.tolist(),
            'distortion_coefficients': dist_coeffs.tolist(),
            'rotation_vectors': [rvec.tolist() for rvec in rvecs],
            'translation_vectors': [tvec.tolist() for tvec in tvecs],
            'image_width': int(image_size[0]),
            'image_height': int(image_size[1])
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
    
    @staticmethod
    def load_calibration_results(filename: str) -> dict:
        """Load calibration results from JSON file"""
        with open(filename, 'r') as f:
            return json.load(f)


class Board:
    """Chessboard object model"""
    
    def __init__(self, pattern_size: Tuple[int, int], square_size: float):
        """
        Initialize chessboard
        Args:
            pattern_size: (width, height) of inner corners
            square_size: size of each square in meters
        """
        self.pattern_size = pattern_size
        self.square_size = square_size
        self.objp = self._create_object_points()
    
    def _create_object_points(self) -> np.ndarray:
        """Create 3D object points for chessboard pattern"""
        objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        objp *= self.square_size
        return objp
    
    @staticmethod
    def find_corners(image: np.ndarray, pattern_size: Tuple[int, int]) -> Tuple[bool, Optional[np.ndarray]]:
        """Find chessboard corners in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)
        
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        return ret, corners


class Img:
    """Image utilities"""
    
    @staticmethod
    def load_image(filepath: str) -> np.ndarray:
        """Load image from file"""
        return cv2.imread(filepath)
    
    @staticmethod
    def resize_image(image: np.ndarray, width: int = 800) -> np.ndarray:
        """Resize image maintaining aspect ratio"""
        height, orig_width = image.shape[:2]
        aspect_ratio = height / orig_width
        new_height = int(width * aspect_ratio)
        return cv2.resize(image, (width, new_height))
    
    @staticmethod
    def undistort_image(image: np.ndarray, camera_matrix: np.ndarray, 
                       dist_coeffs: np.ndarray) -> np.ndarray:
        """Undistort image using calibration parameters"""
        return cv2.undistort(image, camera_matrix, dist_coeffs, None, camera_matrix)


class Overlay:
    """Draw axes overlays and visualization"""
    
    @staticmethod
    def draw_axes(image: np.ndarray, camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                  rvec: np.ndarray, tvec: np.ndarray, length: float = 0.1) -> np.ndarray:
        """Draw 3D coordinate axes on image"""
        axes_points = np.float32([[0,0,0], [length,0,0], [0,length,0], [0,0,-length]]).reshape(-1,3)
        
        imgpts, _ = cv2.projectPoints(axes_points, rvec, tvec, camera_matrix, dist_coeffs)
        imgpts = np.int32(imgpts).reshape(-1, 2)
        
        img_with_axes = image.copy()
        corner = tuple(imgpts[0].ravel())
        
        # Draw axes: X-Red, Y-Green, Z-Blue
        img_with_axes = cv2.line(img_with_axes, corner, tuple(imgpts[1].ravel()), (0,0,255), 5)    # X - Red
        img_with_axes = cv2.line(img_with_axes, corner, tuple(imgpts[2].ravel()), (0,255,0), 5)    # Y - Green
        img_with_axes = cv2.line(img_with_axes, corner, tuple(imgpts[3].ravel()), (255,0,0), 5)    # Z - Blue
        
        return img_with_axes
    
    @staticmethod
    def draw_chessboard_corners(image: np.ndarray, corners: np.ndarray, 
                               pattern_size: Tuple[int, int], found: bool) -> np.ndarray:
        """Draw detected chessboard corners"""
        img_with_corners = image.copy()
        cv2.drawChessboardCorners(img_with_corners, pattern_size, corners, found)
        return img_with_corners


class Cam:
    """Camera math and calibration utilities"""
    
    @staticmethod
    def calibrate_camera(object_points: List[np.ndarray], image_points: List[np.ndarray],
                        image_size: Tuple[int, int]) -> Tuple[float, np.ndarray, np.ndarray, 
                                                             List[np.ndarray], List[np.ndarray]]:
        """
        Calibrate camera using object and image points
        Returns: (ret, camera_matrix, dist_coeffs, rvecs, tvecs)
        """
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            object_points, image_points, image_size, None, None)
        
        return ret, camera_matrix, dist_coeffs, rvecs, tvecs
    
    @staticmethod
    def compute_reprojection_error(object_points: List[np.ndarray], image_points: List[np.ndarray],
                                  camera_matrix: np.ndarray, dist_coeffs: np.ndarray,
                                  rvecs: List[np.ndarray], tvecs: List[np.ndarray]) -> float:
        """Compute mean reprojection error"""
        total_error = 0
        total_points = 0
        
        for i in range(len(object_points)):
            projected_points, _ = cv2.projectPoints(object_points[i], rvecs[i], tvecs[i], 
                                                  camera_matrix, dist_coeffs)
            error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
            total_error += error
            total_points += 1
        
        return total_error / total_points if total_points > 0 else 0


class Render:
    """Visualization and rendering utilities"""
    
    @staticmethod
    def plot_camera_poses_matplotlib(rvecs: List[np.ndarray], tvecs: List[np.ndarray], 
                                   board_size: Tuple[int, int], square_size: float) -> plt.Figure:
        """Plot camera poses using matplotlib"""
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot chessboard
        board_points = []
        for i in range(board_size[1] + 1):
            for j in range(board_size[0] + 1):
                board_points.append([j * square_size, i * square_size, 0])
        
        board_points = np.array(board_points)
        ax.scatter(board_points[:, 0], board_points[:, 1], board_points[:, 2], 
                  c='red', s=20, alpha=0.6, label='Chessboard')
        
        # Plot camera positions
        camera_positions = []
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv2.Rodrigues(rvec)
            camera_pos = -R.T @ tvec.flatten()
            camera_positions.append(camera_pos)
            
            # Draw camera coordinate system
            axis_length = 0.05
            axes = np.array([[axis_length, 0, 0], [0, axis_length, 0], [0, 0, axis_length]])
            rotated_axes = R.T @ axes.T
            
            # X, Y, Z axes
            colors = ['red', 'green', 'blue']
            for j, color in enumerate(colors):
                ax.plot([camera_pos[0], camera_pos[0] + rotated_axes[0, j]],
                       [camera_pos[1], camera_pos[1] + rotated_axes[1, j]],
                       [camera_pos[2], camera_pos[2] + rotated_axes[2, j]], color=color, linewidth=2)
        
        camera_positions = np.array(camera_positions)
        ax.scatter(camera_positions[:, 0], camera_positions[:, 1], camera_positions[:, 2], 
                  c='blue', s=100, alpha=0.8, label='Cameras', marker='^')
        
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.legend()
        ax.set_title('Camera Poses and Chessboard')
        
        return fig
    
    @staticmethod
    def plot_camera_poses_plotly(rvecs: List[np.ndarray], tvecs: List[np.ndarray], 
                                board_size: Tuple[int, int], square_size: float):
        """Plot camera poses using plotly for interactive visualization"""
        fig = go.Figure()
        
        # Plot chessboard points
        board_points = []
        for i in range(board_size[1] + 1):
            for j in range(board_size[0] + 1):
                board_points.append([j * square_size, i * square_size, 0])
        
        board_points = np.array(board_points)
        fig.add_trace(go.Scatter3d(
            x=board_points[:, 0], y=board_points[:, 1], z=board_points[:, 2],
            mode='markers',
            marker=dict(size=3, color='red'),
            name='Chessboard',
            opacity=0.6
        ))
        
        # Plot camera positions and orientations
        camera_positions = []
        for i, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
            R, _ = cv2.Rodrigues(rvec)
            camera_pos = -R.T @ tvec.flatten()
            camera_positions.append(camera_pos)
        
        camera_positions = np.array(camera_positions)
        fig.add_trace(go.Scatter3d(
            x=camera_positions[:, 0], y=camera_positions[:, 1], z=camera_positions[:, 2],
            mode='markers',
            marker=dict(size=8, color='blue', symbol='diamond'),
            name='Cameras'
        ))
        
        fig.update_layout(
            title='3D Camera Poses and Chessboard',
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Z (m)'
            ),
            width=800,
            height=600
        )
        
        return fig


class Calib:
    """Main calibration class"""
    
    def __init__(self, pattern_size: Tuple[int, int] = (9, 6), square_size: float = 0.025):
        self.board = Board(pattern_size, square_size)
        self.object_points = []
        self.image_points = []
        self.image_size = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.rvecs = None
        self.tvecs = None
        self.calibration_error = None
    
    def process_images(self, image_paths: List[str]) -> Tuple[int, int]:
        """
        Process calibration images to find chessboard corners
        Returns: (successful_detections, total_images)
        """
        self.object_points = []
        self.image_points = []
        successful_detections = 0
        
        for image_path in image_paths:
            image = Img.load_image(image_path)
            if image is None:
                continue
                
            if self.image_size is None:
                self.image_size = (image.shape[1], image.shape[0])
            
            ret, corners = Board.find_corners(image, self.board.pattern_size)
            
            if ret:
                self.object_points.append(self.board.objp)
                self.image_points.append(corners)
                successful_detections += 1
        
        return successful_detections, len(image_paths)
    
    def calibrate(self) -> bool:
        """Run camera calibration"""
        if len(self.object_points) < 10:
            return False
        
        ret, self.camera_matrix, self.dist_coeffs, self.rvecs, self.tvecs = Cam.calibrate_camera(
            self.object_points, self.image_points, self.image_size)
        
        if ret:
            self.calibration_error = Cam.compute_reprojection_error(
                self.object_points, self.image_points, self.camera_matrix, 
                self.dist_coeffs, self.rvecs, self.tvecs)
            return True
        
        return False
    
    def save_results(self, filename: str) -> None:
        """Save calibration results"""
        if self.camera_matrix is not None:
            IO.save_calibration_results(filename, self.camera_matrix, self.dist_coeffs, 
                                      self.rvecs, self.tvecs, self.image_size)
    
    def get_sample_images_with_axes(self, image_paths: List[str], num_samples: int = 5) -> List[np.ndarray]:
        """Get sample images with coordinate axes drawn"""
        if not self.rvecs or not self.tvecs:
            return []
        
        sample_images = []
        indices = np.linspace(0, len(image_paths) - 1, min(num_samples, len(image_paths)), dtype=int)
        
        for idx in indices:
            if idx < len(self.rvecs):
                image = Img.load_image(image_paths[idx])
                if image is not None:
                    img_with_axes = Overlay.draw_axes(
                        image, self.camera_matrix, self.dist_coeffs,
                        self.rvecs[idx], self.tvecs[idx], self.board.square_size * 3)
                    sample_images.append(img_with_axes)
        
        return sample_images