#!/usr/bin/env python3
"""
Demo script to show camera calibration system capabilities
"""

import sys
import os

# Add the current directory to path so we can import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from calibration_utils import Calib, IO, Board, Img, Overlay, Cam, Render

def demo_system():
    """Demonstrate the calibration system capabilities"""
    print("🎯 Camera Calibration System Demo\n")
    
    print("1. Creating calibration objects...")
    calibrator = Calib(pattern_size=(9, 6), square_size=0.025)
    print(f"   ✓ Calibrator initialized for {calibrator.board.pattern_size} pattern")
    print(f"   ✓ Square size: {calibrator.board.square_size}m")
    print(f"   ✓ Object points shape: {calibrator.board.objp.shape}")
    
    print("\n2. Available helper classes:")
    print("   ✓ IO - File and directory operations")
    print("   ✓ Board - Chessboard pattern model")
    print("   ✓ Img - Image processing utilities")
    print("   ✓ Overlay - Visualization and drawing")
    print("   ✓ Cam - Camera calibration algorithms")
    print("   ✓ Render - 3D plotting and visualization")
    print("   ✓ Calib - Main calibration orchestrator")
    
    print("\n3. System capabilities:")
    print("   📸 Process chessboard calibration images")
    print("   🎯 Compute camera intrinsic parameters")
    print("   📊 Generate 3D visualizations")
    print("   🖼️  Create undistortion previews")
    print("   💾 Save results to JSON format")
    print("   🎨 Interactive Gradio interface")
    
    print("\n4. Ready to use! 🚀")
    print("   Next: Open camera_calibration.ipynb in Jupyter Notebook")
    print("   or run: jupyter notebook camera_calibration.ipynb")

if __name__ == "__main__":
    demo_system()