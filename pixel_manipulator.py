import os
import cv2
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
import logging
import math
from pixel_watermark import PixelWatermark

app_logger = logging.getLogger(__name__)

class PixelManipulator:
    """
    Comprehensive pixel-level manipulation suite supporting:
    - RGB percentage-based and ROI area manipulation (5%, 10%, 25%, 75%, 100% matrix).
    - Custom RGB byte / byte-range operations (bitwise, clamp, delta, hex color).
    - Grayscale pixel manipulation suite (Luminance, BT.709, Bit-Plane Slicing 0-7, 
      Otsu/Multi-Level Thresholding, Gamma Correction, CLAHE, Histogram Equalization, Gray Level Slicing).
    - High-performance cached Excel statistical report generators.
    """

    # Shared style cache for openpyxl to ensure sub-second export
    _fill_cache = {}
    _font_cache = {}

    def __init__(self):
        self.watermark_tool = PixelWatermark()

    @classmethod
    def _get_fill(cls, color_hex):
        color_hex = str(color_hex).upper()
        if color_hex not in cls._fill_cache:
            cls._fill_cache[color_hex] = PatternFill(start_color=color_hex, end_color=color_hex, fill_type="solid")
        return cls._fill_cache[color_hex]

    @classmethod
    def _get_font(cls, size=10, color="000000", bold=False, name="Arial"):
        key = (size, color, bold, name)
        if key not in cls._font_cache:
            cls._font_cache[key] = Font(name=name, size=size, color=color, bold=bold)
        return cls._font_cache[key]

    def _parse_roi(self, roi, width, height):
        """Standardize Region of Interest (ROI) bounding box coordinates."""
        if not roi:
            return None
        try:
            if isinstance(roi, dict):
                if 'x' in roi and 'w' in roi:
                    x1 = int(roi.get('x', 0))
                    y1 = int(roi.get('y', 0))
                    w = int(roi.get('w', width))
                    h = int(roi.get('h', height))
                    x2 = x1 + w
                    y2 = y1 + h
                elif 'x1' in roi and 'x2' in roi:
                    x1 = int(roi.get('x1', 0))
                    y1 = int(roi.get('y1', 0))
                    x2 = int(roi.get('x2', width))
                    y2 = int(roi.get('y2', height))
                    w = x2 - x1
                    h = y2 - y1
                else:
                    return None
            elif isinstance(roi, (list, tuple)) and len(roi) >= 4:
                x1, y1, v3, v4 = int(roi[0]), int(roi[1]), int(roi[2]), int(roi[3])
                if v3 > x1 and v4 > y1:
                    x2, y2 = v3, v4
                    w, h = x2 - x1, y2 - y1
                else:
                    w, h = max(1, v3), max(1, v4)
                    x2, y2 = x1 + w, y1 + h
            else:
                return None

            x1 = max(0, min(width - 1, x1))
            y1 = max(0, min(height - 1, y1))
            x2 = max(x1 + 1, min(width, x2))
            y2 = max(y1 + 1, min(height, y2))
            w = x2 - x1
            h = y2 - y1

            if x1 == 0 and y1 == 0 and x2 == width and y2 == height:
                return None

            return {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'w': w, 'h': h}
        except Exception as e:
            app_logger.warning(f"Failed to parse ROI {roi}: {e}")
            return None

    # =========================================================================
    # GRAYSCALE MANIPULATION ENGINE
    # =========================================================================

    def convert_to_grayscale_array(self, img_rgb, method='luminance'):
        """
        Convert RGB array to 2D Grayscale uint8 array using designated algorithm.
        
        Methods:
            - 'luminance' / 'rec601': Perceptual ITU-R BT.601 (0.299 R + 0.587 G + 0.114 B)
            - 'bt709' / 'rec709': Modern HDTV BT.709 (0.2126 R + 0.7152 G + 0.0722 B)
            - 'average': (R + G + B) / 3
            - 'desaturation' / 'lightness': (max(R,G,B) + min(R,G,B)) / 2
            - 'channel_r': Red channel extracted
            - 'channel_g': Green channel extracted
            - 'channel_b': Blue channel extracted
        """
        r = img_rgb[:, :, 0].astype(np.float32)
        g = img_rgb[:, :, 1].astype(np.float32)
        b = img_rgb[:, :, 2].astype(np.float32)

        method = str(method).lower()
        if method in ['luminance', 'rec601', 'default']:
            gray = 0.299 * r + 0.587 * g + 0.114 * b
        elif method in ['bt709', 'rec709', 'hdtv']:
            gray = 0.2126 * r + 0.7152 * g + 0.0722 * b
        elif method in ['average', 'mean']:
            gray = (r + g + b) / 3.0
        elif method in ['desaturation', 'lightness']:
            max_c = np.maximum(np.maximum(r, g), b)
            min_c = np.minimum(np.minimum(r, g), b)
            gray = (max_c + min_c) / 2.0
        elif method in ['channel_r', 'red']:
            gray = r
        elif method in ['channel_g', 'green']:
            gray = g
        elif method in ['channel_b', 'blue']:
            gray = b
        else:
            gray = 0.299 * r + 0.587 * g + 0.114 * b

        return np.clip(np.round(gray), 0, 255).astype(np.uint8)

    def manipulate_grayscale(self, image_path, output_path, gray_method='luminance', op_type='threshold', op_params=None, roi=None):
        """
        Comprehensive Grayscale pixel manipulation.
        
        Args:
            image_path (str): Source image.
            output_path (str): Output manipulated image path.
            gray_method (str): Grayscale conversion method.
            op_type (str): Operation type ('bit_plane_slice', 'threshold', 'level_slice', 'gamma', 
                           'hist_equalize', 'clahe', 'contrast_stretch', 'invert', 'noise', 'custom_byte').
            op_params (dict): Operation parameters.
            roi (dict | list | None): Optional ROI area bounding box.
            
        Returns:
            dict: Comprehensive statistical and pixel verification payload.
        """
        try:
            if op_params is None:
                op_params = {}

            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Unable to read image at {image_path}")

            img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width, _ = img_rgb_orig.shape
            total_pixels = height * width

            # 1. Convert original to grayscale baseline
            gray_orig = self.convert_to_grayscale_array(img_rgb_orig, method=gray_method)
            gray_manip = gray_orig.copy().astype(np.float32)

            # 2. Parse ROI
            parsed_roi = self._parse_roi(roi, width, height)
            target_mask = np.zeros((height, width), dtype=bool)
            if parsed_roi:
                target_mask[parsed_roi['y1']:parsed_roi['y2'], parsed_roi['x1']:parsed_roi['x2']] = True
            else:
                target_mask[:, :] = True

            # 3. Apply Grayscale Operation
            op_type = str(op_type).lower()

            if op_type == 'bit_plane_slice':
                plane = int(op_params.get('bit_plane', 0))  # 0 = LSB, 7 = MSB
                plane = max(0, min(7, plane))
                mode = op_params.get('mode', 'isolate')  # 'isolate', 'zero_out', 'toggle'

                bit_val = 1 << plane
                if mode == 'isolate':
                    # Extract single bit plane scaled to full 0/255 contrast
                    extracted_bit = (gray_orig & bit_val) > 0
                    gray_manip[target_mask] = (extracted_bit[target_mask] * 255).astype(np.float32)
                elif mode == 'zero_out':
                    # Clear chosen bit plane (attacks LSB watermark if plane=0)
                    gray_int = gray_manip.astype(np.uint8)
                    gray_int[target_mask] = gray_int[target_mask] & (~bit_val & 0xFF)
                    gray_manip = gray_int.astype(np.float32)
                elif mode == 'toggle':
                    # Flip chosen bit plane
                    gray_int = gray_manip.astype(np.uint8)
                    gray_int[target_mask] = gray_int[target_mask] ^ bit_val
                    gray_manip = gray_int.astype(np.float32)

            elif op_type == 'threshold':
                threshold_mode = op_params.get('threshold_mode', 'manual')  # 'manual', 'otsu', 'multilevel'
                if threshold_mode == 'otsu':
                    _, otsu_thresh = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    gray_manip[target_mask] = otsu_thresh[target_mask]
                elif threshold_mode == 'multilevel':
                    levels = int(op_params.get('levels', 4))
                    levels = max(2, min(16, levels))
                    step = 256 / levels
                    quantized = (np.floor(gray_orig / step) * (255.0 / (levels - 1))).astype(np.float32)
                    gray_manip[target_mask] = quantized[target_mask]
                else:
                    cutoff = int(op_params.get('cutoff', 128))
                    cutoff = max(0, min(255, cutoff))
                    high_val = int(op_params.get('high_val', 255))
                    low_val = int(op_params.get('low_val', 0))
                    thresh_res = np.where(gray_orig >= cutoff, high_val, low_val).astype(np.float32)
                    gray_manip[target_mask] = thresh_res[target_mask]

            elif op_type == 'level_slice':
                min_val = int(op_params.get('min_val', 100))
                max_val = int(op_params.get('max_val', 180))
                highlight_val = int(op_params.get('highlight_val', 255))
                preserve_bg = bool(op_params.get('preserve_bg', True))

                in_slice = (gray_orig >= min_val) & (gray_orig <= max_val)
                sliced = gray_orig.copy().astype(np.float32)
                sliced[in_slice] = highlight_val
                if not preserve_bg:
                    sliced[~in_slice] = 0
                gray_manip[target_mask] = sliced[target_mask]

            elif op_type == 'gamma':
                gamma_val = float(op_params.get('gamma', 1.0))
                gamma_val = max(0.05, min(10.0, gamma_val))
                # Normalized power-law transform: s = 255 * (r / 255) ^ gamma
                norm = gray_orig.astype(np.float32) / 255.0
                gamma_corrected = np.power(norm, gamma_val) * 255.0
                gray_manip[target_mask] = gamma_corrected[target_mask]

            elif op_type == 'hist_equalize':
                eq_type = op_params.get('eq_type', 'global')  # 'global' or 'clahe'
                if eq_type == 'clahe':
                    clip_limit = float(op_params.get('clip_limit', 2.0))
                    grid_size = int(op_params.get('grid_size', 8))
                    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
                    clahe_img = clahe.apply(gray_orig)
                    gray_manip[target_mask] = clahe_img[target_mask].astype(np.float32)
                else:
                    eq_img = cv2.equalizeHist(gray_orig)
                    gray_manip[target_mask] = eq_img[target_mask].astype(np.float32)

            elif op_type == 'contrast_stretch':
                p_min = int(op_params.get('p_min', 0))
                p_max = int(op_params.get('p_max', 255))
                c_min = float(np.min(gray_orig))
                c_max = float(np.max(gray_orig))
                if c_max > c_min:
                    stretched = ((gray_orig.astype(np.float32) - c_min) / (c_max - c_min)) * (p_max - p_min) + p_min
                    gray_manip[target_mask] = np.clip(stretched, 0, 255)[target_mask]

            elif op_type == 'invert':
                gray_manip[target_mask] = 255.0 - gray_manip[target_mask]

            elif op_type == 'noise':
                noise_type = op_params.get('noise_type', 'salt_pepper')  # 'salt_pepper', 'gaussian', 'uniform'
                intensity = int(op_params.get('intensity', 50))
                pct = float(op_params.get('percentage', 10.0)) / 100.0

                if noise_type == 'salt_pepper':
                    noise_mask = np.random.random((height, width)) < pct
                    chosen_targets = target_mask & noise_mask
                    salt_or_pepper = np.random.choice([0.0, 255.0], size=(height, width))
                    gray_manip[chosen_targets] = salt_or_pepper[chosen_targets]
                elif noise_type == 'gaussian':
                    std_dev = (intensity / 255.0) * 50.0
                    gauss = np.random.normal(0, std_dev, (height, width))
                    gray_manip[target_mask] = np.clip(gray_manip[target_mask] + gauss[target_mask], 0, 255)
                else:
                    jitter = np.random.randint(-intensity, intensity + 1, size=(height, width)).astype(np.float32)
                    gray_manip[target_mask] = np.clip(gray_manip[target_mask] + jitter[target_mask], 0, 255)

            elif op_type == 'custom_byte':
                custom_op = op_params.get('operation', 'set_value')
                val = int(op_params.get('value', 255))
                delta = int(op_params.get('delta', 25))
                mask = int(op_params.get('mask', 1))

                if custom_op == 'set_value':
                    gray_manip[target_mask] = max(0, min(255, val))
                elif custom_op == 'offset':
                    gray_manip[target_mask] = np.clip(gray_manip[target_mask] + delta, 0, 255)
                elif custom_op == 'bitwise_xor':
                    gray_int = gray_manip.astype(np.uint8)
                    gray_int[target_mask] = gray_int[target_mask] ^ mask
                    gray_manip = gray_int.astype(np.float32)
                elif custom_op == 'bitwise_and':
                    gray_int = gray_manip.astype(np.uint8)
                    gray_int[target_mask] = gray_int[target_mask] & mask
                    gray_manip = gray_int.astype(np.float32)
                elif custom_op == 'bitwise_or':
                    gray_int = gray_manip.astype(np.uint8)
                    gray_int[target_mask] = gray_int[target_mask] | mask
                    gray_manip = gray_int.astype(np.float32)
                elif custom_op == 'clamp':
                    min_v = int(op_params.get('min_val', 0))
                    max_v = int(op_params.get('max_val', 255))
                    gray_manip[target_mask] = np.clip(gray_manip[target_mask], min_v, max_v)

            # Final uint8 manipulated grayscale image
            gray_manip_uint8 = np.clip(np.round(gray_manip), 0, 255).astype(np.uint8)

            # Save as 3-channel grayscale PNG image for browser display and watermarking
            gray_manip_3ch = cv2.cvtColor(gray_manip_uint8, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(output_path, gray_manip_3ch)

            # Generate Difference Heatmap Mask
            heatmap_path = output_path.rsplit('.', 1)[0] + "_heatmap.png"
            diff_mask = gray_manip_uint8 != gray_orig
            manip_count = int(np.sum(diff_mask))
            actual_pct = round((manip_count / total_pixels) * 100.0, 2)

            diff_visual = np.zeros((height, width, 3), dtype=np.uint8)
            diff_visual[diff_mask] = [255, 60, 60]
            gray_orig_3ch = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
            heatmap_bgr = cv2.addWeighted(gray_orig_3ch, 0.4, diff_visual, 0.6, 0)
            if parsed_roi:
                cv2.rectangle(heatmap_bgr, (parsed_roi['x1'], parsed_roi['y1']), (parsed_roi['x2'], parsed_roi['y2']), (255, 255, 0), 1)
            cv2.imwrite(heatmap_path, heatmap_bgr)

            # Error Metrics (MSE, MAE, PSNR)
            diff_sq = (gray_orig.astype(np.float64) - gray_manip_uint8.astype(np.float64)) ** 2
            diff_abs = np.abs(gray_orig.astype(np.float64) - gray_manip_uint8.astype(np.float64))
            mse_val = float(np.mean(diff_sq))
            mae_val = float(np.mean(diff_abs))
            psnr_val = 100.0 if mse_val == 0 else float(20 * math.log10(255.0 / math.sqrt(mse_val)))

            # Watermark check
            watermark_survival = self.watermark_tool.measure_watermark_robustness(image_path, output_path)
            extracted_text = self.watermark_tool.extract_lsb(output_path)

            # Grayscale Distribution Histogram (256 bins)
            orig_hist, _ = np.histogram(gray_orig, bins=256, range=(0, 256))
            manip_hist, _ = np.histogram(gray_manip_uint8, bins=256, range=(0, 256))

            # Grayscale Statistics
            stats = {
                'orig_min': int(np.min(gray_orig)),
                'orig_max': int(np.max(gray_orig)),
                'orig_mean': round(float(np.mean(gray_orig)), 2),
                'orig_std': round(float(np.std(gray_orig)), 2),
                'orig_median': int(np.median(gray_orig)),
                'manip_min': int(np.min(gray_manip_uint8)),
                'manip_max': int(np.max(gray_manip_uint8)),
                'manip_mean': round(float(np.mean(gray_manip_uint8)), 2),
                'manip_std': round(float(np.std(gray_manip_uint8)), 2),
                'manip_median': int(np.median(gray_manip_uint8)),
                'entropy_orig': round(float(self._calculate_entropy(gray_orig)), 3),
                'entropy_manip': round(float(self._calculate_entropy(gray_manip_uint8)), 3),
            }

            # Sample Display Grid (Max 40x40)
            max_dim = 40
            scale = min(1.0, max_dim / max(width, height))
            gw = max(1, int(round(width * scale)))
            gh = max(1, int(round(height * scale)))

            orig_small = cv2.resize(gray_orig, (gw, gh), interpolation=cv2.INTER_NEAREST)
            manip_small = cv2.resize(gray_manip_uint8, (gw, gh), interpolation=cv2.INTER_NEAREST)
            mask_small = cv2.resize(diff_mask.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0

            display_grid = []
            for y in range(gh):
                row = []
                for x in range(gw):
                    o_v = int(orig_small[y, x])
                    m_v = int(manip_small[y, x])
                    row.append({
                        'x': x,
                        'y': y,
                        'orig_gray': o_v,
                        'orig_hex': f"{o_v:02x}{o_v:02x}{o_v:02x}",
                        'manip_gray': m_v,
                        'manip_hex': f"{m_v:02x}{m_v:02x}{m_v:02x}",
                        'diff_gray': m_v - o_v,
                        'is_manipulated': bool(mask_small[y, x])
                    })
                display_grid.append(row)

            # Sample Pixel Records
            y_idx, x_idx = np.where(diff_mask)
            max_rec = min(len(y_idx), 1000)
            pixel_records = []
            if len(y_idx) > 0:
                if len(y_idx) > max_rec:
                    sample_idx = np.random.choice(len(y_idx), size=max_rec, replace=False)
                    sy, sx = y_idx[sample_idx], x_idx[sample_idx]
                else:
                    sy, sx = y_idx, x_idx

                for y, x in zip(sy, sx):
                    ov = int(gray_orig[y, x])
                    mv = int(gray_manip_uint8[y, x])
                    pixel_records.append({
                        'x': int(x),
                        'y': int(y),
                        'orig_gray': ov,
                        'orig_hex': f"{ov:02x}{ov:02x}{ov:02x}",
                        'orig_bin': f"{ov:08b}",
                        'manip_gray': mv,
                        'manip_hex': f"{mv:02x}{mv:02x}{mv:02x}",
                        'manip_bin': f"{mv:08b}",
                        'diff_gray': mv - ov,
                        'is_manipulated': True
                    })

            # Add sample unchanged
            uy, ux = np.where(~diff_mask)
            max_u = min(len(uy), max(200, 1200 - len(pixel_records)))
            if max_u > 0:
                su = np.random.choice(len(uy), size=max_u, replace=False)
                for idx in su:
                    y, x = uy[idx], ux[idx]
                    ov = int(gray_orig[y, x])
                    pixel_records.append({
                        'x': int(x),
                        'y': int(y),
                        'orig_gray': ov,
                        'orig_hex': f"{ov:02x}{ov:02x}{ov:02x}",
                        'orig_bin': f"{ov:08b}",
                        'manip_gray': ov,
                        'manip_hex': f"{ov:02x}{ov:02x}{ov:02x}",
                        'manip_bin': f"{ov:08b}",
                        'diff_gray': 0,
                        'is_manipulated': False
                    })

            return {
                'success': True,
                'is_grayscale': True,
                'width': width,
                'height': height,
                'total_pixels': total_pixels,
                'manipulated_count': manip_count,
                'actual_percentage': actual_pct,
                'roi': parsed_roi,
                'gray_method': gray_method,
                'operation': op_type,
                'op_params': op_params,
                'mse': round(mse_val, 3),
                'mae': round(mae_val, 3),
                'psnr': round(psnr_val, 2),
                'stats': stats,
                'histogram': {
                    'orig': orig_hist.tolist(),
                    'manip': manip_hist.tolist()
                },
                'watermark_survival': watermark_survival,
                'extracted_text': extracted_text or "None detected",
                'watermark_intact': bool(extracted_text and "Capstone" in extracted_text),
                'manipulated_filename': os.path.basename(output_path),
                'heatmap_filename': os.path.basename(heatmap_path),
                'display_grid': display_grid,
                'grid_width': gw,
                'grid_height': gh,
                'pixel_records': pixel_records,
                'total_records_count': len(pixel_records)
            }

        except Exception as e:
            app_logger.error(f"Error in grayscale manipulation: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    @staticmethod
    def _calculate_entropy(img_2d):
        """Calculate Shannon entropy of an 8-bit image array."""
        hist, _ = np.histogram(img_2d, bins=256, range=(0, 256))
        hist = hist.astype(np.float64) / img_2d.size
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    def manipulate_grayscale_percentage(self, image_path, output_path, percentage=10.0, method='gray_noise', intensity=50, roi=None, gray_method='luminance', op_params=None, seed=None, op_type=None):
        """
        Manipulate a chosen percentage (e.g. 5%, 10%, 25%, 50%, 75%, 100% or any arbitrary percentage
        like 37%, 48%, 99%) of pixels in a Grayscale image or within a user-selected ROI area.
        
        Args:
            image_path (str): Source image path.
            output_path (str): Output manipulated grayscale image path.
            percentage (float): Percentage of pixels to manipulate (0.1 to 100.0).
            method (str): Manipulation method ('gray_noise', 'lsb_corrupt', 'salt_pepper', 'gray_shift', 
                          'gray_invert', 'quantize', 'threshold', 'gamma', 'bit_plane_slice', 'clahe', 'hist_equalize').
            intensity (int): Intensity parameter (1-255).
            roi (dict | list | None): Optional bounding box {x1, y1, x2, y2}.
            gray_method (str): Algorithm for baseline grayscale conversion ('luminance', 'bt709', 'average', etc.).
            op_params (dict): Optional extra parameters for specific operations.
            seed (int | None): Random seed for reproducible manipulation.
            op_type (str | None): Optional alias for method.
            
        Returns:
            dict: Comprehensive statistical and pixel verification payload.
        """
        try:
            if op_type is not None:
                method = op_type
            if seed is not None:
                np.random.seed(seed)
            if op_params is None:
                op_params = {}

            percentage = float(max(0.1, min(100.0, percentage)))
            intensity = int(max(1, min(255, intensity)))

            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Unable to read image at {image_path}")

            img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width, _ = img_rgb_orig.shape
            total_pixels = height * width

            # 1. Convert to 2D Grayscale uint8 baseline
            gray_orig = self.convert_to_grayscale_array(img_rgb_orig, method=gray_method)
            gray_manip = gray_orig.copy().astype(np.float32)

            # 2. Parse ROI
            parsed_roi = self._parse_roi(roi, width, height)
            chosen_mask_2d = np.zeros((height, width), dtype=bool)

            if parsed_roi:
                rx1, ry1, rx2, ry2 = parsed_roi['x1'], parsed_roi['y1'], parsed_roi['x2'], parsed_roi['y2']
                roi_pixel_count = parsed_roi['w'] * parsed_roi['h']

                y_coords, x_coords = np.mgrid[ry1:ry2, rx1:rx2]
                roi_flat_y = y_coords.flatten()
                roi_flat_x = x_coords.flatten()

                num_to_manipulate = int(round((percentage / 100.0) * roi_pixel_count))
                num_to_manipulate = max(1, min(roi_pixel_count, num_to_manipulate))

                chosen_subset = np.random.choice(roi_pixel_count, size=num_to_manipulate, replace=False)
                chosen_y = roi_flat_y[chosen_subset]
                chosen_x = roi_flat_x[chosen_subset]
                chosen_mask_2d[chosen_y, chosen_x] = True
            else:
                num_to_manipulate = int(round((percentage / 100.0) * total_pixels))
                num_to_manipulate = max(1, min(total_pixels, num_to_manipulate))

                all_indices = np.arange(total_pixels)
                chosen_flat_indices = np.random.choice(all_indices, size=num_to_manipulate, replace=False)
                chosen_mask_flat = np.zeros(total_pixels, dtype=bool)
                chosen_mask_flat[chosen_flat_indices] = True
                chosen_mask_2d = chosen_mask_flat.reshape((height, width))

            # 3. Apply Chosen Grayscale Operation strictly on chosen_mask_2d pixels
            method = str(method).lower()

            if method in ['gray_noise', 'noise']:
                jitter = np.random.randint(-intensity, intensity + 1, size=(height, width)).astype(np.float32)
                gray_manip[chosen_mask_2d] = np.clip(gray_manip[chosen_mask_2d] + jitter[chosen_mask_2d], 0, 255)

            elif method in ['lsb_corrupt', 'lsb']:
                gray_int = gray_manip.astype(np.uint8)
                bit_plane = int(op_params.get('bit_plane', 0))
                mask_val = 1 << max(0, min(7, bit_plane))
                gray_int[chosen_mask_2d] = gray_int[chosen_mask_2d] ^ mask_val
                gray_manip = gray_int.astype(np.float32)

            elif method in ['salt_pepper', 'salt_and_pepper']:
                salt_or_pepper = np.random.choice([0.0, 255.0], size=(height, width))
                gray_manip[chosen_mask_2d] = salt_or_pepper[chosen_mask_2d]

            elif method in ['gray_shift', 'shift', 'offset']:
                shift_val = intensity if op_params.get('direction', 'positive') == 'positive' else -intensity
                gray_manip[chosen_mask_2d] = np.clip(gray_manip[chosen_mask_2d] + shift_val, 0, 255)

            elif method in ['gray_invert', 'invert']:
                gray_manip[chosen_mask_2d] = 255.0 - gray_manip[chosen_mask_2d]

            elif method in ['quantize', 'levels']:
                levels = int(op_params.get('levels', max(2, min(16, int(256 / max(16, intensity))))))
                step = 256.0 / levels
                quantized = (np.floor(gray_orig / step) * (255.0 / (levels - 1))).astype(np.float32)
                gray_manip[chosen_mask_2d] = quantized[chosen_mask_2d]

            elif method in ['threshold', 'otsu']:
                thresh_mode = op_params.get('threshold_mode', 'manual')
                if thresh_mode == 'otsu' or method == 'otsu':
                    _, otsu_res = cv2.threshold(gray_orig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    gray_manip[chosen_mask_2d] = otsu_res[chosen_mask_2d].astype(np.float32)
                else:
                    cutoff = int(op_params.get('cutoff', intensity))
                    high_val = int(op_params.get('high_val', 255))
                    low_val = int(op_params.get('low_val', 0))
                    thresh_res = np.where(gray_orig >= cutoff, high_val, low_val).astype(np.float32)
                    gray_manip[chosen_mask_2d] = thresh_res[chosen_mask_2d]

            elif method in ['gamma', 'power_law']:
                gamma_val = float(op_params.get('gamma', 1.0 + (intensity / 100.0)))
                gamma_val = max(0.05, min(10.0, gamma_val))
                norm = gray_orig.astype(np.float32) / 255.0
                gamma_corrected = np.power(norm, gamma_val) * 255.0
                gray_manip[chosen_mask_2d] = gamma_corrected[chosen_mask_2d]

            elif method in ['clahe', 'hist_equalize']:
                if method == 'clahe':
                    clahe = cv2.createCLAHE(clipLimit=float(op_params.get('clip_limit', 2.0)), tileGridSize=(8, 8))
                    eq_img = clahe.apply(gray_orig).astype(np.float32)
                else:
                    eq_img = cv2.equalizeHist(gray_orig).astype(np.float32)
                gray_manip[chosen_mask_2d] = eq_img[chosen_mask_2d]

            elif method in ['bit_plane_slice']:
                plane = int(op_params.get('bit_plane', 0))
                plane = max(0, min(7, plane))
                bit_val = 1 << plane
                mode = op_params.get('mode', 'isolate')
                if mode == 'isolate':
                    extracted_bit = ((gray_orig & bit_val) > 0).astype(np.float32) * 255.0
                    gray_manip[chosen_mask_2d] = extracted_bit[chosen_mask_2d]
                elif mode == 'zero_out':
                    gray_int = gray_manip.astype(np.uint8)
                    gray_int[chosen_mask_2d] = gray_int[chosen_mask_2d] & (~bit_val & 0xFF)
                    gray_manip = gray_int.astype(np.float32)
                else:
                    gray_int = gray_manip.astype(np.uint8)
                    gray_int[chosen_mask_2d] = gray_int[chosen_mask_2d] ^ bit_val
                    gray_manip = gray_int.astype(np.float32)

            else:
                jitter = np.random.randint(-intensity, intensity + 1, size=(height, width)).astype(np.float32)
                gray_manip[chosen_mask_2d] = np.clip(gray_manip[chosen_mask_2d] + jitter[chosen_mask_2d], 0, 255)

            # 4. Final uint8 manipulated grayscale image
            gray_manip_uint8 = np.clip(np.round(gray_manip), 0, 255).astype(np.uint8)

            # Save 3-channel grayscale PNG for browser display and watermarking
            gray_manip_3ch = cv2.cvtColor(gray_manip_uint8, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(output_path, gray_manip_3ch)

            # 5. Difference Heatmap Mask
            heatmap_path = output_path.rsplit('.', 1)[0] + "_heatmap.png"
            diff_mask = gray_manip_uint8 != gray_orig
            manip_count = int(np.sum(diff_mask))
            actual_pct = round((manip_count / total_pixels) * 100.0, 2)
            roi_actual_pct = round((manip_count / (parsed_roi['w'] * parsed_roi['h'])) * 100.0, 2) if parsed_roi else actual_pct

            diff_visual = np.zeros((height, width, 3), dtype=np.uint8)
            diff_visual[diff_mask] = [255, 60, 60]  # Red highlight for modified pixels
            gray_orig_3ch = cv2.cvtColor(gray_orig, cv2.COLOR_GRAY2BGR)
            heatmap_bgr = cv2.addWeighted(gray_orig_3ch, 0.4, diff_visual, 0.6, 0)
            if parsed_roi:
                cv2.rectangle(heatmap_bgr, (parsed_roi['x1'], parsed_roi['y1']), (parsed_roi['x2'], parsed_roi['y2']), (255, 255, 0), 1)
            cv2.imwrite(heatmap_path, heatmap_bgr)

            # 6. Error Metrics (MSE, MAE, PSNR)
            diff_sq = (gray_orig.astype(np.float64) - gray_manip_uint8.astype(np.float64)) ** 2
            diff_abs = np.abs(gray_orig.astype(np.float64) - gray_manip_uint8.astype(np.float64))
            mse_val = float(np.mean(diff_sq))
            mae_val = float(np.mean(diff_abs))
            psnr_val = 100.0 if mse_val == 0 else float(20 * math.log10(255.0 / math.sqrt(mse_val)))

            # Watermark check
            watermark_survival = self.watermark_tool.measure_watermark_robustness(image_path, output_path)
            extracted_text = self.watermark_tool.extract_lsb(output_path)

            # Grayscale Distribution Histogram (256 bins)
            orig_hist, _ = np.histogram(gray_orig, bins=256, range=(0, 256))
            manip_hist, _ = np.histogram(gray_manip_uint8, bins=256, range=(0, 256))

            # Grayscale Statistics
            stats = {
                'orig_min': int(np.min(gray_orig)),
                'orig_max': int(np.max(gray_orig)),
                'orig_mean': round(float(np.mean(gray_orig)), 2),
                'orig_std': round(float(np.std(gray_orig)), 2),
                'orig_median': int(np.median(gray_orig)),
                'manip_min': int(np.min(gray_manip_uint8)),
                'manip_max': int(np.max(gray_manip_uint8)),
                'manip_mean': round(float(np.mean(gray_manip_uint8)), 2),
                'manip_std': round(float(np.std(gray_manip_uint8)), 2),
                'manip_median': int(np.median(gray_manip_uint8)),
                'entropy_orig': round(float(self._calculate_entropy(gray_orig)), 3),
                'entropy_manip': round(float(self._calculate_entropy(gray_manip_uint8)), 3),
            }

            # 7. Sample Display Grid (Max 40x40)
            max_dim = 40
            scale = min(1.0, max_dim / max(width, height))
            gw = max(1, int(round(width * scale)))
            gh = max(1, int(round(height * scale)))

            orig_small = cv2.resize(gray_orig, (gw, gh), interpolation=cv2.INTER_NEAREST)
            manip_small = cv2.resize(gray_manip_uint8, (gw, gh), interpolation=cv2.INTER_NEAREST)
            mask_small = cv2.resize(diff_mask.astype(np.uint8), (gw, gh), interpolation=cv2.INTER_NEAREST) > 0

            display_grid = []
            for y in range(gh):
                row = []
                for x in range(gw):
                    o_v = int(orig_small[y, x])
                    m_v = int(manip_small[y, x])
                    row.append({
                        'x': x,
                        'y': y,
                        'orig_gray': o_v,
                        'orig_hex': f"{o_v:02x}{o_v:02x}{o_v:02x}",
                        'manip_gray': m_v,
                        'manip_hex': f"{m_v:02x}{m_v:02x}{m_v:02x}",
                        'diff_gray': m_v - o_v,
                        'is_manipulated': bool(mask_small[y, x])
                    })
                display_grid.append(row)

            # 8. Sample Pixel Records for Table Inspector
            y_idx, x_idx = np.where(diff_mask)
            max_rec = min(len(y_idx), 1000)
            pixel_records = []
            if len(y_idx) > 0:
                if len(y_idx) > max_rec:
                    sample_idx = np.random.choice(len(y_idx), size=max_rec, replace=False)
                    sy, sx = y_idx[sample_idx], x_idx[sample_idx]
                else:
                    sy, sx = y_idx, x_idx

                for y, x in zip(sy, sx):
                    ov = int(gray_orig[y, x])
                    mv = int(gray_manip_uint8[y, x])
                    pixel_records.append({
                        'x': int(x),
                        'y': int(y),
                        'orig_gray': ov,
                        'orig_hex': f"{ov:02x}{ov:02x}{ov:02x}",
                        'orig_bin': f"{ov:08b}",
                        'manip_gray': mv,
                        'manip_hex': f"{mv:02x}{mv:02x}{mv:02x}",
                        'manip_bin': f"{mv:08b}",
                        'diff_gray': mv - ov,
                        'is_manipulated': True
                    })

            uy, ux = np.where(~diff_mask)
            max_u = min(len(uy), max(200, 1200 - len(pixel_records)))
            if max_u > 0:
                su = np.random.choice(len(uy), size=max_u, replace=False)
                for idx in su:
                    y, x = uy[idx], ux[idx]
                    ov = int(gray_orig[y, x])
                    pixel_records.append({
                        'x': int(x),
                        'y': int(y),
                        'orig_gray': ov,
                        'orig_hex': f"{ov:02x}{ov:02x}{ov:02x}",
                        'orig_bin': f"{ov:08b}",
                        'manip_gray': ov,
                        'manip_hex': f"{ov:02x}{ov:02x}{ov:02x}",
                        'manip_bin': f"{ov:08b}",
                        'diff_gray': 0,
                        'is_manipulated': False
                    })

            return {
                'success': True,
                'is_grayscale': True,
                'width': width,
                'height': height,
                'total_pixels': total_pixels,
                'manipulated_count': manip_count,
                'actual_percentage': actual_pct,
                'roi_actual_percentage': roi_actual_pct,
                'requested_percentage': percentage,
                'roi': parsed_roi,
                'gray_method': gray_method,
                'method': method,
                'intensity': intensity,
                'op_params': op_params,
                'mse': round(mse_val, 3),
                'mae': round(mae_val, 3),
                'psnr': round(psnr_val, 2),
                'stats': stats,
                'histogram': {
                    'orig': orig_hist.tolist(),
                    'manip': manip_hist.tolist()
                },
                'watermark_survival': watermark_survival,
                'extracted_text': extracted_text or "None detected",
                'watermark_intact': bool(extracted_text and "Capstone" in extracted_text),
                'manipulated_filename': os.path.basename(output_path),
                'heatmap_filename': os.path.basename(heatmap_path),
                'display_grid': display_grid,
                'grid_width': gw,
                'grid_height': gh,
                'pixel_records': pixel_records,
                'total_records_count': len(pixel_records)
            }

        except Exception as e:
            app_logger.error(f"Error in grayscale percentage manipulation: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def manipulate_grayscale_multi_percentages(self, image_path, output_dir, percentages=None, method='gray_noise', intensity=50, roi=None, gray_method='luminance', op_params=None, op_type=None):
        """
        Simultaneously execute grayscale manipulation across standard percentages (5%, 10%, 25%, 50%, 75%, 100%)
        or any custom list of percentages.
        """
        try:
            if op_type is not None:
                method = op_type
            if percentages is None:
                percentages = [5.0, 10.0, 25.0, 50.0, 75.0, 100.0]

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            roi_suffix = ""
            if roi:
                parsed_r = self._parse_roi(roi, 10000, 10000)
                if parsed_r:
                    roi_suffix = f"_roi_{parsed_r['x1']}_{parsed_r['y1']}_{parsed_r['w']}x{parsed_r['h']}"

            tiers_result = []

            for pct in percentages:
                pct_clean = int(pct) if pct == int(pct) else pct
                tier_filename = f"{base_name}_gray_manip_{pct_clean}pct_{method}{roi_suffix}.png"
                tier_path = os.path.join(output_dir, tier_filename)

                tier_stat = self.manipulate_grayscale_percentage(
                    image_path=image_path,
                    output_path=tier_path,
                    percentage=pct,
                    method=method,
                    intensity=intensity,
                    roi=roi,
                    gray_method=gray_method,
                    op_params=op_params,
                    seed=42 + int(pct * 10)
                )

                if tier_stat.get('success'):
                    tiers_result.append({
                        'percentage': pct,
                        'requested_percentage': tier_stat.get('requested_percentage'),
                        'actual_percentage': tier_stat.get('actual_percentage'),
                        'roi_actual_percentage': tier_stat.get('roi_actual_percentage'),
                        'manipulated_count': tier_stat.get('manipulated_count'),
                        'total_pixels': tier_stat.get('total_pixels'),
                        'mse': tier_stat.get('mse'),
                        'mae': tier_stat.get('mae'),
                        'psnr': tier_stat.get('psnr'),
                        'entropy_orig': tier_stat.get('stats', {}).get('entropy_orig'),
                        'entropy_manip': tier_stat.get('stats', {}).get('entropy_manip'),
                        'watermark_survival': tier_stat.get('watermark_survival'),
                        'extracted_text': tier_stat.get('extracted_text'),
                        'watermark_intact': tier_stat.get('watermark_intact'),
                        'manipulated_filename': tier_stat.get('manipulated_filename'),
                        'heatmap_filename': tier_stat.get('heatmap_filename'),
                        'method': method,
                        'intensity': intensity,
                        'gray_method': gray_method
                    })
                else:
                    tiers_result.append({
                        'percentage': pct,
                        'error': tier_stat.get('error')
                    })

            img_bgr = cv2.imread(image_path)
            h, w = img_bgr.shape[:2] if img_bgr is not None else (0, 0)
            parsed_roi = self._parse_roi(roi, w, h)

            return {
                'success': True,
                'is_grayscale': True,
                'source_filename': os.path.basename(image_path),
                'method': method,
                'intensity': intensity,
                'gray_method': gray_method,
                'roi': parsed_roi,
                'width': w,
                'height': h,
                'total_pixels': w * h,
                'tiers': tiers_result
            }

        except Exception as e:
            app_logger.error(f"Grayscale multi-percentage manipulation error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    # =========================================================================
    # RGB PERCENTAGE AND ROI MANIPULATION ENGINE
    # =========================================================================

    def manipulate_image(self, image_path, output_path, percentage=10.0, method='noise', intensity=50, roi=None, seed=None):
        """
        Manipulate a chosen percentage of pixels in the image or within a selected ROI area.
        """
        try:
            if seed is not None:
                np.random.seed(seed)

            percentage = float(max(0.1, min(100.0, percentage)))
            
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Unable to read image at {image_path}")

            img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width, channels = img_rgb_orig.shape
            total_pixels = height * width

            parsed_roi = self._parse_roi(roi, width, height)
            
            if parsed_roi:
                rx1, ry1, rx2, ry2 = parsed_roi['x1'], parsed_roi['y1'], parsed_roi['x2'], parsed_roi['y2']
                roi_pixel_count = parsed_roi['w'] * parsed_roi['h']
                
                y_coords, x_coords = np.mgrid[ry1:ry2, rx1:rx2]
                roi_flat_y = y_coords.flatten()
                roi_flat_x = x_coords.flatten()
                
                num_to_manipulate = int(round((percentage / 100.0) * roi_pixel_count))
                num_to_manipulate = max(1, min(roi_pixel_count, num_to_manipulate))
                
                chosen_subset = np.random.choice(roi_pixel_count, size=num_to_manipulate, replace=False)
                chosen_y = roi_flat_y[chosen_subset]
                chosen_x = roi_flat_x[chosen_subset]
                
                chosen_mask_2d = np.zeros((height, width), dtype=bool)
                chosen_mask_2d[chosen_y, chosen_x] = True
            else:
                num_to_manipulate = int(round((percentage / 100.0) * total_pixels))
                num_to_manipulate = max(1, min(total_pixels, num_to_manipulate))
                
                all_indices = np.arange(total_pixels)
                chosen_flat_indices = np.random.choice(all_indices, size=num_to_manipulate, replace=False)
                chosen_mask_flat = np.zeros(total_pixels, dtype=bool)
                chosen_mask_flat[chosen_flat_indices] = True
                chosen_mask_2d = chosen_mask_flat.reshape((height, width))

            manip_rgb = img_rgb_orig.copy().astype(np.float32)

            # Apply manipulation based on selected method
            if method == 'noise':
                noise = np.random.randint(-intensity, intensity + 1, size=(height, width, channels)).astype(np.float32)
                manip_rgb[chosen_mask_2d] = np.clip(manip_rgb[chosen_mask_2d] + noise[chosen_mask_2d], 0, 255)

            elif method == 'lsb_corrupt':
                manip_int = manip_rgb.astype(np.uint8)
                bit_mask = np.random.randint(1, 8, size=(height, width, channels), dtype=np.uint8)
                manip_int[chosen_mask_2d] = manip_int[chosen_mask_2d] ^ bit_mask[chosen_mask_2d]
                manip_rgb = manip_int.astype(np.float32)

            elif method == 'salt_pepper':
                salt_or_pepper = np.random.choice([0.0, 255.0], size=(height, width, channels))
                manip_rgb[chosen_mask_2d] = salt_or_pepper[chosen_mask_2d]

            elif method == 'channel_shift':
                shift_r = intensity
                shift_g = -int(intensity * 0.5)
                shift_b = int(intensity * 0.8)
                shift_arr = np.array([shift_r, shift_g, shift_b], dtype=np.float32)
                manip_rgb[chosen_mask_2d] = np.clip(manip_rgb[chosen_mask_2d] + shift_arr, 0, 255)

            elif method == 'invert':
                manip_rgb[chosen_mask_2d] = 255.0 - manip_rgb[chosen_mask_2d]

            elif method == 'brightness':
                factor = 1.0 + ((intensity - 128) / 128.0)
                offset = (intensity - 128) * 0.5
                manip_rgb[chosen_mask_2d] = np.clip(manip_rgb[chosen_mask_2d] * factor + offset, 0, 255)

            else:
                noise = np.random.randint(-intensity, intensity + 1, size=(height, width, channels)).astype(np.float32)
                manip_rgb[chosen_mask_2d] = np.clip(manip_rgb[chosen_mask_2d] + noise[chosen_mask_2d], 0, 255)

            manip_rgb_uint8 = np.clip(manip_rgb, 0, 255).astype(np.uint8)

            return self._build_stats_payload(
                img_rgb_orig=img_rgb_orig,
                manip_rgb_uint8=manip_rgb_uint8,
                chosen_mask_2d=chosen_mask_2d,
                image_path=image_path,
                output_path=output_path,
                method=method,
                intensity=intensity,
                requested_percentage=percentage,
                num_manipulated=num_to_manipulate,
                roi=parsed_roi
            )

        except Exception as e:
            app_logger.error(f"Error in pixel manipulation: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def manipulate_multi_percentages(self, image_path, output_dir, percentages=None, method='noise', intensity=50, roi=None):
        """
        Simultaneously execute manipulation across all standard percentages (5%, 10%, 25%, 75%, 100%).
        """
        try:
            if percentages is None:
                percentages = [5.0, 10.0, 25.0, 75.0, 100.0]

            base_name = os.path.splitext(os.path.basename(image_path))[0]
            roi_suffix = ""
            if roi:
                parsed_r = self._parse_roi(roi, 10000, 10000)
                if parsed_r:
                    roi_suffix = f"_roi_{parsed_r['x1']}_{parsed_r['y1']}_{parsed_r['w']}x{parsed_r['h']}"

            tiers_result = []

            for pct in percentages:
                pct_clean = int(pct) if pct == int(pct) else pct
                tier_filename = f"{base_name}_manip_{pct_clean}pct_{method}{roi_suffix}.png"
                tier_path = os.path.join(output_dir, tier_filename)

                tier_stat = self.manipulate_image(
                    image_path=image_path,
                    output_path=tier_path,
                    percentage=pct,
                    method=method,
                    intensity=intensity,
                    roi=roi,
                    seed=42 + int(pct * 10)
                )

                if tier_stat.get('success'):
                    tiers_result.append({
                        'percentage': pct,
                        'requested_percentage': tier_stat.get('requested_percentage'),
                        'actual_percentage': tier_stat.get('actual_percentage'),
                        'manipulated_count': tier_stat.get('manipulated_count'),
                        'total_pixels': tier_stat.get('total_pixels'),
                        'mae': tier_stat.get('mae'),
                        'mse': tier_stat.get('mse'),
                        'psnr': tier_stat.get('psnr'),
                        'watermark_survival': tier_stat.get('watermark_survival'),
                        'extracted_text': tier_stat.get('extracted_text'),
                        'watermark_intact': tier_stat.get('watermark_intact'),
                        'manipulated_filename': tier_stat.get('manipulated_filename'),
                        'heatmap_filename': tier_stat.get('heatmap_filename'),
                        'method': method,
                        'intensity': intensity
                    })
                else:
                    tiers_result.append({
                        'percentage': pct,
                        'error': tier_stat.get('error')
                    })

            img_bgr = cv2.imread(image_path)
            h, w = img_bgr.shape[:2] if img_bgr is not None else (0, 0)
            parsed_roi = self._parse_roi(roi, w, h)

            return {
                'success': True,
                'source_filename': os.path.basename(image_path),
                'method': method,
                'intensity': intensity,
                'roi': parsed_roi,
                'width': w,
                'height': h,
                'total_pixels': w * h,
                'tiers': tiers_result
            }

        except Exception as e:
            app_logger.error(f"Multi-percentage manipulation batch error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def manipulate_custom_bytes(self, image_path, output_path, roi=None, channels=None, operation='set_value', op_params=None, byte_range_filter=None):
        """
        Directly manipulate pixel bytes across any channel, coordinate range, or byte value threshold.
        """
        try:
            if op_params is None:
                op_params = {}

            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Unable to read image at {image_path}")

            img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width, _ = img_rgb_orig.shape
            total_pixels = height * width

            # Parse channels
            if channels is None or channels == 'all' or channels == ['all']:
                target_channels = [0, 1, 2]
                channels_label = ['R', 'G', 'B']
            elif isinstance(channels, str):
                ch_map = {'R': [0], 'G': [1], 'B': [2], 'RGB': [0, 1, 2], 'ALL': [0, 1, 2]}
                target_channels = ch_map.get(channels.upper(), [0, 1, 2])
                channels_label = [channels.upper()]
            elif isinstance(channels, (list, tuple)):
                ch_map = {'R': 0, 'G': 1, 'B': 2, 'RED': 0, 'GREEN': 1, 'BLUE': 2, 0: 0, 1: 1, 2: 2}
                target_channels = [ch_map[c] for c in channels if c in ch_map or (isinstance(c, str) and c.upper() in ch_map)]
                if not target_channels:
                    target_channels = [0, 1, 2]
                channels_label = [str(c).upper() for c in channels]
            else:
                target_channels = [0, 1, 2]
                channels_label = ['R', 'G', 'B']

            parsed_roi = self._parse_roi(roi, width, height)
            target_mask = np.zeros((height, width), dtype=bool)

            if parsed_roi:
                rx1, ry1, rx2, ry2 = parsed_roi['x1'], parsed_roi['y1'], parsed_roi['x2'], parsed_roi['y2']
                target_mask[ry1:ry2, rx1:rx2] = True
            else:
                target_mask[:, :] = True

            if byte_range_filter:
                b_min = int(byte_range_filter.get('min', 0)) if isinstance(byte_range_filter, dict) else int(byte_range_filter[0])
                b_max = int(byte_range_filter.get('max', 255)) if isinstance(byte_range_filter, dict) else int(byte_range_filter[1])
                channel_filter_mask = np.ones((height, width), dtype=bool)
                for ch_idx in target_channels:
                    ch_vals = img_rgb_orig[:, :, ch_idx]
                    channel_filter_mask = channel_filter_mask & (ch_vals >= b_min) & (ch_vals <= b_max)
                target_mask = target_mask & channel_filter_mask

            manip_rgb = img_rgb_orig.copy().astype(np.float32)

            if operation == 'set_value':
                hex_color = op_params.get('hex_color')
                if hex_color and isinstance(hex_color, str) and len(hex_color.replace('#', '')) == 6:
                    hc = hex_color.replace('#', '')
                    r_val = int(hc[0:2], 16)
                    g_val = int(hc[2:4], 16)
                    b_val = int(hc[4:6], 16)
                    manip_rgb[target_mask] = [r_val, g_val, b_val]
                else:
                    byte_val = int(op_params.get('value', 255))
                    byte_val = max(0, min(255, byte_val))
                    for ch in target_channels:
                        manip_rgb[target_mask, ch] = byte_val

            elif operation == 'offset':
                delta = int(op_params.get('delta', 30))
                for ch in target_channels:
                    manip_rgb[target_mask, ch] = np.clip(manip_rgb[target_mask, ch].astype(np.int16) + delta, 0, 255)

            elif operation == 'bitwise_xor':
                mask = int(op_params.get('mask', 1))
                manip_int = manip_rgb.astype(np.uint8)
                for ch in target_channels:
                    manip_int[target_mask, ch] = manip_int[target_mask, ch] ^ mask
                manip_rgb = manip_int.astype(np.float32)

            elif operation == 'bitwise_and':
                mask = int(op_params.get('mask', 254))
                manip_int = manip_rgb.astype(np.uint8)
                for ch in target_channels:
                    manip_int[target_mask, ch] = manip_int[target_mask, ch] & mask
                manip_rgb = manip_int.astype(np.float32)

            elif operation == 'bitwise_or':
                mask = int(op_params.get('mask', 1))
                manip_int = manip_rgb.astype(np.uint8)
                for ch in target_channels:
                    manip_int[target_mask, ch] = manip_int[target_mask, ch] | mask
                manip_rgb = manip_int.astype(np.float32)

            elif operation == 'invert':
                for ch in target_channels:
                    manip_rgb[target_mask, ch] = 255.0 - manip_rgb[target_mask, ch]

            elif operation == 'random_range':
                min_v = max(0, min(255, int(op_params.get('min_val', 0))))
                max_v = max(min_v, min(255, int(op_params.get('max_val', 255))))
                num_target = np.sum(target_mask)
                if num_target > 0:
                    for ch in target_channels:
                        rand_bytes = np.random.randint(min_v, max_v + 1, size=num_target).astype(np.float32)
                        manip_rgb[target_mask, ch] = rand_bytes

            elif operation == 'clamp':
                min_v = max(0, min(255, int(op_params.get('min_val', 0))))
                max_v = max(min_v, min(255, int(op_params.get('max_val', 255))))
                for ch in target_channels:
                    manip_rgb[target_mask, ch] = np.clip(manip_rgb[target_mask, ch], min_v, max_v)

            else:
                delta = int(op_params.get('delta', 20))
                for ch in target_channels:
                    manip_rgb[target_mask, ch] = np.clip(manip_rgb[target_mask, ch].astype(np.int16) + delta, 0, 255)

            manip_rgb_uint8 = np.clip(manip_rgb, 0, 255).astype(np.uint8)
            actual_diff_mask = np.any(manip_rgb_uint8 != img_rgb_orig, axis=2)
            actual_manip_count = int(np.sum(actual_diff_mask))
            actual_pct = round((actual_manip_count / total_pixels) * 100.0, 2)

            method_desc = f"custom_byte_{operation}"
            return self._build_stats_payload(
                img_rgb_orig=img_rgb_orig,
                manip_rgb_uint8=manip_rgb_uint8,
                chosen_mask_2d=actual_diff_mask,
                image_path=image_path,
                output_path=output_path,
                method=method_desc,
                intensity=op_params.get('delta', op_params.get('value', op_params.get('mask', 50))),
                requested_percentage=actual_pct,
                num_manipulated=actual_manip_count,
                roi=parsed_roi,
                custom_details={
                    'operation': operation,
                    'channels': channels_label,
                    'op_params': op_params,
                    'byte_range_filter': byte_range_filter
                }
            )

        except Exception as e:
            app_logger.error(f"Error in custom byte manipulation: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _build_stats_payload(self, img_rgb_orig, manip_rgb_uint8, chosen_mask_2d, image_path, output_path, method, intensity, requested_percentage, num_manipulated, roi=None, custom_details=None):
        """
        Fast vectorized helper method to compute error metrics, watermark robustness,
        RGB channel stats, display grid, and per-pixel records.
        """
        height, width, _ = img_rgb_orig.shape
        total_pixels = height * width

        # Save manipulated image
        manip_bgr = cv2.cvtColor(manip_rgb_uint8, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, manip_bgr)

        # Generate Difference Heatmap Mask
        heatmap_path = output_path.rsplit('.', 1)[0] + "_heatmap.png"
        diff_visual = np.zeros((height, width, 3), dtype=np.uint8)
        diff_visual[chosen_mask_2d] = [255, 60, 60]
        
        orig_gray = cv2.cvtColor(img_rgb_orig, cv2.COLOR_RGB2GRAY)
        orig_gray_3ch = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
        heatmap_bgr = cv2.addWeighted(orig_gray_3ch, 0.35, diff_visual, 0.65, 0)
        
        if roi:
            cv2.rectangle(heatmap_bgr, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), (255, 255, 0), 1)

        cv2.imwrite(heatmap_path, heatmap_bgr)

        # Overall Error Metrics
        diff_sq = (img_rgb_orig.astype(np.float64) - manip_rgb_uint8.astype(np.float64)) ** 2
        diff_abs = np.abs(img_rgb_orig.astype(np.float64) - manip_rgb_uint8.astype(np.float64))
        mse_val = float(np.mean(diff_sq))
        mae_val = float(np.mean(diff_abs))
        psnr_val = 100.0 if mse_val == 0 else float(20 * math.log10(255.0 / math.sqrt(mse_val)))

        watermark_survival = self.watermark_tool.measure_watermark_robustness(image_path, output_path)
        extracted_text = self.watermark_tool.extract_lsb(output_path)

        # Channel Statistics
        channel_stats = {}
        for ch_idx, ch_name in enumerate(['red', 'green', 'blue']):
            o_ch = img_rgb_orig[:, :, ch_idx]
            m_ch = manip_rgb_uint8[:, :, ch_idx]
            channel_stats[ch_name] = {
                'orig_min': int(np.min(o_ch)),
                'orig_max': int(np.max(o_ch)),
                'orig_mean': round(float(np.mean(o_ch)), 2),
                'orig_std': round(float(np.std(o_ch)), 2),
                'orig_median': int(np.median(o_ch)),
                'manip_min': int(np.min(m_ch)),
                'manip_max': int(np.max(m_ch)),
                'manip_mean': round(float(np.mean(m_ch)), 2),
                'manip_std': round(float(np.std(m_ch)), 2),
                'manip_median': int(np.median(m_ch)),
                'mae': round(float(np.mean(np.abs(o_ch.astype(np.float64) - m_ch.astype(np.float64)))), 2),
                'mse': round(float(np.mean((o_ch.astype(np.float64) - m_ch.astype(np.float64)) ** 2)), 2),
            }

        # Generate Sampled Display Grid (Max 40x40)
        max_grid_dim = 40
        scale = min(1.0, max_grid_dim / max(width, height))
        grid_w = max(1, int(round(width * scale)))
        grid_h = max(1, int(round(height * scale)))

        orig_resized = cv2.resize(img_rgb_orig, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
        manip_resized = cv2.resize(manip_rgb_uint8, (grid_w, grid_h), interpolation=cv2.INTER_NEAREST)
        mask_resized = cv2.resize(chosen_mask_2d.astype(np.uint8), (grid_w, grid_h), interpolation=cv2.INTER_NEAREST) > 0

        display_grid = []
        for y in range(grid_h):
            row = []
            for x in range(grid_w):
                o_r, o_g, o_b = int(orig_resized[y, x, 0]), int(orig_resized[y, x, 1]), int(orig_resized[y, x, 2])
                m_r, m_g, m_b = int(manip_resized[y, x, 0]), int(manip_resized[y, x, 1]), int(manip_resized[y, x, 2])
                row.append({
                    'x': x,
                    'y': y,
                    'orig_r': o_r,
                    'orig_g': o_g,
                    'orig_b': o_b,
                    'orig_hex': f"{o_r:02x}{o_g:02x}{o_b:02x}",
                    'manip_r': m_r,
                    'manip_g': m_g,
                    'manip_b': m_b,
                    'manip_hex': f"{m_r:02x}{m_g:02x}{m_b:02x}",
                    'diff_r': m_r - o_r,
                    'diff_g': m_g - o_g,
                    'diff_b': m_b - o_b,
                    'is_manipulated': bool(mask_resized[y, x])
                })
            display_grid.append(row)

        # Build Pixel Records List
        pixel_records = []
        y_indices, x_indices = np.where(chosen_mask_2d)
        max_manip_records = min(len(y_indices), 1000)
        
        orig_r, orig_g, orig_b = img_rgb_orig[:, :, 0], img_rgb_orig[:, :, 1], img_rgb_orig[:, :, 2]
        manip_r, manip_g, manip_b = manip_rgb_uint8[:, :, 0], manip_rgb_uint8[:, :, 1], manip_rgb_uint8[:, :, 2]

        if len(y_indices) > 0:
            if len(y_indices) > max_manip_records:
                selected_sample = np.random.choice(len(y_indices), size=max_manip_records, replace=False)
                sampled_y, sampled_x = y_indices[selected_sample], x_indices[selected_sample]
            else:
                sampled_y, sampled_x = y_indices, x_indices

            for y, x in zip(sampled_y, sampled_x):
                o_r, o_g, o_b = int(orig_r[y, x]), int(orig_g[y, x]), int(orig_b[y, x])
                m_r, m_g, m_b = int(manip_r[y, x]), int(manip_g[y, x]), int(manip_b[y, x])
                pixel_records.append({
                    'x': int(x),
                    'y': int(y),
                    'orig_r': o_r,
                    'orig_g': o_g,
                    'orig_b': o_b,
                    'orig_hex': f"{o_r:02x}{o_g:02x}{o_b:02x}",
                    'manip_r': m_r,
                    'manip_g': m_g,
                    'manip_b': m_b,
                    'manip_hex': f"{m_r:02x}{m_g:02x}{m_b:02x}",
                    'diff_r': m_r - o_r,
                    'diff_g': m_g - o_g,
                    'diff_b': m_b - o_b,
                    'is_manipulated': True
                })

        unchanged_y, unchanged_x = np.where(~chosen_mask_2d)
        max_unchanged = min(len(unchanged_y), max(200, 1200 - len(pixel_records)))
        if max_unchanged > 0:
            sampled_unchanged = np.random.choice(len(unchanged_y), size=max_unchanged, replace=False)
            for idx in sampled_unchanged:
                y, x = unchanged_y[idx], unchanged_x[idx]
                o_r, o_g, o_b = int(orig_r[y, x]), int(orig_g[y, x]), int(orig_b[y, x])
                pixel_records.append({
                    'x': int(x),
                    'y': int(y),
                    'orig_r': o_r,
                    'orig_g': o_g,
                    'orig_b': o_b,
                    'orig_hex': f"{o_r:02x}{o_g:02x}{o_b:02x}",
                    'manip_r': o_r,
                    'manip_g': o_g,
                    'manip_b': o_b,
                    'manip_hex': f"{o_r:02x}{o_g:02x}{o_b:02x}",
                    'diff_r': 0,
                    'diff_g': 0,
                    'diff_b': 0,
                    'is_manipulated': False
                })

        actual_pct = round((num_manipulated / total_pixels) * 100.0, 2)
        if roi:
            roi_area = roi['w'] * roi['h']
            roi_actual_pct = round((num_manipulated / roi_area) * 100.0, 2)
        else:
            roi_actual_pct = actual_pct

        return {
            'success': True,
            'is_grayscale': False,
            'width': width,
            'height': height,
            'total_pixels': total_pixels,
            'manipulated_count': num_manipulated,
            'requested_percentage': requested_percentage,
            'actual_percentage': actual_pct,
            'roi_actual_percentage': roi_actual_pct,
            'roi': roi,
            'method': method,
            'intensity': intensity,
            'mse': round(mse_val, 3),
            'mae': round(mae_val, 3),
            'psnr': round(psnr_val, 2),
            'channel_stats': channel_stats,
            'watermark_survival': watermark_survival,
            'extracted_text': extracted_text or "None detected",
            'watermark_intact': bool(extracted_text and "Capstone" in extracted_text),
            'manipulated_filename': os.path.basename(output_path),
            'heatmap_filename': os.path.basename(heatmap_path),
            'display_grid': display_grid,
            'grid_width': grid_w,
            'grid_height': grid_h,
            'pixel_records': pixel_records,
            'total_records_count': len(pixel_records),
            'custom_details': custom_details
        }

    def export_manipulation_excel(self, image_path, manipulated_path, excel_path, stats_payload):
        """
        Export a styled Excel spreadsheet containing RGB or Grayscale values
        before and after manipulation with full statistical summary.
        """
        try:
            app_logger.info(f"Generating Excel manipulation report: {excel_path}")
            wb = openpyxl.Workbook()

            title_font = Font(name="Arial", size=14, bold=True, color="1E3A8A")
            header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
            header_fill_blue = self._get_fill("3B82F6")
            header_fill_dark = self._get_fill("1E293B")
            bold_font = Font(name="Arial", size=10, bold=True)
            regular_font = Font(name="Arial", size=10)
            center_align = Alignment(horizontal="center", vertical="center")
            left_align = Alignment(horizontal="left", vertical="center")

            # SHEET 1: Summary & Statistics
            ws_summary = wb.active
            ws_summary.title = "Summary & Stats"
            ws_summary.views.sheetView[0].showGridLines = True

            ws_summary['A1'] = "PIXEL MANIPULATION & STATISTICAL REPORT"
            ws_summary['A1'].font = title_font

            roi_data = stats_payload.get('roi')
            roi_str = f"X: {roi_data['x1']}..{roi_data['x2']}, Y: {roi_data['y1']}..{roi_data['y2']} ({roi_data['w']}x{roi_data['h']} px)" if roi_data else "Full Image"

            general_info = [
                ("Source Image", os.path.basename(image_path)),
                ("Manipulated Image", os.path.basename(manipulated_path)),
                ("Image Dimensions", f"{stats_payload.get('width')} x {stats_payload.get('height')} px"),
                ("Total Pixels", f"{stats_payload.get('total_pixels'):,}"),
                ("Target Area / ROI", roi_str),
                ("Pixels Manipulated", f"{stats_payload.get('manipulated_count'):,} ({stats_payload.get('actual_percentage')}%)"),
                ("Manipulation Method / Op", str(stats_payload.get('method') or stats_payload.get('operation', '')).upper()),
                ("Mean Absolute Error (MAE)", stats_payload.get('mae')),
                ("Mean Squared Error (MSE)", stats_payload.get('mse')),
                ("Peak Signal-to-Noise Ratio (PSNR)", f"{stats_payload.get('psnr')} dB"),
                ("Watermark Extracted Text", stats_payload.get('extracted_text')),
                ("Watermark Status", "INTACT / DETECTED" if stats_payload.get('watermark_intact') else "TAMPERED / CORRUPTED")
            ]

            ws_summary['A3'] = "PARAMETER"
            ws_summary['B3'] = "VALUE"
            ws_summary['A3'].font = header_font
            ws_summary['A3'].fill = header_fill_blue
            ws_summary['B3'].font = header_font
            ws_summary['B3'].fill = header_fill_blue

            curr_row = 4
            for param, val in general_info:
                ws_summary[f'A{curr_row}'] = param
                ws_summary[f'B{curr_row}'] = str(val)
                ws_summary[f'A{curr_row}'].font = bold_font
                ws_summary[f'B{curr_row}'].font = regular_font
                curr_row += 1

            ws_summary.column_dimensions['A'].width = 32
            ws_summary.column_dimensions['B'].width = 45

            # SHEET 2: Pixel Sample Records
            records = stats_payload.get('pixel_records', [])
            if records:
                ws_records = wb.create_sheet("Pixel Sample Records")
                is_gray = stats_payload.get('is_grayscale', False)

                if is_gray:
                    headers = ["X", "Y", "Status", "Orig Gray", "Orig Hex", "Manip Gray", "Manip Hex", "Delta (ΔY)"]
                else:
                    headers = ["X", "Y", "Status", "Orig R", "Orig G", "Orig B", "Orig Hex", "Manip R", "Manip G", "Manip B", "Manip Hex", "ΔR", "ΔG", "ΔB"]

                for c_idx, h_text in enumerate(headers, start=1):
                    cell = ws_records.cell(row=1, column=c_idx, value=h_text)
                    cell.font = header_font
                    cell.fill = header_fill_dark
                    cell.alignment = center_align

                for r_idx, rec in enumerate(records[:1500], start=2):
                    if is_gray:
                        row_vals = [
                            rec['x'], rec['y'],
                            "MANIPULATED" if rec['is_manipulated'] else "UNCHANGED",
                            rec.get('orig_gray', 0), f"#{rec.get('orig_hex', '')}",
                            rec.get('manip_gray', 0), f"#{rec.get('manip_hex', '')}",
                            rec.get('diff_gray', 0)
                        ]
                    else:
                        row_vals = [
                            rec['x'], rec['y'],
                            "MANIPULATED" if rec['is_manipulated'] else "UNCHANGED",
                            rec.get('orig_r', 0), rec.get('orig_g', 0), rec.get('orig_b', 0), f"#{rec.get('orig_hex', '')}",
                            rec.get('manip_r', 0), rec.get('manip_g', 0), rec.get('manip_b', 0), f"#{rec.get('manip_hex', '')}",
                            rec.get('diff_r', 0), rec.get('diff_g', 0), rec.get('diff_b', 0)
                        ]

                    for c_idx, val in enumerate(row_vals, start=1):
                        cell = ws_records.cell(row=r_idx, column=c_idx, value=val)
                        cell.font = regular_font
                        cell.alignment = center_align

            wb.save(excel_path)
            app_logger.info(f"Excel report saved: {excel_path}")
            return {'success': True, 'excel_path': excel_path}

        except Exception as e:
            app_logger.error(f"Excel export error: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
