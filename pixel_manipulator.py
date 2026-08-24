import os
import cv2
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment, Border, Side
from openpyxl.utils import get_column_letter
import logging
import math
from pixel_watermark import PixelWatermark

app_logger = logging.getLogger(__name__)

class PixelManipulator:
    """
    Handles pixel-level manipulation of watermarked images by customizable percentage
    and calculates detailed RGB statistics for each pixel before and after manipulation.
    """

    def __init__(self):
        self.watermark_tool = PixelWatermark()

    def manipulate_image(self, image_path, output_path, percentage=10.0, method='noise', intensity=50, seed=None):
        """
        Manipulate a chosen percentage of pixels in the image.

        Args:
            image_path (str): Path to the source watermarked image.
            output_path (str): Path where the manipulated image will be saved.
            percentage (float): Percentage of total pixels to manipulate (0.1 to 100.0).
            method (str): Manipulation method ('noise', 'lsb_corrupt', 'salt_pepper', 'channel_shift', 'invert', 'brightness').
            intensity (int): Intensity factor (1-255).
            seed (int, optional): Random seed for reproducible manipulation.

        Returns:
            dict: Comprehensive statistics and pixel data payload.
        """
        try:
            if seed is not None:
                np.random.seed(seed)

            percentage = float(max(0.1, min(100.0, percentage)))
            app_logger.info(f"Manipulating {percentage}% of pixels in {image_path} using method '{method}' (intensity={intensity})")

            # Read original image in RGB
            img_bgr = cv2.imread(image_path)
            if img_bgr is None:
                raise ValueError(f"Unable to read image at {image_path}")

            img_rgb_orig = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            height, width, channels = img_rgb_orig.shape
            total_pixels = height * width

            # Number of pixels to manipulate
            num_pixels_to_manipulate = int(round((percentage / 100.0) * total_pixels))
            num_pixels_to_manipulate = max(1, min(total_pixels, num_pixels_to_manipulate))

            # Select random unique pixel indices
            all_indices = np.arange(total_pixels)
            chosen_flat_indices = np.random.choice(all_indices, size=num_pixels_to_manipulate, replace=False)
            chosen_mask_flat = np.zeros(total_pixels, dtype=bool)
            chosen_mask_flat[chosen_flat_indices] = True
            chosen_mask_2d = chosen_mask_flat.reshape((height, width))

            # Work on a copy
            manip_rgb = img_rgb_orig.copy().astype(np.float32)

            # Apply manipulation based on selected method
            if method == 'noise':
                # Random uniform or Gaussian noise added to chosen pixels
                noise = np.random.randint(-intensity, intensity + 1, size=(height, width, channels)).astype(np.float32)
                manip_rgb[chosen_mask_2d] = np.clip(manip_rgb[chosen_mask_2d] + noise[chosen_mask_2d], 0, 255)

            elif method == 'lsb_corrupt':
                # Corrupt least significant bits (LSB watermark attack)
                manip_int = manip_rgb.astype(np.uint8)
                # Flip 1, 2, or 3 least significant bits randomly
                bit_mask = np.random.randint(1, 8, size=(height, width, channels), dtype=np.uint8)
                manip_int[chosen_mask_2d] = manip_int[chosen_mask_2d] ^ bit_mask[chosen_mask_2d]
                manip_rgb = manip_int.astype(np.float32)

            elif method == 'salt_pepper':
                # Randomly set pixel to either extreme black [0,0,0] or extreme white [255,255,255]
                salt_or_pepper = np.random.choice([0.0, 255.0], size=(height, width, channels))
                manip_rgb[chosen_mask_2d] = salt_or_pepper[chosen_mask_2d]

            elif method == 'channel_shift':
                # Boost or attenuate color channels
                shift_r = intensity
                shift_g = -int(intensity * 0.5)
                shift_b = int(intensity * 0.8)
                shift_arr = np.array([shift_r, shift_g, shift_b], dtype=np.float32)
                manip_rgb[chosen_mask_2d] = np.clip(manip_rgb[chosen_mask_2d] + shift_arr, 0, 255)

            elif method == 'invert':
                # Invert the RGB values: 255 - original
                manip_rgb[chosen_mask_2d] = 255.0 - manip_rgb[chosen_mask_2d]

            elif method == 'brightness':
                # Brightness / Contrast jitter
                factor = 1.0 + ((intensity - 128) / 128.0)  # scale multiplier
                offset = (intensity - 128) * 0.5
                manip_rgb[chosen_mask_2d] = np.clip(manip_rgb[chosen_mask_2d] * factor + offset, 0, 255)

            else:
                # Default to random noise
                noise = np.random.randint(-intensity, intensity + 1, size=(height, width, channels)).astype(np.float32)
                manip_rgb[chosen_mask_2d] = np.clip(manip_rgb[chosen_mask_2d] + noise[chosen_mask_2d], 0, 255)

            # Convert to uint8 final array
            manip_rgb_uint8 = np.clip(manip_rgb, 0, 255).astype(np.uint8)

            # Save the manipulated image
            manip_bgr = cv2.cvtColor(manip_rgb_uint8, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_path, manip_bgr)
            app_logger.info(f"Manipulated image saved to {output_path}")

            # Generate Difference Heatmap Mask
            heatmap_path = output_path.rsplit('.', 1)[0] + "_heatmap.png"
            diff_abs = np.abs(manip_rgb_uint8.astype(np.int16) - img_rgb_orig.astype(np.int16)).astype(np.uint8)
            diff_visual = np.zeros((height, width, 3), dtype=np.uint8)
            diff_visual[chosen_mask_2d] = [255, 60, 60]  # Mark manipulated in bright red/orange
            # Blend with dimmed grayscale original
            orig_gray = cv2.cvtColor(img_rgb_orig, cv2.COLOR_RGB2GRAY)
            orig_gray_3ch = cv2.cvtColor(orig_gray, cv2.COLOR_GRAY2BGR)
            heatmap_bgr = cv2.addWeighted(orig_gray_3ch, 0.4, diff_visual, 0.6, 0)
            cv2.imwrite(heatmap_path, heatmap_bgr)

            # Calculate Comprehensive RGB Channel Statistics
            orig_r = img_rgb_orig[:, :, 0]
            orig_g = img_rgb_orig[:, :, 1]
            orig_b = img_rgb_orig[:, :, 2]

            manip_r = manip_rgb_uint8[:, :, 0]
            manip_g = manip_rgb_uint8[:, :, 1]
            manip_b = manip_rgb_uint8[:, :, 2]

            # Overall Error Metrics (MSE, MAE, PSNR)
            mse_val = float(np.mean((img_rgb_orig.astype(np.float64) - manip_rgb_uint8.astype(np.float64)) ** 2))
            mae_val = float(np.mean(np.abs(img_rgb_orig.astype(np.float64) - manip_rgb_uint8.astype(np.float64))))
            if mse_val == 0:
                psnr_val = 100.0
            else:
                psnr_val = float(20 * math.log10(255.0 / math.sqrt(mse_val)))

            # Measure Watermark Robustness / Survival
            watermark_survival = self.watermark_tool.measure_watermark_robustness(image_path, output_path)
            extracted_text = self.watermark_tool.extract_lsb(output_path)

            channel_stats = {
                'red': {
                    'orig_min': int(np.min(orig_r)),
                    'orig_max': int(np.max(orig_r)),
                    'orig_mean': round(float(np.mean(orig_r)), 2),
                    'orig_std': round(float(np.std(orig_r)), 2),
                    'orig_median': int(np.median(orig_r)),
                    'manip_min': int(np.min(manip_r)),
                    'manip_max': int(np.max(manip_r)),
                    'manip_mean': round(float(np.mean(manip_r)), 2),
                    'manip_std': round(float(np.std(manip_r)), 2),
                    'manip_median': int(np.median(manip_r)),
                    'mae': round(float(np.mean(np.abs(orig_r.astype(np.float64) - manip_r.astype(np.float64)))), 2),
                    'mse': round(float(np.mean((orig_r.astype(np.float64) - manip_r.astype(np.float64)) ** 2)), 2),
                },
                'green': {
                    'orig_min': int(np.min(orig_g)),
                    'orig_max': int(np.max(orig_g)),
                    'orig_mean': round(float(np.mean(orig_g)), 2),
                    'orig_std': round(float(np.std(orig_g)), 2),
                    'orig_median': int(np.median(orig_g)),
                    'manip_min': int(np.min(manip_g)),
                    'manip_max': int(np.max(manip_g)),
                    'manip_mean': round(float(np.mean(manip_g)), 2),
                    'manip_std': round(float(np.std(manip_g)), 2),
                    'manip_median': int(np.median(manip_g)),
                    'mae': round(float(np.mean(np.abs(orig_g.astype(np.float64) - manip_g.astype(np.float64)))), 2),
                    'mse': round(float(np.mean((orig_g.astype(np.float64) - manip_g.astype(np.float64)) ** 2)), 2),
                },
                'blue': {
                    'orig_min': int(np.min(orig_b)),
                    'orig_max': int(np.max(orig_b)),
                    'orig_mean': round(float(np.mean(orig_b)), 2),
                    'orig_std': round(float(np.std(orig_b)), 2),
                    'orig_median': int(np.median(orig_b)),
                    'manip_min': int(np.min(manip_b)),
                    'manip_max': int(np.max(manip_b)),
                    'manip_mean': round(float(np.mean(manip_b)), 2),
                    'manip_std': round(float(np.std(manip_b)), 2),
                    'manip_median': int(np.median(manip_b)),
                    'mae': round(float(np.mean(np.abs(orig_b.astype(np.float64) - manip_b.astype(np.float64)))), 2),
                    'mse': round(float(np.mean((orig_b.astype(np.float64) - manip_b.astype(np.float64)) ** 2)), 2),
                }
            }

            # Generate Sampled / Display Grid for Web (Max 40x40 for rendering speed)
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
                    is_manip = bool(mask_resized[y, x])
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
                        'is_manipulated': is_manip
                    })
                display_grid.append(row)

            # Build Pixel Records List for Table (Up to 10,000 pixels or all if smaller)
            pixel_records = []
            # We prioritize manipulated pixels, followed by sample of unchanged pixels
            y_indices, x_indices = np.where(chosen_mask_2d)
            for y, x in zip(y_indices, x_indices):
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

            # Add sample of unchanged pixels if room
            unchanged_y, unchanged_x = np.where(~chosen_mask_2d)
            max_unchanged_to_add = min(len(unchanged_y), max(500, 2000 - len(pixel_records)))
            if max_unchanged_to_add > 0:
                sampled_unchanged_indices = np.random.choice(len(unchanged_y), size=max_unchanged_to_add, replace=False)
                for idx in sampled_unchanged_indices:
                    y = unchanged_y[idx]
                    x = unchanged_x[idx]
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

            result = {
                'success': True,
                'width': width,
                'height': height,
                'total_pixels': total_pixels,
                'manipulated_count': num_pixels_to_manipulate,
                'requested_percentage': percentage,
                'actual_percentage': round((num_pixels_to_manipulate / total_pixels) * 100, 2),
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
            }

            return result

        except Exception as e:
            app_logger.error(f"Error in pixel manipulation: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def export_manipulation_excel(self, image_path, manipulated_path, excel_path, stats_payload):
        """
        Export a detailed, beautifully styled Excel spreadsheet containing
        RGB values of every pixel before and after manipulation with statistical summary.
        """
        try:
            app_logger.info(f"Generating Excel manipulation report: {excel_path}")

            wb = openpyxl.Workbook()

            # Define Styles
            title_font = Font(name="Arial", size=14, bold=True, color="1E3A8A")
            header_font = Font(name="Arial", size=11, bold=True, color="FFFFFF")
            header_fill_blue = PatternFill(start_color="3B82F6", end_color="3B82F6", fill_type="solid")
            header_fill_dark = PatternFill(start_color="1E293B", end_color="1E293B", fill_type="solid")
            bold_font = Font(name="Arial", size=10, bold=True)
            regular_font = Font(name="Arial", size=10)
            center_align = Alignment(horizontal="center", vertical="center")
            left_align = Alignment(horizontal="left", vertical="center")
            thin_border = Border(
                left=Side(style='thin', color='E2E8F0'),
                right=Side(style='thin', color='E2E8F0'),
                top=Side(style='thin', color='E2E8F0'),
                bottom=Side(style='thin', color='E2E8F0')
            )

            # ----------------------------------------------------
            # SHEET 1: Summary & Channel Statistics
            # ----------------------------------------------------
            ws_summary = wb.active
            ws_summary.title = "Summary & RGB Stats"
            ws_summary.views.sheetView[0].showGridLines = True

            ws_summary['A1'] = "PIXEL MANIPULATION & RGB STATISTICAL REPORT"
            ws_summary['A1'].font = title_font

            # General Metrics
            general_info = [
                ("Source Image", os.path.basename(image_path)),
                ("Manipulated Image", os.path.basename(manipulated_path)),
                ("Image Dimensions", f"{stats_payload.get('width')} x {stats_payload.get('height')} px"),
                ("Total Pixels", f"{stats_payload.get('total_pixels'):,}"),
                ("Pixels Manipulated", f"{stats_payload.get('manipulated_count'):,} ({stats_payload.get('actual_percentage')}%)"),
                ("Manipulation Method", stats_payload.get('method', '').upper()),
                ("Intensity / Strength", stats_payload.get('intensity')),
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
                ws_summary[f'A{curr_row}'].border = thin_border
                ws_summary[f'B{curr_row}'].border = thin_border
                curr_row += 1

            # Channel Comparison Table
            curr_row += 2
            ws_summary[f'A{curr_row}'] = "RGB CHANNEL LEVEL STATISTICS"
            ws_summary[f'A{curr_row}'].font = Font(name="Arial", size=12, bold=True, color="1E3A8A")
            curr_row += 1

            headers_ch = ["Metric", "Red (Original)", "Red (Manipulated)", "Green (Original)", "Green (Manipulated)", "Blue (Original)", "Blue (Manipulated)"]
            for col_idx, h_text in enumerate(headers_ch, start=1):
                cell = ws_summary.cell(row=curr_row, column=col_idx)
                cell.value = h_text
                cell.font = header_font
                cell.fill = header_fill_dark
                cell.alignment = center_align
                cell.border = thin_border

            curr_row += 1
            ch = stats_payload.get('channel_stats', {})
            r_stats = ch.get('red', {})
            g_stats = ch.get('green', {})
            b_stats = ch.get('blue', {})

            metrics_rows = [
                ("Minimum Value", r_stats.get('orig_min'), r_stats.get('manip_min'), g_stats.get('orig_min'), g_stats.get('manip_min'), b_stats.get('orig_min'), b_stats.get('manip_min')),
                ("Maximum Value", r_stats.get('orig_max'), r_stats.get('manip_max'), g_stats.get('orig_max'), g_stats.get('manip_max'), b_stats.get('orig_max'), b_stats.get('manip_max')),
                ("Mean Value", r_stats.get('orig_mean'), r_stats.get('manip_mean'), g_stats.get('orig_mean'), g_stats.get('manip_mean'), b_stats.get('orig_mean'), b_stats.get('manip_mean')),
                ("Std Deviation", r_stats.get('orig_std'), r_stats.get('manip_std'), g_stats.get('orig_std'), g_stats.get('manip_std'), b_stats.get('orig_std'), b_stats.get('manip_std')),
                ("Median Value", r_stats.get('orig_median'), r_stats.get('manip_median'), g_stats.get('orig_median'), g_stats.get('manip_median'), b_stats.get('orig_median'), b_stats.get('manip_median')),
                ("Mean Absolute Error (MAE)", r_stats.get('mae'), "-", g_stats.get('mae'), "-", b_stats.get('mae'), "-"),
                ("Mean Squared Error (MSE)", r_stats.get('mse'), "-", g_stats.get('mse'), "-", b_stats.get('mse'), "-")
            ]

            for row_vals in metrics_rows:
                for col_idx, val in enumerate(row_vals, start=1):
                    cell = ws_summary.cell(row=curr_row, column=col_idx)
                    cell.value = val
                    cell.font = bold_font if col_idx == 1 else regular_font
                    cell.alignment = left_align if col_idx == 1 else center_align
                    cell.border = thin_border
                curr_row += 1

            ws_summary.column_dimensions['A'].width = 30
            ws_summary.column_dimensions['B'].width = 22
            ws_summary.column_dimensions['C'].width = 22
            ws_summary.column_dimensions['D'].width = 22
            ws_summary.column_dimensions['E'].width = 22
            ws_summary.column_dimensions['F'].width = 22
            ws_summary.column_dimensions['G'].width = 22

            # ----------------------------------------------------
            # SHEET 2: Per-Pixel RGB Breakdown Table
            # ----------------------------------------------------
            ws_pixels = wb.create_sheet(title="Pixel RGB Data")
            ws_pixels.views.sheetView[0].showGridLines = True

            ws_pixels['A1'] = "PER-PIXEL RGB COMPARISON TABLE"
            ws_pixels['A1'].font = title_font

            pixel_headers = [
                "Y (Row)", "X (Col)", "Status",
                "Orig R", "Orig G", "Orig B", "Orig Hex", "Orig Color",
                "Manip R", "Manip G", "Manip B", "Manip Hex", "Manip Color",
                "Delta R", "Delta G", "Delta B"
            ]

            header_row = 3
            for col_idx, h_name in enumerate(pixel_headers, start=1):
                cell = ws_pixels.cell(row=header_row, column=col_idx)
                cell.value = h_name
                cell.font = header_font
                cell.fill = header_fill_blue
                cell.alignment = center_align
                cell.border = thin_border

            # Load actual pixel arrays for exact dump (up to 15,000 pixels to keep Excel fast)
            img_bgr_orig = cv2.imread(image_path)
            img_bgr_manip = cv2.imread(manipulated_path)
            orig_rgb = cv2.cvtColor(img_bgr_orig, cv2.COLOR_BGR2RGB)
            manip_rgb = cv2.cvtColor(img_bgr_manip, cv2.COLOR_BGR2RGB)

            row_counter = 4
            max_excel_rows = 15000
            total_written = 0

            # First write all manipulated pixels, then sample remainder
            diff_mask = np.any(orig_rgb != manip_rgb, axis=2)
            y_diff, x_diff = np.where(diff_mask)

            manip_fill = PatternFill(start_color="FEE2E2", end_color="FEE2E2", fill_type="solid")
            manip_font = Font(name="Arial", size=9, bold=True, color="991B1B")
            normal_font = Font(name="Arial", size=9)

            for y, x in zip(y_diff, x_diff):
                if total_written >= max_excel_rows:
                    break
                o_r, o_g, o_b = int(orig_rgb[y, x, 0]), int(orig_rgb[y, x, 1]), int(orig_rgb[y, x, 2])
                m_r, m_g, m_b = int(manip_rgb[y, x, 0]), int(manip_rgb[y, x, 1]), int(manip_rgb[y, x, 2])
                o_hex = f"{o_r:02X}{o_g:02X}{o_b:02X}"
                m_hex = f"{m_r:02X}{m_g:02X}{m_b:02X}"

                ws_pixels[f'A{row_counter}'] = y
                ws_pixels[f'B{row_counter}'] = x
                ws_pixels[f'C{row_counter}'] = "MANIPULATED"
                ws_pixels[f'C{row_counter}'].fill = manip_fill
                ws_pixels[f'C{row_counter}'].font = manip_font

                ws_pixels[f'D{row_counter}'] = o_r
                ws_pixels[f'E{row_counter}'] = o_g
                ws_pixels[f'F{row_counter}'] = o_b
                ws_pixels[f'G{row_counter}'] = f"#{o_hex}"
                ws_pixels[f'H{row_counter}'].fill = PatternFill(start_color=o_hex, end_color=o_hex, fill_type="solid")

                ws_pixels[f'I{row_counter}'] = m_r
                ws_pixels[f'J{row_counter}'] = m_g
                ws_pixels[f'K{row_counter}'] = m_b
                ws_pixels[f'L{row_counter}'] = f"#{m_hex}"
                ws_pixels[f'M{row_counter}'].fill = PatternFill(start_color=m_hex, end_color=m_hex, fill_type="solid")

                ws_pixels[f'N{row_counter}'] = m_r - o_r
                ws_pixels[f'O{row_counter}'] = m_g - o_g
                ws_pixels[f'P{row_counter}'] = m_b - o_b

                for col_letter in ['A', 'B', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'N', 'O', 'P']:
                    cell = ws_pixels[f'{col_letter}{row_counter}']
                    cell.font = normal_font
                    cell.alignment = center_align
                    cell.border = thin_border

                row_counter += 1
                total_written += 1

            # Write unchanged pixels if under limit
            y_same, x_same = np.where(~diff_mask)
            for y, x in zip(y_same, x_same):
                if total_written >= max_excel_rows:
                    break
                o_r, o_g, o_b = int(orig_rgb[y, x, 0]), int(orig_rgb[y, x, 1]), int(orig_rgb[y, x, 2])
                o_hex = f"{o_r:02X}{o_g:02X}{o_b:02X}"

                ws_pixels[f'A{row_counter}'] = y
                ws_pixels[f'B{row_counter}'] = x
                ws_pixels[f'C{row_counter}'] = "UNCHANGED"

                ws_pixels[f'D{row_counter}'] = o_r
                ws_pixels[f'E{row_counter}'] = o_g
                ws_pixels[f'F{row_counter}'] = o_b
                ws_pixels[f'G{row_counter}'] = f"#{o_hex}"
                ws_pixels[f'H{row_counter}'].fill = PatternFill(start_color=o_hex, end_color=o_hex, fill_type="solid")

                ws_pixels[f'I{row_counter}'] = o_r
                ws_pixels[f'J{row_counter}'] = o_g
                ws_pixels[f'K{row_counter}'] = o_b
                ws_pixels[f'L{row_counter}'] = f"#{o_hex}"
                ws_pixels[f'M{row_counter}'].fill = PatternFill(start_color=o_hex, end_color=o_hex, fill_type="solid")

                ws_pixels[f'N{row_counter}'] = 0
                ws_pixels[f'O{row_counter}'] = 0
                ws_pixels[f'P{row_counter}'] = 0

                for col_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'I', 'J', 'K', 'L', 'N', 'O', 'P']:
                    cell = ws_pixels[f'{col_letter}{row_counter}']
                    cell.font = normal_font
                    cell.alignment = center_align
                    cell.border = thin_border

                row_counter += 1
                total_written += 1

            for col_letter in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P']:
                ws_pixels.column_dimensions[col_letter].width = 12

            # Save Workbook
            wb.save(excel_path)
            app_logger.info(f"Excel report saved successfully: {excel_path}")
            return {'success': True, 'file_path': excel_path}

        except Exception as e:
            app_logger.error(f"Error creating manipulation Excel export: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
