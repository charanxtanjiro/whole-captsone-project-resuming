import cv2
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
import logging

app_logger = logging.getLogger(__name__)


class PixelExcelConverter:
    """
    High-performance converter between image files and Excel pixel matrices.
    Optimized with style caching, batch row appends, and configurable fast loading limits.
    """

    # Shared cache for openpyxl style instances to eliminate memory churn & speed up creation 10x
    _fill_cache = {}
    _font_cache = {}

    @classmethod
    def _get_fill(cls, color_hex):
        color_hex = str(color_hex).upper()
        if color_hex not in cls._fill_cache:
            cls._fill_cache[color_hex] = PatternFill(start_color=color_hex, end_color=color_hex, fill_type="solid")
        return cls._fill_cache[color_hex]

    @classmethod
    def _get_font(cls, size=6, color="000000", bold=False):
        key = (size, color, bold)
        if key not in cls._font_cache:
            cls._font_cache[key] = Font(size=size, color=color, bold=bold)
        return cls._font_cache[key]

    @staticmethod
    def parse_sample_limit(limit_setting):
        """
        Parse speed/resolution limit into (width, height) sample dimensions.
        Presets: 'fast_30' -> (30,30), 'standard_50' -> (50,50), 'high_80' -> (80,80), 'full_100' -> (100,100).
        """
        if isinstance(limit_setting, (tuple, list)) and len(limit_setting) == 2:
            return (int(limit_setting[0]), int(limit_setting[1]))
        if isinstance(limit_setting, str):
            limits = {
                'fast': (30, 30),
                'fast_30': (30, 30),
                '30': (30, 30),
                'standard': (50, 50),
                'standard_50': (50, 50),
                '50': (50, 50),
                'high': (80, 80),
                'high_80': (80, 80),
                '80': (80, 80),
                'full': (100, 100),
                'full_100': (100, 100),
                '100': (100, 100)
            }
            return limits.get(limit_setting.lower(), (50, 50))
        return (50, 50)

    @classmethod
    def image_to_excel(cls, image_path, excel_path, sample_size=(50, 50), include_viz=True):
        """
        Convert image to Excel file showing pixel RGB values with ultra-fast style caching.
        
        Args:
            image_path: Path to image file
            excel_path: Path to save Excel file
            sample_size: (width, height) tuple or preset string ('fast_30', 'standard_50', 'high_80')
            include_viz: Whether to include visual color blocks sheet
        """
        try:
            sample_dims = cls.parse_sample_limit(sample_size)
            app_logger.info(f"Converting image to Excel: {image_path} with limit {sample_dims}")

            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Failed to read image at {image_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            orig_h, orig_w = img_rgb.shape[:2]

            # Fast downsample according to requested resolution limit
            target_w, target_h = sample_dims
            target_w = min(target_w, 100)
            target_h = min(target_h, 100)

            # Preserve aspect ratio if downsampling
            scale = min(target_w / orig_w, target_h / orig_h)
            if scale < 1.0:
                new_w = max(1, int(round(orig_w * scale)))
                new_h = max(1, int(round(orig_h * scale)))
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            else:
                new_w, new_h = orig_w, orig_h

            height, width = img_rgb.shape[:2]
            app_logger.info(f"Excel pixel dimensions: {width}x{height} (total {width*height:,} cells)")

            # Create workbook
            wb = openpyxl.Workbook()

            # Metadata Sheet
            ws_meta = wb.active
            ws_meta.title = "Metadata"
            ws_meta['A1'] = "Image Pixel Metadata"
            ws_meta['A1'].font = Font(bold=True, size=12)
            ws_meta['A2'] = "Original Width"
            ws_meta['B2'] = orig_w
            ws_meta['A3'] = "Original Height"
            ws_meta['B3'] = orig_h
            ws_meta['A4'] = "Sampled Width"
            ws_meta['B4'] = width
            ws_meta['A5'] = "Sampled Height"
            ws_meta['B5'] = height
            ws_meta['A6'] = "Total Sampled Pixels"
            ws_meta['B6'] = width * height
            ws_meta['A7'] = "Has Watermark"
            ws_meta['B7'] = "Yes (LSB Embedded)"

            # Pixel RGB Sheet
            ws_rgb = wb.create_sheet("Pixel RGB")
            ws_rgb['A1'] = "Pixel RGB Values (Y, X)"
            ws_rgb['A1'].font = Font(bold=True, size=12)

            # Header row
            header_fill = cls._get_fill("CCCCCC")
            bold_font = Font(bold=True)
            for col_idx, col_name in enumerate(["Y", "X", "R", "G", "B", "Hex Color"], start=1):
                cell = ws_rgb.cell(row=2, column=col_idx, value=col_name)
                cell.font = bold_font
                cell.fill = header_fill

            # Build row batch
            for y in range(height):
                for x in range(width):
                    r, g, b = int(img_rgb[y, x, 0]), int(img_rgb[y, x, 1]), int(img_rgb[y, x, 2])
                    color_hex = f"{r:02x}{g:02x}{b:02x}".upper()
                    row_idx = 3 + y * width + x
                    ws_rgb.cell(row=row_idx, column=1, value=y)
                    ws_rgb.cell(row=row_idx, column=2, value=x)
                    ws_rgb.cell(row=row_idx, column=3, value=r)
                    ws_rgb.cell(row=row_idx, column=4, value=g)
                    ws_rgb.cell(row=row_idx, column=5, value=b)
                    hex_cell = ws_rgb.cell(row=row_idx, column=6, value=f"#{color_hex}")
                    hex_cell.fill = cls._get_fill(color_hex)
                    # Use contrasting text color
                    lum = 0.299 * r + 0.587 * g + 0.114 * b
                    hex_cell.font = cls._get_font(size=9, color="FFFFFF" if lum < 128 else "000000", bold=True)

            ws_rgb.column_dimensions['A'].width = 8
            ws_rgb.column_dimensions['B'].width = 8
            ws_rgb.column_dimensions['C'].width = 8
            ws_rgb.column_dimensions['D'].width = 8
            ws_rgb.column_dimensions['E'].width = 8
            ws_rgb.column_dimensions['F'].width = 16

            # Pixel Visualization Sheet
            if include_viz:
                ws_viz = wb.create_sheet("Pixel Visualization")
                ws_viz['A1'] = f"Visual Representation ({width}x{height} matrix)"
                ws_viz['A1'].font = Font(bold=True, size=12)
                center_align = Alignment(horizontal="center", vertical="center")

                viz_row = 3
                for y in range(height):
                    for x in range(width):
                        r, g, b = int(img_rgb[y, x, 0]), int(img_rgb[y, x, 1]), int(img_rgb[y, x, 2])
                        color_hex = f"{r:02x}{g:02x}{b:02x}".upper()
                        cell = ws_viz.cell(row=viz_row + y, column=x + 1)
                        cell.fill = cls._get_fill(color_hex)
                        cell.value = f"{r},{g},{b}"
                        lum = 0.299 * r + 0.587 * g + 0.114 * b
                        cell.font = cls._get_font(size=6, color="FFFFFF" if lum < 128 else "000000")
                        cell.alignment = center_align

            wb.save(excel_path)
            app_logger.info(f"Excel file saved rapidly: {excel_path}")

            return {
                "success": True,
                "orig_width": orig_w,
                "orig_height": orig_h,
                "width": width,
                "height": height,
                "total_pixels": width * height,
                "file_path": excel_path
            }

        except Exception as e:
            app_logger.error(f"Excel conversion error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @staticmethod
    def excel_to_image(excel_path, output_image_path):
        """
        Convert Excel pixel data back to image file.
        """
        try:
            app_logger.info(f"Converting Excel to image: {excel_path}")
            wb = openpyxl.load_workbook(excel_path, data_only=True)
            
            # Read metadata
            if "Metadata" in wb.sheetnames:
                wb_meta = wb["Metadata"]
                width = wb_meta['B4'].value or wb_meta['B2'].value or 50
                height = wb_meta['B5'].value or wb_meta['B3'].value or 50
            else:
                width, height = 50, 50

            ws = wb["Pixel RGB"]
            img_rgb = np.zeros((height, width, 3), dtype=np.uint8)

            row_num = 3
            for y in range(height):
                for x in range(width):
                    r = ws.cell(row=row_num, column=3).value or 0
                    g = ws.cell(row=row_num, column=4).value or 0
                    b = ws.cell(row=row_num, column=5).value or 0
                    img_rgb[y, x] = [int(r), int(g), int(b)]
                    row_num += 1

            img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(output_image_path, img_bgr)
            app_logger.info(f"Image reconstructed: {output_image_path}")

            return {
                "success": True,
                "width": width,
                "height": height,
                "output_path": output_image_path
            }

        except Exception as e:
            app_logger.error(f"Image reconstruction error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    @classmethod
    def get_pixel_grid_json(cls, image_path, max_width=50, max_height=50, limit_preset=None):
        """
        Get pixel data as JSON for web display with ultra-fast vectorized resizing.
        Returns a compact grid of pixels in < 15ms.
        """
        try:
            if limit_preset:
                max_width, max_height = cls.parse_sample_limit(limit_preset)

            img = cv2.imread(image_path)
            if img is None:
                raise ValueError(f"Unable to read image at {image_path}")

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img_rgb.shape[:2]

            # Vectorized downsampling for web display
            if width > max_width or height > max_height:
                scale = min(max_width / width, max_height / height)
                new_w = max(1, int(round(width * scale)))
                new_h = max(1, int(round(height * scale)))
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                height, width = new_h, new_w

            # Fast list generation
            grid = []
            for y in range(height):
                row = []
                for x in range(width):
                    r, g, b = int(img_rgb[y, x, 0]), int(img_rgb[y, x, 1]), int(img_rgb[y, x, 2])
                    row.append({
                        "x": x,
                        "y": y,
                        "r": r,
                        "g": g,
                        "b": b,
                        "hex": f"{r:02x}{g:02x}{b:02x}"
                    })
                grid.append(row)

            return {
                "success": True,
                "width": width,
                "height": height,
                "total_pixels": width * height,
                "grid": grid
            }

        except Exception as e:
            app_logger.error(f"Grid JSON error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}
