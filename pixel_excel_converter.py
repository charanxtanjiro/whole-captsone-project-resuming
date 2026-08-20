import cv2
import numpy as np
import openpyxl
from openpyxl.styles import PatternFill, Font, Alignment
from openpyxl.utils import get_column_letter
import logging

app_logger = logging.getLogger(__name__)


class PixelExcelConverter:
    """Convert images to/from Excel pixel format"""
    
    @staticmethod
    def image_to_excel(image_path, excel_path, sample_size=None):
        """
        Convert image to Excel file showing pixel RGB values
        
        Args:
            image_path: Path to image file
            excel_path: Path to save Excel file
            sample_size: (width, height) to downsample. If None, uses original size (max 100x100)
        """
        try:
            app_logger.info(f"Converting image to Excel: {image_path}")
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to read image")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img_rgb.shape[:2]
            
            app_logger.info(f"Original image size: {width}x{height}")
            
            # Downsample if needed (Excel has max 1,048,576 rows)
            if sample_size:
                target_w, target_h = sample_size
                img_rgb = cv2.resize(img_rgb, (target_w, target_h), interpolation=cv2.INTER_NEAREST)
                height, width = img_rgb.shape[:2]
                app_logger.info(f"Downsampled to: {width}x{height}")
            elif max(height, width) > 100:
                scale = 100 / max(height, width)
                new_w = int(width * scale)
                new_h = int(height * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
                height, width = img_rgb.shape[:2]
                app_logger.info(f"Auto-downsampled to: {width}x{height}")
            
            # Create workbook
            wb = openpyxl.Workbook()
            ws_rgb = wb.active
            ws_rgb.title = "Pixel RGB"
            
            # Add RGB pixel data
            ws_rgb['A1'] = "Pixel RGB Values (Row, Col)"
            ws_rgb['A1'].font = Font(bold=True, size=12)
            
            # Header row
            ws_rgb['A2'] = "Y"
            ws_rgb['B2'] = "X"
            ws_rgb['C2'] = "R"
            ws_rgb['D2'] = "G"
            ws_rgb['E2'] = "B"
            
            for col in ['A', 'B', 'C', 'D', 'E']:
                ws_rgb[f'{col}2'].font = Font(bold=True)
                ws_rgb[f'{col}2'].fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
            
            # Add pixel data
            row_num = 3
            for y in range(height):
                for x in range(width):
                    r, g, b = img_rgb[y, x]
                    
                    ws_rgb[f'A{row_num}'] = y
                    ws_rgb[f'B{row_num}'] = x
                    ws_rgb[f'C{row_num}'] = int(r)
                    ws_rgb[f'D{row_num}'] = int(g)
                    ws_rgb[f'E{row_num}'] = int(b)
                    
                    # Color the pixel cell based on RGB value
                    color_hex = f'{int(r):02x}{int(g):02x}{int(b):02x}'
                    ws_rgb[f'E{row_num}'].fill = PatternFill(start_color=color_hex, end_color=color_hex, fill_type="solid")
                    
                    row_num += 1
            
            # Set column widths
            ws_rgb.column_dimensions['A'].width = 8
            ws_rgb.column_dimensions['B'].width = 8
            ws_rgb.column_dimensions['C'].width = 8
            ws_rgb.column_dimensions['D'].width = 8
            ws_rgb.column_dimensions['E'].width = 25
            
            # Create visualization sheet with color blocks
            ws_viz = wb.create_sheet("Pixel Visualization")
            ws_viz['A1'] = "Visual Representation (Colors)"
            ws_viz['A1'].font = Font(bold=True, size=12)
            
            viz_row = 3
            for y in range(height):
                for x in range(width):
                    r, g, b = img_rgb[y, x]
                    color_hex = f'{int(r):02x}{int(g):02x}{int(b):02x}'
                    
                    cell = ws_viz.cell(row=viz_row + y, column=x + 1)
                    cell.fill = PatternFill(start_color=color_hex, end_color=color_hex, fill_type="solid")
                    cell.value = f"({int(r)},{int(g)},{int(b)})"
                    cell.font = Font(size=6, color="FFFFFF" if (int(r)+int(g)+int(b))/3 < 128 else "000000")
                    cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
            
            # Add metadata sheet
            ws_meta = wb.create_sheet("Metadata", 0)
            ws_meta['A1'] = "Image Metadata"
            ws_meta['A1'].font = Font(bold=True, size=12)
            ws_meta['A2'] = "Original Width"
            ws_meta['B2'] = width
            ws_meta['A3'] = "Original Height"
            ws_meta['B3'] = height
            ws_meta['A4'] = "Total Pixels"
            ws_meta['B4'] = width * height
            ws_meta['A5'] = "Has Watermark"
            ws_meta['B5'] = "Yes (LSB)"
            
            # Save workbook
            wb.save(excel_path)
            app_logger.info(f"Excel file saved: {excel_path}")
            
            return {
                "success": True,
                "width": width,
                "height": height,
                "total_pixels": width * height,
                "file_path": excel_path
            }
            
        except Exception as e:
            app_logger.error(f"Excel conversion error: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def excel_to_image(excel_path, output_image_path):
        """
        Convert Excel pixel data back to image
        
        Args:
            excel_path: Path to Excel file with pixel data
            output_image_path: Path to save reconstructed image
        """
        try:
            app_logger.info(f"Converting Excel to image: {excel_path}")
            
            # Load Excel
            wb = openpyxl.load_workbook(excel_path)
            ws = wb["Pixel RGB"]
            
            # Read metadata
            wb_meta = wb["Metadata"]
            width = wb_meta['B2'].value
            height = wb_meta['B3'].value
            
            app_logger.info(f"Reconstructing image: {width}x{height}")
            
            # Create image array
            img_rgb = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Read pixel data
            row_num = 3
            for y in range(height):
                for x in range(width):
                    r = ws[f'C{row_num}'].value or 0
                    g = ws[f'D{row_num}'].value or 0
                    b = ws[f'E{row_num}'].value or 0
                    
                    img_rgb[y, x] = [int(r), int(g), int(b)]
                    row_num += 1
            
            # Convert to BGR and save
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
            app_logger.error(f"Image reconstruction error: {e}")
            return {"success": False, "error": str(e)}
    
    @staticmethod
    def get_pixel_grid_json(image_path, max_width=50, max_height=50):
        """
        Get pixel data as JSON for web display
        
        Returns a compact grid of pixels
        """
        try:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width = img_rgb.shape[:2]
            
            # Downsample for web display
            if width > max_width or height > max_height:
                scale = min(max_width/width, max_height/height)
                new_w = int(width * scale)
                new_h = int(height * scale)
                img_rgb = cv2.resize(img_rgb, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
            
            height, width = img_rgb.shape[:2]
            
            # Create grid data
            grid = []
            for y in range(height):
                row = []
                for x in range(width):
                    r, g, b = img_rgb[y, x]
                    row.append({
                        "r": int(r),
                        "g": int(g),
                        "b": int(b),
                        "hex": f'{int(r):02x}{int(g):02x}{int(b):02x}'
                    })
                grid.append(row)
            
            return {
                "success": True,
                "width": width,
                "height": height,
                "grid": grid
            }
            
        except Exception as e:
            app_logger.error(f"Grid JSON error: {e}")
            return {"success": False, "error": str(e)}


# Test usage
if __name__ == "__main__":
    converter = PixelExcelConverter()
    
    # Example: Convert image to Excel
    result = converter.image_to_excel("test.jpg", "pixels.xlsx", sample_size=(30, 30))
    print("Image to Excel:", result)
    
    # Example: Convert Excel back to image
    result = converter.excel_to_image("pixels.xlsx", "reconstructed.jpg")
    print("Excel to Image:", result)
