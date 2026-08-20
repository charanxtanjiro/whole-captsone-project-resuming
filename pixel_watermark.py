import cv2
import numpy as np
import logging
from PIL import Image

app_logger = logging.getLogger(__name__)

class PixelWatermark:
    """Pixel-level invisible watermark embedding"""
    
    def __init__(self, strength=5):
        """
        strength: how much to modify pixels (1-10, higher = more robust but more visible)
        """
        self.strength = strength
    
    def text_to_binary(self, text):
        """Convert text to binary string"""
        return ''.join(format(ord(char), '08b') for char in text)
    
    def embed_in_lsb(self, image_path, watermark_text="Capstone"):
        """
        Least Significant Bit (LSB) embedding - most invisible method
        Modifies only the least significant bits of pixels
        """
        try:
            app_logger.info(f"Embedding LSB watermark in {image_path}")
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to read image")
            
            app_logger.info(f"Original image shape: {img.shape}")
            
            # Convert BGR to RGB for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Get image dimensions
            height, width, channels = img_rgb.shape
            max_bits = height * width * channels
            
            # Convert watermark text to binary
            watermark_binary = self.text_to_binary(watermark_text)
            app_logger.info(f"Watermark binary length: {len(watermark_binary)} bits")
            
            if len(watermark_binary) > max_bits:
                raise Exception(f"Watermark too large. Max bits: {max_bits}, Required: {len(watermark_binary)}")
            
            # Flatten image to 1D array for easier bit manipulation
            flat_img = img_rgb.flatten()
            
            # Embed watermark bits into LSB of pixel values
            bit_index = 0
            for pixel_index in range(len(flat_img)):
                if bit_index < len(watermark_binary):
                    bit = int(watermark_binary[bit_index])
                    # Clear LSB and set new bit
                    flat_img[pixel_index] = (flat_img[pixel_index] & 0xFE) | bit
                    bit_index += 1
            
            # Reshape back to 2D
            watermarked_img = flat_img.reshape((height, width, channels))
            
            # Convert back to BGR for saving
            watermarked_bgr = cv2.cvtColor(watermarked_img.astype(np.uint8), cv2.COLOR_RGB2BGR)
            
            # Save watermarked image
            cv2.imwrite(image_path, watermarked_bgr)
            app_logger.info(f"LSB watermark embedded successfully. Embedded {bit_index} bits")
            
            return True
            
        except Exception as e:
            app_logger.error(f"LSB watermark error: {e}")
            return False
    
    def embed_in_dct(self, image_path, watermark_text="Capstone"):
        """
        DCT (Discrete Cosine Transform) embedding - more robust
        Embeds in frequency domain, survives compression
        """
        try:
            app_logger.info(f"Embedding DCT watermark in {image_path}")
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to read image")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Convert to YUV (Y is luminance, important for perception)
            img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
            y_channel = img_yuv[:, :, 0].astype(np.float32)
            
            # Apply DCT to Y channel
            dct = cv2.dct(y_channel)
            app_logger.info(f"DCT shape: {dct.shape}")
            
            # Convert watermark to binary
            watermark_binary = self.text_to_binary(watermark_text)
            
            # Embed in middle-frequency components (more robust, less visible)
            bit_index = 0
            for i in range(10, min(dct.shape[0]-10, 50)):
                for j in range(10, min(dct.shape[1]-10, 50)):
                    if bit_index < len(watermark_binary):
                        bit = int(watermark_binary[bit_index])
                        # Modify DCT coefficient based on bit
                        dct[i, j] = dct[i, j] + (bit * self.strength)
                        bit_index += 1
            
            # Inverse DCT
            y_watermarked = cv2.idct(dct)
            y_watermarked = np.uint8(np.clip(y_watermarked, 0, 255))
            
            # Put watermarked Y back into YUV
            img_yuv[:, :, 0] = y_watermarked
            
            # Convert back to RGB then BGR
            result_rgb = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
            result_bgr = cv2.cvtColor(result_rgb, cv2.COLOR_RGB2BGR)
            
            # Save
            cv2.imwrite(image_path, result_bgr)
            app_logger.info(f"DCT watermark embedded successfully. Embedded {bit_index} bits")
            
            return True
            
        except Exception as e:
            app_logger.error(f"DCT watermark error: {e}")
            return False
    
    def embed_in_rgb_channels(self, image_path, watermark_text="Capstone"):
        """
        RGB Channel embedding - embeds watermark differently in R, G, B channels
        More redundant, very robust
        """
        try:
            app_logger.info(f"Embedding RGB channel watermark in {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to read image")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            height, width, channels = img_rgb.shape
            
            watermark_binary = self.text_to_binary(watermark_text)
            
            # Split into channels
            r_channel = img_rgb[:, :, 0].flatten()
            g_channel = img_rgb[:, :, 1].flatten()
            b_channel = img_rgb[:, :, 2].flatten()
            
            # Embed same watermark in each channel at different positions
            bit_index = 0
            for pixel_idx in range(len(r_channel)):
                if bit_index < len(watermark_binary):
                    bit = int(watermark_binary[bit_index])
                    
                    # Embed in R channel (bits 0-6)
                    r_channel[pixel_idx] = (r_channel[pixel_idx] & 0xFE) | bit
                    
                    # Embed in G channel (bits 7-13)
                    if bit_index + 1 < len(watermark_binary):
                        bit2 = int(watermark_binary[bit_index + 1])
                        g_channel[pixel_idx] = (g_channel[pixel_idx] & 0xFE) | bit2
                    
                    # Embed in B channel (bits 14-20)
                    if bit_index + 2 < len(watermark_binary):
                        bit3 = int(watermark_binary[bit_index + 2])
                        b_channel[pixel_idx] = (b_channel[pixel_idx] & 0xFE) | bit3
                    
                    bit_index += 3
            
            # Reshape and recombine
            r_channel = r_channel.reshape((height, width))
            g_channel = g_channel.reshape((height, width))
            b_channel = b_channel.reshape((height, width))
            
            watermarked_rgb = np.dstack([r_channel, g_channel, b_channel]).astype(np.uint8)
            watermarked_bgr = cv2.cvtColor(watermarked_rgb, cv2.COLOR_RGB2BGR)
            
            cv2.imwrite(image_path, watermarked_bgr)
            app_logger.info(f"RGB channel watermark embedded successfully. Embedded {bit_index} bits")
            
            return True
            
        except Exception as e:
            app_logger.error(f"RGB watermark error: {e}")
            return False
    
    def embed_hybrid(self, image_path, watermark_text="Capstone"):
        """
        Hybrid approach: Combines LSB + DCT for maximum robustness
        Very resistant to attacks
        """
        try:
            app_logger.info(f"Embedding Hybrid watermark in {image_path}")
            
            # First apply LSB
            self.embed_in_lsb(image_path, watermark_text)
            
            # Then apply DCT on top
            self.embed_in_dct(image_path, watermark_text)
            
            app_logger.info("Hybrid watermark embedded successfully")
            return True
            
        except Exception as e:
            app_logger.error(f"Hybrid watermark error: {e}")
            return False
    
    def extract_lsb(self, image_path):
        """
        Extract watermark from LSB
        Recovers the hidden text watermark
        """
        try:
            app_logger.info(f"Extracting LSB watermark from {image_path}")
            
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to read image")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Flatten image
            flat_img = img_rgb.flatten()
            
            # Extract LSB bits
            extracted_binary = ""
            for pixel_value in flat_img:
                extracted_binary += str(pixel_value & 0x01)
            
            # Convert binary to text (assuming 8-bit ASCII)
            extracted_text = ""
            for i in range(0, len(extracted_binary) - 8, 8):
                byte = extracted_binary[i:i+8]
                char_code = int(byte, 2)
                # Only accept printable ASCII characters
                if 32 <= char_code <= 126:
                    extracted_text += chr(char_code)
                else:
                    # Stop if we hit non-printable character (likely end of watermark)
                    break
            
            app_logger.info(f"Extracted watermark: {extracted_text}")
            return extracted_text if extracted_text else None
            
        except Exception as e:
            app_logger.error(f"LSB extraction error: {e}")
            return None
    
    def extract_dct(self, image_path):
        """
        Extract watermark from DCT domain
        """
        try:
            app_logger.info(f"Extracting DCT watermark from {image_path}")
            
            img = cv2.imread(image_path)
            if img is None:
                raise Exception("Failed to read image")
            
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_yuv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YUV)
            y_channel = img_yuv[:, :, 0].astype(np.float32)
            
            # Apply DCT
            dct = cv2.dct(y_channel)
            
            # Extract bits from middle-frequency components
            extracted_binary = ""
            for i in range(10, min(dct.shape[0]-10, 50)):
                for j in range(10, min(dct.shape[1]-10, 50)):
                    extracted_binary += "1" if dct[i, j] > 0 else "0"
            
            # Convert to text
            extracted_text = ""
            for i in range(0, len(extracted_binary) - 8, 8):
                byte = extracted_binary[i:i+8]
                char_code = int(byte, 2)
                if 32 <= char_code <= 126:
                    extracted_text += chr(char_code)
                else:
                    break
            
            app_logger.info(f"Extracted DCT watermark: {extracted_text}")
            return extracted_text if extracted_text else None
            
        except Exception as e:
            app_logger.error(f"DCT extraction error: {e}")
            return None
    
    def measure_watermark_robustness(self, original_path, attacked_path):
        """
        Measure how robust the watermark is after attacks
        Returns percentage of watermark still detectable
        """
        try:
            app_logger.info(f"Measuring watermark robustness")
            
            # Extract from both images
            original_wm = self.extract_lsb(original_path)
            attacked_wm = self.extract_lsb(attacked_path)
            
            if not original_wm:
                return {"robustness": 0, "status": "No watermark in original"}
            
            if not attacked_wm:
                return {"robustness": 0, "status": "Watermark completely removed by attack"}
            
            # Calculate matching bits
            orig_binary = self.text_to_binary(original_wm)
            attacked_binary = self.text_to_binary(attacked_wm)
            
            # Pad to same length
            max_len = max(len(orig_binary), len(attacked_binary))
            orig_binary = orig_binary.ljust(max_len, '0')
            attacked_binary = attacked_binary.ljust(max_len, '0')
            
            # Count matching bits
            matching_bits = sum(1 for a, b in zip(orig_binary, attacked_binary) if a == b)
            robustness_percent = (matching_bits / max_len) * 100
            
            return {
                "robustness": robustness_percent,
                "original_watermark": original_wm,
                "recovered_watermark": attacked_wm,
                "status": f"Watermark {robustness_percent:.2f}% intact after attack"
            }
            
        except Exception as e:
            app_logger.error(f"Robustness measurement error: {e}")
            return {"robustness": 0, "status": f"Error measuring robustness: {e}"}
    

    def get_image_pixel_info(self, image_path):
        """Get detailed pixel information about an image"""
        try:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            height, width, channels = img_rgb.shape
            
            info = {
                'width': width,
                'height': height,
                'channels': channels,
                'total_pixels': height * width,
                'total_bits_available': height * width * channels,
                'file_size': len(cv2.imencode('.jpg', img)[1]),
                'min_pixel_value': int(img_rgb.min()),
                'max_pixel_value': int(img_rgb.max()),
                'mean_pixel_value': float(img_rgb.mean()),
            }
            
            return info
            
        except Exception as e:
            app_logger.error(f"Error getting pixel info: {e}")
            return None


def convert_image_to_pixels(image_path):
    """Convert image file to pixel array"""
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return {
            'shape': img_rgb.shape,
            'pixels': img_rgb.tolist(),  # Convert to list for JSON serialization
            'dtype': str(img_rgb.dtype),
        }
    except Exception as e:
        app_logger.error(f"Error converting image to pixels: {e}")
        return None


def get_pixel_statistics(image_path):
    """Get detailed statistics about image pixels"""
    try:
        img = cv2.imread(image_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        stats = {
            'dimensions': f"{img_rgb.shape[0]}x{img_rgb.shape[1]} pixels",
            'channels': img_rgb.shape[2],
            'total_pixels': img_rgb.shape[0] * img_rgb.shape[1],
            'r_channel': {
                'min': int(img_rgb[:, :, 0].min()),
                'max': int(img_rgb[:, :, 0].max()),
                'mean': float(img_rgb[:, :, 0].mean()),
                'std': float(img_rgb[:, :, 0].std()),
            },
            'g_channel': {
                'min': int(img_rgb[:, :, 1].min()),
                'max': int(img_rgb[:, :, 1].max()),
                'mean': float(img_rgb[:, :, 1].mean()),
                'std': float(img_rgb[:, :, 1].std()),
            },
            'b_channel': {
                'min': int(img_rgb[:, :, 2].min()),
                'max': int(img_rgb[:, :, 2].max()),
                'mean': float(img_rgb[:, :, 2].mean()),
                'std': float(img_rgb[:, :, 2].std()),
            },
        }
        
        return stats
        
    except Exception as e:
        app_logger.error(f"Error getting pixel statistics: {e}")
        return None
