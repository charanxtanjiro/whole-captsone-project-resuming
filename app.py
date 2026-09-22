import logging
import os
import sqlite3
import traceback
from uuid import uuid4
import cv2
import numpy as np

from flask import Flask, redirect, render_template, request, send_from_directory, session, url_for, jsonify
from PIL import Image
from werkzeug.utils import secure_filename
from pixel_watermark import PixelWatermark, get_pixel_statistics
from image_pixel_converter import image_to_pixel_array
from pixel_excel_converter import PixelExcelConverter
from pixel_manipulator import PixelManipulator
from sipi_database_manager import SipiDatabaseManager, SIPI_MAIN_URL, SIPI_VOLUMES, SIPI_PRESETS

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "capstone-secret-key-change-in-prod")
app.config["UPLOAD_FOLDER"] = os.environ.get("UPLOAD_FOLDER", os.path.join(os.path.dirname(__file__), "uploads"))
app.config["DATABASE"] = os.environ.get("DATABASE", os.path.join(os.path.dirname(__file__), "data", "app.db"))
app.config["MAX_CONTENT_LENGTH"] = int(os.environ.get("MAX_FILE_SIZE", 16 * 1024 * 1024))

# SUPPORT ALL IMAGE FORMATS
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'gif', 'webp', 'bmp', 'tiff', 'tif', 'ico', 'svg', 'heic', 'heif', 'raw', 'psd', 'jp2', 'j2k', 'dds', 'jfif'}

logging.basicConfig(level=logging.INFO)
app.logger.setLevel(logging.INFO)

DEFAULT_USERNAME = os.environ.get("DEFAULT_USERNAME", "admin")
DEFAULT_PASSWORD = os.environ.get("DEFAULT_PASSWORD", "password")

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)
os.makedirs(os.path.dirname(app.config["DATABASE"]), exist_ok=True)

# Initialize pixel watermarking, manipulator, and SIPI database manager
pixel_wm = PixelWatermark(strength=5)
pixel_converter = PixelExcelConverter()
pixel_manipulator = PixelManipulator()
sipi_manager = SipiDatabaseManager()



def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def convert_image_to_compatible_format(file_path):
    """Convert any image format to PNG for processing"""
    try:
        img = Image.open(file_path)
        
        # Convert RGBA to RGB if needed
        if img.mode == 'RGBA':
            rgb_img = Image.new('RGB', img.size, (255, 255, 255))
            rgb_img.paste(img, mask=img.split()[3])
            img = rgb_img
        elif img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Save as PNG for consistency
        png_path = file_path + ".png"
        img.save(png_path, "PNG")
        os.remove(file_path)
        return png_path
    except Exception as e:
        app.logger.error(f"Image format conversion error: {e}")
        raise


def get_db_connection():
    try:
        conn = sqlite3.connect(app.config["DATABASE"])
        conn.row_factory = sqlite3.Row
        return conn
    except sqlite3.Error as e:
        app.logger.error(f"Database connection error: {e}")
        raise


def init_db():
    try:
        conn = get_db_connection()
        
        # Create uploads table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS uploads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL UNIQUE,
                original_name TEXT NOT NULL,
                uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                has_watermark BOOLEAN DEFAULT 1,
                pixels_converted BOOLEAN DEFAULT 0
            )
        """)
        
        # Create users table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL UNIQUE,
                password TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Add missing column if needed
        try:
            cursor = conn.cursor()
            cursor.execute("PRAGMA table_info(uploads)")
            columns = [col[1] for col in cursor.fetchall()]
            
            if 'pixels_converted' not in columns:
                app.logger.info("Adding missing pixels_converted column...")
                conn.execute("ALTER TABLE uploads ADD COLUMN pixels_converted BOOLEAN DEFAULT 0")
                app.logger.info("Column added successfully")
        except Exception as e:
            app.logger.warning(f"Column check warning: {e}")
        
        conn.commit()
        conn.close()
        app.logger.info("Database initialized successfully")
    except sqlite3.Error as e:
        app.logger.error(f"Database initialization error: {e}")


init_db()


def extract_lsb_watermark(image_path):
    """Extract watermark from LSB"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
            
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        flat_img = img_rgb.flatten()
        
        # Extract LSBs
        watermark_bits = ''
        for pixel_value in flat_img[:72]:  # "Capstone" = 8 chars * 8 bits
            lsb = pixel_value & 0x01
            watermark_bits += str(lsb)
        
        # Convert binary to text
        watermark_text = ''
        for i in range(0, len(watermark_bits), 8):
            byte = watermark_bits[i:i+8]
            if len(byte) == 8:
                char = chr(int(byte, 2))
                watermark_text += char
        
        return watermark_text
    except Exception as e:
        app.logger.error(f"LSB extraction error: {e}")
        return None


@app.route("/")
def index():
    """Main dashboard"""
    if "logged_in" not in session:
        return redirect(url_for("login"))

    try:
        conn = get_db_connection()
        uploads = conn.execute(
            "SELECT id, filename, original_name, uploaded_at, pixels_converted FROM uploads ORDER BY uploaded_at DESC"
        ).fetchall()
        conn.close()
    except sqlite3.Error as e:
        app.logger.error(f"Error fetching uploads: {e}")
        uploads = []

    return render_template("index.html", uploads=uploads)


@app.route("/login", methods=["GET", "POST"])
def login():
    """User login"""
    error = None

    if request.method == "POST":
        username = request.form.get("username", "")
        password = request.form.get("password", "")

        try:
            conn = get_db_connection()
            user = conn.execute("SELECT username, password FROM users WHERE username = ?", (username,)).fetchone()
            conn.close()
        except sqlite3.Error as e:
            app.logger.error(f"Login database error: {e}")
            error = "Unable to access login database. Please try again later."
            return render_template("login.html", error=error)

        if user is not None and password == user["password"]:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))

        if username == DEFAULT_USERNAME and password == DEFAULT_PASSWORD:
            session["logged_in"] = True
            session["username"] = username
            return redirect(url_for("index"))

        error = "Invalid username or password"
        app.logger.warning(f"Failed login attempt for user: {username}")

    return render_template("login.html", error=error)


@app.route("/register", methods=["GET", "POST"])
def register():
    """User registration"""
    error = None
    success = None

    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")

        if not username or not password:
            error = "Username and password are required"
        elif password != confirm_password:
            error = "Passwords do not match"
        else:
            try:
                conn = get_db_connection()
                existing_user = conn.execute("SELECT id FROM users WHERE username = ?", (username,)).fetchone()

                if existing_user is not None:
                    error = "Username already exists"
                else:
                    conn.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
                    conn.commit()
                    success = "Account created successfully. You can now log in."
                    conn.close()
                    return render_template("login.html", success=success)

                conn.close()
            except sqlite3.Error as e:
                app.logger.error(f"Registration error: {e}")
                error = "Registration failed. Please try again."

    return render_template("register.html", error=error, success=success)


@app.route("/logout")
def logout():
    """User logout"""
    session.pop("logged_in", None)
    session.pop("username", None)
    return redirect(url_for("login"))


@app.route("/upload", methods=["POST"])
def upload_file():
    """Upload image with watermark"""
    app.logger.info("=== UPLOAD START ===")
    
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file provided"}), 400

    file = request.files["file"]
    
    if file.filename == "":
        return jsonify({"success": False, "error": "No filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"success": False, "error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 400

    original_name = secure_filename(file.filename)
    
    if original_name == "":
        return jsonify({"success": False, "error": "Invalid filename"}), 400

    file_path = None
    try:
        # Save original file
        stored_name = f"{uuid4().hex}_{original_name}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)
        file.save(file_path)
        
        app.logger.info(f"Uploaded: {stored_name}, Size: {os.path.getsize(file_path)} bytes")

        # Convert to compatible format (PNG)
        try:
            file_path = convert_image_to_compatible_format(file_path)
            stored_name = os.path.basename(file_path)
            app.logger.info(f"Converted to: {stored_name}")
        except Exception as e:
            app.logger.warning(f"Format conversion skipped: {e}")

        # Add pixel-level watermark (LSB method)
        try:
            app.logger.info("Adding LSB watermark...")
            pixel_wm.embed_in_lsb(file_path, "Capstone")
            app.logger.info("Watermark embedded successfully")
        except Exception as e:
            app.logger.warning(f"Watermark embedding warning: {e}")

        # Save to DB
        conn = get_db_connection()
        conn.execute("INSERT INTO uploads (filename, original_name, pixels_converted) VALUES (?, ?, ?)",
                    (stored_name, original_name, 0))
        conn.commit()
        conn.close()
        
        app.logger.info("=== UPLOAD SUCCESS ===")
        return jsonify({"success": True, "message": "Image uploaded successfully", "filename": stored_name})
        
    except Exception as e:
        app.logger.error(f"UPLOAD ERROR: {e}")
        if file_path and os.path.exists(file_path):
            try:
                os.remove(file_path)
            except:
                pass
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/uploads/<filename>")
def uploaded_file(filename):
    """Serve uploaded file"""
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename)
    except Exception as e:
        app.logger.error(f"File download error: {e}")
        return "File not found", 404


@app.route("/api/sipi/catalog")
def api_sipi_catalog():
    """Get full catalog of USC-SIPI Image Database volumes and curated presets"""
    try:
        return jsonify({
            "success": True,
            "catalog": sipi_manager.get_catalog()
        }), 200
    except Exception as e:
        app.logger.error(f"SIPI catalog error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sipi/import", methods=["POST"])
def api_sipi_import():
    """Import a USC-SIPI benchmark preset or custom URL, embed LSB watermark, and register in database"""
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401

    try:
        req_data = request.get_json() or {}
        preset_id = req_data.get("preset_id")
        custom_url = req_data.get("url")
        custom_name = req_data.get("name")

        clean_name = secure_filename(custom_name or f"sipi_{preset_id or 'benchmark'}.png")
        if not clean_name.endswith('.png'):
            clean_name += '.png'
        stored_name = f"{uuid4().hex}_{clean_name}"
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], stored_name)

        if custom_url:
            app.logger.info(f"Importing image from URL: {custom_url}")
            dl_res = sipi_manager.import_from_url(custom_url, file_path)
            if not dl_res.get("success"):
                return jsonify({"success": False, "error": dl_res.get("error")}), 400
        elif preset_id:
            app.logger.info(f"Generating SIPI preset asset: {preset_id}")
            sipi_manager.generate_benchmark_asset(preset_id, file_path)
        else:
            return jsonify({"success": False, "error": "Either preset_id or url must be provided"}), 400

        # Embed LSB watermark
        try:
            pixel_wm.embed_in_lsb(file_path, "Capstone")
            app.logger.info(f"Watermark embedded into imported SIPI image {stored_name}")
        except Exception as e:
            app.logger.warning(f"Watermark embedding warning: {e}")

        # Save to database
        conn = get_db_connection()
        conn.execute("INSERT INTO uploads (filename, original_name, pixels_converted) VALUES (?, ?, ?)",
                     (stored_name, f"SIPI_{preset_id or 'Custom'}_{clean_name}", 0))
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": f"Successfully imported SIPI benchmark '{clean_name}' with embedded watermark!",
            "filename": stored_name
        }), 200

    except Exception as e:
        app.logger.error(f"SIPI import error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/convert-pixels/<filename>", methods=["POST"])
def convert_pixels(filename):
    """Convert image to pixels with fast limit option"""
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Not logged in"}), 401
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        req_data = request.get_json() if request.is_json else {}
        limit_setting = request.args.get("limit") or (req_data.get("limit") if req_data else None) or "standard_50"
        max_w, max_h = pixel_converter.parse_sample_limit(limit_setting)
        
        app.logger.info(f"Converting {filename} to pixels with limit {max_w}x{max_h}...")
        
        # Get pixel grid as JSON with fast limit
        grid_data = pixel_converter.get_pixel_grid_json(file_path, max_width=max_w, max_height=max_h)
        
        if not grid_data["success"]:
            return jsonify({"success": False, "error": grid_data["error"]}), 500
        
        # Update database
        conn = get_db_connection()
        conn.execute("UPDATE uploads SET pixels_converted = 1 WHERE filename = ?", (filename,))
        conn.commit()
        conn.close()
        
        app.logger.info(f"Pixels converted successfully for {filename}")
        return jsonify({
            "success": True, 
            "message": "Pixels converted successfully",
            "grid": grid_data,
            "redirect_url": url_for("pixel_grid_view", filename=filename)
        })
        
    except Exception as e:
        app.logger.error(f"Pixel conversion error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/pixel-grid/<filename>")
def pixel_grid_view(filename):
    """Display pixel grid"""
    if "logged_in" not in session:
        return redirect(url_for("login"))
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return render_template("error.html", message="Image not found"), 404
        
        limit_setting = request.args.get("limit", "standard_50")
        max_w, max_h = pixel_converter.parse_sample_limit(limit_setting)
        grid_data = pixel_converter.get_pixel_grid_json(file_path, max_width=max_w, max_height=max_h)
        
        if not grid_data["success"]:
            return render_template("error.html", message=grid_data["error"]), 500
        
        return render_template("pixel_grid.html", filename=filename, grid=grid_data, active_limit=limit_setting)
        
    except Exception as e:
        app.logger.error(f"Pixel grid error: {e}")
        return render_template("error.html", message=f"Error: {e}"), 500


@app.route("/export-excel/<filename>")
def export_excel(filename):
    """Export to Excel with fast limits and style caching"""
    if "logged_in" not in session:
        return redirect(url_for("login"))
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        limit_setting = request.args.get("limit", "standard_50")
        include_viz = request.args.get("viz", "true").lower() == "true"
        excel_filename = f"{filename.split('.')[0]}_pixels_{limit_setting}.xlsx"
        excel_path = os.path.join(app.config["UPLOAD_FOLDER"], excel_filename)
        
        result = pixel_converter.image_to_excel(file_path, excel_path, sample_size=limit_setting, include_viz=include_viz)
        
        if result["success"]:
            app.logger.info(f"Excel exported rapidly: {excel_path}")
            return send_from_directory(app.config["UPLOAD_FOLDER"], excel_filename, as_attachment=True)
        else:
            return jsonify({"success": False, "error": result["error"]}), 500
        
    except Exception as e:
        app.logger.error(f"Excel export error: {e}")
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/reconstruct/<filename>")
def reconstruct_image(filename):
    """Reconstruct image from Excel"""
    if "logged_in" not in session:
        return redirect(url_for("login"))
    
    try:
        if not filename.endswith(".xlsx"):
            return render_template("error.html", message="Invalid file type. Only Excel files supported."), 400
        
        excel_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(excel_path):
            return render_template("error.html", message="Excel file not found"), 404
        
        reconstructed_filename = f"reconstructed_{uuid4().hex}.png"
        reconstructed_path = os.path.join(app.config["UPLOAD_FOLDER"], reconstructed_filename)
        
        result = pixel_converter.excel_to_image(excel_path, reconstructed_path)
        
        if result["success"]:
            app.logger.info(f"Image reconstructed: {reconstructed_path}")
            return render_template("reconstruction_result.html",
                                 original_filename=filename,
                                 reconstructed_filename=reconstructed_filename,
                                 result=result)
        else:
            return render_template("error.html", message=result["error"]), 500
        
    except Exception as e:
        app.logger.error(f"Reconstruction error: {e}")
        return render_template("error.html", message=f"Error: {e}"), 500


@app.route("/pixel-info/<filename>")
def pixel_info(filename):
    """Show pixel info"""
    if "logged_in" not in session:
        return redirect(url_for("login"))
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return render_template("error.html", message="Image not found"), 404
        
        pixel_stats = get_pixel_statistics(file_path)
        
        if not pixel_stats:
            return render_template("error.html", message="Failed to process image"), 500
        
        return render_template("pixel_info.html", filename=filename, stats=pixel_stats)
        
    except Exception as e:
        app.logger.error(f"Pixel info error: {e}")
        return render_template("error.html", message=f"Error: {e}"), 500


@app.route("/pixel-manipulation/<filename>")
def pixel_manipulation_view(filename):
    """Render the Pixel Manipulation & RGB Statistics interface"""
    if "logged_in" not in session:
        return redirect(url_for("login"))
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return render_template("error.html", message="Image not found"), 404
        
        return render_template("pixel_manipulation.html", filename=filename)
        
    except Exception as e:
        app.logger.error(f"Pixel manipulation view error: {e}")
        return render_template("error.html", message=f"Error: {e}"), 500


@app.route("/api/pixel-manipulate/<filename>", methods=["POST"])
def api_pixel_manipulate(filename):
    """API endpoint to execute pixel manipulation (single percentage or ROI) and calculate RGB statistics"""
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Source image not found"}), 404
        
        req_data = request.get_json() or {}
        percentage = float(req_data.get("percentage", 10))
        method = req_data.get("method", "noise")
        intensity = int(req_data.get("intensity", 50))
        roi = req_data.get("roi", None)
        
        # Prepare output manipulated file path
        base_name, ext = os.path.splitext(filename)
        pct_str = int(percentage) if percentage == int(percentage) else percentage
        roi_suffix = ""
        if roi:
            roi_suffix = f"_roi_{roi.get('x1', roi.get('x', 0))}_{roi.get('y1', roi.get('y', 0))}"
        manipulated_filename = f"{base_name}_manip_{pct_str}pct_{method}{roi_suffix}.png"
        manipulated_path = os.path.join(app.config["UPLOAD_FOLDER"], manipulated_filename)
        
        # Execute manipulation
        result = pixel_manipulator.manipulate_image(
            image_path=file_path,
            output_path=manipulated_path,
            percentage=percentage,
            method=method,
            intensity=intensity,
            roi=roi
        )
        
        if not result.get("success"):
            return jsonify({"success": False, "error": result.get("error")}), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"API pixel manipulation error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pixel-manipulate-multi/<filename>", methods=["POST"])
def api_pixel_manipulate_multi(filename):
    """API endpoint to execute manipulation simultaneously across all standard percentages (5%, 10%, 25%, 75%, 100%)"""
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Source image not found"}), 404
        
        req_data = request.get_json() or {}
        percentages = req_data.get("percentages", [5.0, 10.0, 25.0, 75.0, 100.0])
        method = req_data.get("method", "noise")
        intensity = int(req_data.get("intensity", 50))
        roi = req_data.get("roi", None)
        
        result = pixel_manipulator.manipulate_multi_percentages(
            image_path=file_path,
            output_dir=app.config["UPLOAD_FOLDER"],
            percentages=percentages,
            method=method,
            intensity=intensity,
            roi=roi
        )
        
        if not result.get("success"):
            return jsonify({"success": False, "error": result.get("error")}), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"API multi-percentage manipulation error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pixel-custom-byte-manipulate/<filename>", methods=["POST"])
def api_pixel_custom_byte_manipulate(filename):
    """API endpoint for direct custom pixel byte / byte range manipulation"""
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Source image not found"}), 404
        
        req_data = request.get_json() or {}
        operation = req_data.get("operation", "set_value")
        channels = req_data.get("channels", ["R", "G", "B"])
        op_params = req_data.get("op_params", {})
        roi = req_data.get("roi", None)
        byte_range_filter = req_data.get("byte_range_filter", None)
        
        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}_custom_{operation}.png"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        
        result = pixel_manipulator.manipulate_custom_bytes(
            image_path=file_path,
            output_path=output_path,
            roi=roi,
            channels=channels,
            operation=operation,
            op_params=op_params,
            byte_range_filter=byte_range_filter
        )
        
        if not result.get("success"):
            return jsonify({"success": False, "error": result.get("error")}), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"API custom byte manipulation error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pixel-manipulate-gray/<filename>", methods=["POST"])
def api_pixel_manipulate_gray(filename):
    """API endpoint to execute Grayscale pixel manipulation (bit planes, threshold, CLAHE, gamma, etc.)"""
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Source image not found"}), 404
        
        req_data = request.get_json() or {}
        gray_method = req_data.get("gray_method", "luminance")
        op_type = req_data.get("op_type", "bit_plane_slice")
        op_params = req_data.get("op_params", {})
        roi = req_data.get("roi", None)
        
        base_name, ext = os.path.splitext(filename)
        output_filename = f"{base_name}_gray_{op_type}.png"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)
        
        result = pixel_manipulator.manipulate_grayscale(
            image_path=file_path,
            output_path=output_path,
            gray_method=gray_method,
            op_type=op_type,
            op_params=op_params,
            roi=roi
        )
        
        if not result.get("success"):
            return jsonify({"success": False, "error": result.get("error")}), 500
        
        return jsonify(result), 200
        
    except Exception as e:
        app.logger.error(f"API Grayscale manipulation error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pixel-manipulate-gray-percentage/<filename>", methods=["POST"])
def api_pixel_manipulate_gray_percentage(filename):
    """
    API endpoint for Grayscale pixel manipulation with preset (5%, 10%, 25%, 50%, 75%, 100%)
    or arbitrary custom percentages (e.g. 37%, 48%, 99%) strictly within user-selected ROI area.
    """
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Source image not found"}), 404

        req_data = request.get_json() or {}
        percentage = float(req_data.get("percentage", 10.0))
        method = req_data.get("method", "gray_noise")
        intensity = int(req_data.get("intensity", 50))
        roi = req_data.get("roi", None)
        gray_method = req_data.get("gray_method", "luminance")
        op_params = req_data.get("op_params", {})

        base_name, ext = os.path.splitext(filename)
        pct_str = int(percentage) if percentage == int(percentage) else percentage
        roi_suffix = ""
        if roi:
            roi_suffix = f"_roi_{roi.get('x1', roi.get('x', 0))}_{roi.get('y1', roi.get('y', 0))}"
        output_filename = f"{base_name}_gray_manip_{pct_str}pct_{method}{roi_suffix}.png"
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], output_filename)

        result = pixel_manipulator.manipulate_grayscale_percentage(
            image_path=file_path,
            output_path=output_path,
            percentage=percentage,
            method=method,
            intensity=intensity,
            roi=roi,
            gray_method=gray_method,
            op_params=op_params
        )

        if not result.get("success"):
            return jsonify({"success": False, "error": result.get("error")}), 500

        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"API Grayscale percentage manipulation error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/pixel-manipulate-gray-multi/<filename>", methods=["POST"])
def api_pixel_manipulate_gray_multi(filename):
    """
    API endpoint for simultaneous multi-percentage (5%, 10%, 25%, 50%, 75%, 100%) Grayscale matrix execution.
    """
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Source image not found"}), 404

        req_data = request.get_json() or {}
        percentages = req_data.get("percentages", [5.0, 10.0, 25.0, 50.0, 75.0, 100.0])
        method = req_data.get("method", "gray_noise")
        intensity = int(req_data.get("intensity", 50))
        roi = req_data.get("roi", None)
        gray_method = req_data.get("gray_method", "luminance")
        op_params = req_data.get("op_params", {})

        result = pixel_manipulator.manipulate_grayscale_multi_percentages(
            image_path=file_path,
            output_dir=app.config["UPLOAD_FOLDER"],
            percentages=percentages,
            method=method,
            intensity=intensity,
            roi=roi,
            gray_method=gray_method,
            op_params=op_params
        )

        if not result.get("success"):
            return jsonify({"success": False, "error": result.get("error")}), 500

        return jsonify(result), 200

    except Exception as e:
        app.logger.error(f"API Grayscale multi-percentage manipulation error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/convert-grayscale/<filename>", methods=["POST"])
def api_convert_grayscale(filename):
    """Convert an existing image to Grayscale, embed watermark, and register as new uploaded entry"""
    if "logged_in" not in session:
        return jsonify({"success": False, "error": "Unauthorized"}), 401

    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Source image not found"}), 404

        req_data = request.get_json() if request.is_json else {}
        gray_method = req_data.get("gray_method", "luminance") if req_data else "luminance"

        img_bgr = cv2.imread(file_path)
        if img_bgr is None:
            return jsonify({"success": False, "error": "Unable to read image"}), 400

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gray_arr = pixel_manipulator.convert_to_grayscale_array(img_rgb, method=gray_method)
        gray_3ch = cv2.cvtColor(gray_arr, cv2.COLOR_GRAY2BGR)

        base_name, _ = os.path.splitext(filename)
        new_stored_name = f"{uuid4().hex}_gray_{gray_method}.png"
        new_file_path = os.path.join(app.config["UPLOAD_FOLDER"], new_stored_name)

        cv2.imwrite(new_file_path, gray_3ch)

        # Embed watermark into grayscale image
        try:
            pixel_wm.embed_in_lsb(new_file_path, "Capstone")
            app.logger.info(f"Watermark embedded into grayscale image: {new_stored_name}")
        except Exception as e:
            app.logger.warning(f"Grayscale watermark embed warning: {e}")

        # Save to database
        conn = get_db_connection()
        conn.execute("INSERT INTO uploads (filename, original_name, pixels_converted) VALUES (?, ?, ?)",
                     (new_stored_name, f"Grayscale_{gray_method.upper()}_{filename}", 0))
        conn.commit()
        conn.close()

        return jsonify({
            "success": True,
            "message": f"Successfully converted to Grayscale ({gray_method}) with watermark embedded!",
            "filename": new_stored_name
        }), 200

    except Exception as e:
        app.logger.error(f"Grayscale conversion error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/export-manipulation-excel/<filename>")
def export_manipulation_excel(filename):
    """Export pixel manipulation RGB or Grayscale statistical report as an Excel file"""
    if "logged_in" not in session:
        return redirect(url_for("login"))
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return jsonify({"success": False, "error": "Image not found"}), 404
        
        is_gray = request.args.get("is_gray", "false").lower() == "true"
        is_gray_pct = request.args.get("is_gray_percentage", "false").lower() == "true" or (is_gray and request.args.get("gray_percentage") is not None)
        is_custom = request.args.get("is_custom", "false").lower() == "true"
        base_name, ext = os.path.splitext(filename)
        
        # Parse ROI parameters if provided
        roi = None
        if request.args.get("roi_x1") is not None and request.args.get("roi_x2") is not None:
            roi = {
                'x1': int(request.args.get("roi_x1", 0)),
                'y1': int(request.args.get("roi_y1", 0)),
                'x2': int(request.args.get("roi_x2", 0)),
                'y2': int(request.args.get("roi_y2", 0))
            }
        
        if is_gray_pct:
            percentage = float(request.args.get("percentage") or request.args.get("gray_percentage", 10.0))
            method = request.args.get("method", "gray_noise")
            intensity = int(request.args.get("intensity", 50))
            gray_method = request.args.get("gray_method", "luminance")
            pct_str = int(percentage) if percentage == int(percentage) else percentage
            roi_suffix = f"_roi_{roi['x1']}_{roi['y1']}" if roi else ""
            manipulated_filename = f"{base_name}_gray_manip_{pct_str}pct_{method}{roi_suffix}.png"
            manipulated_path = os.path.join(app.config["UPLOAD_FOLDER"], manipulated_filename)

            stats_payload = pixel_manipulator.manipulate_grayscale_percentage(
                image_path=file_path,
                output_path=manipulated_path,
                percentage=percentage,
                method=method,
                intensity=intensity,
                roi=roi,
                gray_method=gray_method
            )
        elif is_gray:
            gray_method = request.args.get("gray_method", "luminance")
            op_type = request.args.get("op_type", "bit_plane_slice")
            manipulated_filename = f"{base_name}_gray_{op_type}.png"
            manipulated_path = os.path.join(app.config["UPLOAD_FOLDER"], manipulated_filename)

            op_params = {}
            if request.args.get("bit_plane") is not None: op_params['bit_plane'] = int(request.args.get("bit_plane"))
            if request.args.get("cutoff") is not None: op_params['cutoff'] = int(request.args.get("cutoff"))
            if request.args.get("gamma") is not None: op_params['gamma'] = float(request.args.get("gamma"))

            stats_payload = pixel_manipulator.manipulate_grayscale(
                image_path=file_path,
                output_path=manipulated_path,
                gray_method=gray_method,
                op_type=op_type,
                op_params=op_params,
                roi=roi
            )
        elif is_custom:
            operation = request.args.get("operation", "set_value")
            channels = request.args.getlist("channels") or ["R", "G", "B"]
            val = request.args.get("value")
            delta = request.args.get("delta")
            mask = request.args.get("mask")
            min_val = request.args.get("min_val")
            max_val = request.args.get("max_val")
            hex_color = request.args.get("hex_color")
            
            op_params = {}
            if val is not None: op_params['value'] = int(val)
            if delta is not None: op_params['delta'] = int(delta)
            if mask is not None: op_params['mask'] = int(mask)
            if min_val is not None: op_params['min_val'] = int(min_val)
            if max_val is not None: op_params['max_val'] = int(max_val)
            if hex_color is not None: op_params['hex_color'] = hex_color
            
            manipulated_filename = f"{base_name}_custom_{operation}.png"
            manipulated_path = os.path.join(app.config["UPLOAD_FOLDER"], manipulated_filename)
            
            stats_payload = pixel_manipulator.manipulate_custom_bytes(
                image_path=file_path,
                output_path=manipulated_path,
                roi=roi,
                channels=channels,
                operation=operation,
                op_params=op_params
            )
        else:
            percentage = float(request.args.get("percentage", 10))
            method = request.args.get("method", "noise")
            intensity = int(request.args.get("intensity", 50))
            
            pct_str = int(percentage) if percentage == int(percentage) else percentage
            manipulated_filename = f"{base_name}_manip_{pct_str}pct_{method}.png"
            manipulated_path = os.path.join(app.config["UPLOAD_FOLDER"], manipulated_filename)
            
            stats_payload = pixel_manipulator.manipulate_image(
                image_path=file_path,
                output_path=manipulated_path,
                percentage=percentage,
                method=method,
                intensity=intensity,
                roi=roi
            )
        
        excel_filename = f"{base_name}_pixel_manipulation_stats.xlsx"
        excel_path = os.path.join(app.config["UPLOAD_FOLDER"], excel_filename)
        
        export_res = pixel_manipulator.export_manipulation_excel(
            image_path=file_path,
            manipulated_path=manipulated_path,
            excel_path=excel_path,
            stats_payload=stats_payload
        )
        
        if export_res.get("success"):
            return send_from_directory(app.config["UPLOAD_FOLDER"], excel_filename, as_attachment=True)
        else:
            return jsonify({"success": False, "error": export_res.get("error")}), 500
            
    except Exception as e:
        app.logger.error(f"Manipulation Excel export error: {e}", exc_info=True)
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/detect-watermark/<filename>")
def detect_watermark(filename):
    """Detect watermark"""
    if "logged_in" not in session:
        return redirect(url_for("login"))
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return render_template("error.html", message="Image not found"), 404
        
        watermark_text = extract_lsb_watermark(file_path)
        watermark_detected = watermark_text and "Capstone" in watermark_text
        
        robustness = {
            "watermark_detected": watermark_detected,
            "extracted_text": watermark_text if watermark_text else "No watermark found"
        }
        
        return render_template("watermark_result.html", filename=filename, robustness=robustness)
        
    except Exception as e:
        app.logger.error(f"Watermark detection error: {e}")
        return render_template("error.html", message=f"Error: {e}"), 500


@app.route("/attack/<filename>/<attack_type>")
def attack_image(filename, attack_type):
    """Apply attack"""
    if "logged_in" not in session:
        return redirect(url_for("login"))
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        if not os.path.exists(file_path):
            return render_template("error.html", message="Image not found"), 404
        
        img = cv2.imread(file_path)
        if img is None:
            return render_template("error.html", message="Failed to read image"), 500
        
        attack_descriptions = {
            "compression": "JPEG Compression (50% quality)",
            "noise": "Gaussian Noise Added",
            "blur": "Gaussian Blur Filter",
            "brightness": "Brightness & Contrast Adjustment",
            "rotation": "5-Degree Rotation",
            "resize": "Resize to 50% then back",
            "crop": "Crop 10% from edges",
            "pixelate": "Pixelate"
        }
        
        if attack_type == "compression":
            temp_path = file_path + "_compressed.jpg"
            cv2.imwrite(temp_path, img, [cv2.IMWRITE_JPEG_QUALITY, 50])
            attacked_img = cv2.imread(temp_path)
            os.remove(temp_path)
        elif attack_type == "noise":
            noise = np.random.normal(0, 25, img.shape)
            attacked_img = np.clip(img + noise, 0, 255).astype(np.uint8)
        elif attack_type == "blur":
            attacked_img = cv2.GaussianBlur(img, (5, 5), 0)
        elif attack_type == "brightness":
            attacked_img = cv2.convertScaleAbs(img, alpha=1.5, beta=30)
        elif attack_type == "rotation":
            h, w = img.shape[:2]
            center = (w // 2, h // 2)
            matrix = cv2.getRotationMatrix2D(center, 5, 1.0)
            attacked_img = cv2.warpAffine(img, matrix, (w, h))
        elif attack_type == "resize":
            attacked_img = cv2.resize(img, (img.shape[1]//2, img.shape[0]//2))
            attacked_img = cv2.resize(attacked_img, (img.shape[1], img.shape[0]))
        elif attack_type == "crop":
            h, w = img.shape[:2]
            margin = int(w * 0.1)
            attacked_img = img[margin:h-margin, margin:w-margin]
            attacked_img = cv2.resize(attacked_img, (w, h))
        elif attack_type == "pixelate":
            h, w = img.shape[:2]
            temp_small = cv2.resize(img, (w//8, h//8))
            attacked_img = cv2.resize(temp_small, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            return render_template("error.html", message="Unknown attack"), 400
        
        attacked_path = file_path + f"_{attack_type}_attacked.png"
        cv2.imwrite(attacked_path, attacked_img)
        attack_desc = attack_descriptions.get(attack_type, attack_type)
        
        app.logger.info(f"Attack '{attack_type}' applied to {filename}")
        
        return render_template("attack_result.html", 
                             attack_type=attack_desc,
                             original_filename=filename,
                             attacked_filename=os.path.basename(attacked_path))
        
    except Exception as e:
        app.logger.error(f"Attack error: {e}")
        return render_template("error.html", message=f"Error: {e}"), 500


@app.route("/delete/<filename>", methods=["DELETE"])
def delete_image(filename):
    """Delete image"""
    if "logged_in" not in session:
        return jsonify({"error": "unauthorized"}), 401
    
    try:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404
        
        conn = get_db_connection()
        conn.execute("DELETE FROM uploads WHERE filename = ?", (filename,))
        conn.commit()
        conn.close()
        
        os.remove(file_path)
        
        for attacked_file in os.listdir(app.config["UPLOAD_FOLDER"]):
            if attacked_file.startswith(filename):
                try:
                    os.remove(os.path.join(app.config["UPLOAD_FOLDER"], attacked_file))
                except:
                    pass
        
        app.logger.info(f"Deleted: {filename}")
        return jsonify({"success": True}), 200
        
    except Exception as e:
        app.logger.error(f"Delete error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/api/uploads")
def api_uploads():
    """Get uploads as JSON"""
    if "logged_in" not in session:
        return jsonify({"error": "unauthorized"}), 401

    try:
        conn = get_db_connection()
        uploads = conn.execute(
            "SELECT id, filename, original_name, uploaded_at, pixels_converted FROM uploads ORDER BY uploaded_at DESC"
        ).fetchall()
        conn.close()
        return jsonify({"uploads": [dict(row) for row in uploads]})
    except sqlite3.Error as e:
        return jsonify({"error": "database error"}), 500


@app.route("/health")
def health():
    """Health check"""
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        return jsonify({"status": "ok"}), 200
    except Exception as e:
        return jsonify({"status": "error"}), 500


@app.errorhandler(400)
def bad_request(error):
    app.logger.error(f"Bad request: {error}")
    return render_template("error.html", message="Bad request"), 400


@app.errorhandler(404)
def not_found(error):
    app.logger.error(f"Not found: {error}")
    return render_template("error.html", message="Page not found"), 404


@app.errorhandler(500)
def internal_error(error):
    app.logger.error(f"Internal error: {error}")
    return render_template("error.html", message="Internal server error"), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)), debug=False)
