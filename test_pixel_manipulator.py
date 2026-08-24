import os
import cv2
import numpy as np
from pixel_watermark import PixelWatermark
from pixel_manipulator import PixelManipulator
from app import app

def run_tests():
    print("=== STARTING PIXEL MANIPULATION TESTS ===")
    
    # 1. Create a test RGB image
    test_img_path = "uploads/test_sample_image.png"
    os.makedirs("uploads", exist_ok=True)
    
    # Create 80x80 synthetic color image
    img = np.zeros((80, 80, 3), dtype=np.uint8)
    for y in range(80):
        for x in range(80):
            img[y, x] = [int((x/80)*255), int((y/80)*255), int(((x+y)/160)*255)]
    
    cv2.imwrite(test_img_path, img)
    print(f"Created test image: {test_img_path} (80x80 = 6400 pixels)")

    # 2. Embed invisible watermark
    wm = PixelWatermark(strength=5)
    embed_ok = wm.embed_in_lsb(test_img_path, "Capstone")
    print(f"LSB Watermark Embedded: {embed_ok}")
    extracted = wm.extract_lsb(test_img_path)
    print(f"Extracted initial watermark: '{extracted}'")
    assert "Capstone" in extracted, f"Expected 'Capstone' in extracted text, got '{extracted}'"

    # 3. Test PixelManipulator with different percentages: 5%, 10%, 25%, 50%
    manipulator = PixelManipulator()

    for pct in [5.0, 10.0, 25.0, 50.0]:
        out_path = f"uploads/test_manip_{int(pct)}pct.png"
        res = manipulator.manipulate_image(
            image_path=test_img_path,
            output_path=out_path,
            percentage=pct,
            method='noise',
            intensity=60
        )
        assert res['success'] == True, f"Failed for {pct}%: {res.get('error')}"
        expected_manip_count = int(round((pct / 100.0) * 6400))
        assert res['manipulated_count'] == expected_manip_count, f"Expected {expected_manip_count}, got {res['manipulated_count']}"
        assert res['total_pixels'] == 6400
        print(f"[OK] Percentage {pct}%: Manipulated {res['manipulated_count']} pixels | MAE={res['mae']} | PSNR={res['psnr']} dB | Watermark: '{res['extracted_text']}'")

    # 4. Test Excel Exporter
    excel_path = "uploads/test_manip_stats.xlsx"
    last_res = manipulator.manipulate_image(test_img_path, "uploads/test_manip_10pct.png", percentage=10.0, method='noise', intensity=50)
    exp_res = manipulator.export_manipulation_excel(
        image_path=test_img_path,
        manipulated_path="uploads/test_manip_10pct.png",
        excel_path=excel_path,
        stats_payload=last_res
    )
    assert exp_res['success'] == True, f"Excel export failed: {exp_res.get('error')}"
    assert os.path.exists(excel_path), "Excel file does not exist!"
    excel_size = os.path.getsize(excel_path)
    print(f"[OK] Excel report generated successfully: {excel_path} ({excel_size:,} bytes)")

    # 5. Test Flask Test Client Endpoints
    print("\n--- Testing Flask App Routes ---")
    client = app.test_client()
    
    with client.session_transaction() as sess:
        sess['logged_in'] = True
        sess['username'] = 'admin'

    # GET /pixel-manipulation/test_sample_image.png
    resp = client.get("/pixel-manipulation/test_sample_image.png")
    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}"
    assert b"Pixel Manipulation" in resp.data
    print("[OK] GET /pixel-manipulation/test_sample_image.png -> 200 OK")

    # POST /api/pixel-manipulate/test_sample_image.png
    resp_api = client.post("/api/pixel-manipulate/test_sample_image.png", json={
        "percentage": 25,
        "method": "lsb_corrupt",
        "intensity": 50
    })
    assert resp_api.status_code == 200, f"API error: {resp_api.status_code} {resp_api.data}"
    json_data = resp_api.get_json()
    assert json_data['success'] == True
    assert json_data['requested_percentage'] == 25
    assert len(json_data['pixel_records']) > 0
    print(f"[OK] POST /api/pixel-manipulate -> 200 OK (returned {len(json_data['pixel_records'])} records, {json_data['manipulated_count']} manipulated)")

    # GET /export-manipulation-excel/test_sample_image.png
    resp_exp = client.get("/export-manipulation-excel/test_sample_image.png?percentage=25&method=lsb_corrupt")
    assert resp_exp.status_code == 200, f"Excel download error: {resp_exp.status_code}"
    print(f"[OK] GET /export-manipulation-excel -> 200 OK ({len(resp_exp.data):,} bytes downloaded)")

    print("\n=== ALL PIXEL MANIPULATION & RGB STATS TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_tests()
