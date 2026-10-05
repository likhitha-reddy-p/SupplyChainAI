"""
QR and Barcode Shipment Identification Module
Author: Lead AI/ML Developer
Project: AI-Based Supply Chain Disruption Prediction and Recovery System
"""

import os
import json
import pandas as pd
import numpy as np
import qrcode
from PIL import Image
import cv2

class ShipmentQRIdentifier:
    """
    Physical identification and digital ingestion layer.
    Encodes and decodes Shipment identifiers to query operational databases.
    """
    def __init__(self, data_path=None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.base_dir = base_dir
        if data_path is None:
            data_path = os.path.join(base_dir, "data", "processed", "cleaned_supply_chain.csv")
        self.data_path = data_path
        self._df = None
        
    def _load_data(self):
        if self._df is None:
            self._df = pd.read_csv(self.data_path)
        return self._df

    def generate_shipment_qr(self, shipment_id, save_path=None):
        """Generates a QR code image for a given Shipment_ID."""
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_M,
            box_size=10,
            border=4,
        )
        # Encode identifier payload
        payload = f"SHIPMENT:{shipment_id}"
        qr.add_data(payload)
        qr.make(fit=True)
        
        img = qr.make_image(fill_color="#0D47A1", back_color="white")
        
        if save_path:
            img.save(save_path)
            
        return img

    def decode_shipment_qr(self, image_path_or_bytes):
        """
        Decodes a QR code image using OpenCV QRCodeDetector or pyzbar.
        Returns the parsed Shipment_ID string.
        """
        try:
            if isinstance(image_path_or_bytes, str):
                img = cv2.imread(image_path_or_bytes)
            else:
                # Array or PIL image
                if hasattr(image_path_or_bytes, 'read'):
                    # File-like object / buffer
                    file_bytes = np.asarray(bytearray(image_path_or_bytes.read()), dtype=np.uint8)
                    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                else:
                    img = np.array(image_path_or_bytes)
                    if len(img.shape) == 3 and img.shape[2] == 3:
                        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            detector = cv2.QRCodeDetector()
            data, bbox, _ = detector.detectAndDecode(img)
            
            if data:
                clean_id = data.replace("SHIPMENT:", "").strip()
                return clean_id
                
            # Fallback to pyzbar if available
            try:
                from pyzbar.pyzbar import decode as pyzbar_decode
                decoded_objects = pyzbar_decode(img)
                for obj in decoded_objects:
                    text = obj.data.decode('utf-8')
                    return text.replace("SHIPMENT:", "").strip()
            except:
                pass
                
            return None
        except Exception as e:
            print(f"Error decoding QR image: {e}")
            return None

    def lookup_shipment(self, shipment_id):
        """Retrieves the complete shipment telemetry row by Shipment_ID."""
        df = self._load_data()
        clean_id = str(shipment_id).strip()
        matched = df[df['Shipment_ID'] == clean_id]
        if len(matched) == 0:
            return None
        return matched.iloc[0].to_dict()

def test_qr_module():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    qr_dir = os.path.join(base_dir, "qr_barcode")
    sample_qr_path = os.path.join(qr_dir, "sample_shipment_qr.png")
    
    identifier = ShipmentQRIdentifier()
    test_id = "SHP-0000001"
    
    print(f"1. Generating QR code for shipment: {test_id}...", flush=True)
    identifier.generate_shipment_qr(test_id, save_path=sample_qr_path)
    print(f"   Saved QR code image to: {sample_qr_path}", flush=True)
    
    print("2. Scanning / Decoding QR image...", flush=True)
    decoded_id = identifier.decode_shipment_qr(sample_qr_path)
    print(f"   Decoded Shipment_ID: {decoded_id}", flush=True)
    assert decoded_id == test_id, f"Decoded ID {decoded_id} did not match expected {test_id}!"
    
    print("3. Querying shipment record from database...", flush=True)
    record = identifier.lookup_shipment(decoded_id)
    if record:
        print(f"   Found record! Route: {record.get('Route_ID')}, Supplier: {record.get('Supplier_ID')}, "
              f"Product: {record.get('Product_Category')}, Distance: {record.get('Distance_km')} km")
        print("   [SUCCESS] QR / Barcode identification pipeline verified.", flush=True)
    else:
        print("   Record not found in database!", flush=True)

if __name__ == "__main__":
    test_qr_module()
