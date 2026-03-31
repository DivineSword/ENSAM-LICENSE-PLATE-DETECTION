import os
import cv2
import numpy as np
import torch

from lpr.camera.camera import start_camera, get_frame
from lpr.detection.detection import detect_objects
from lpr.ocr.ocr import read_text
from lpr.utils.text_utils import clean_text

# ---------------------------------------------------------------------------
# Configuration — edit these paths before running
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

YOLO1_PATH = os.path.join(BASE_DIR, "models", "license_plate_detector.pt")
YOLO2_PATH = os.path.join(BASE_DIR, "models", "yolo2.pt")            # add when ready
CNN_PATH   = os.path.join(BASE_DIR, "models", "cnn_model.pt")      # add when ready


class LicensePlateRecognitionSystem:
    def __init__(
        self,
        yolo1_model_path: str = YOLO1_PATH,
        yolo2_model_path: str = YOLO2_PATH,
        cnn_model_path:   str = CNN_PATH,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Initialise the License Plate Recognition System.

        Args:
            yolo1_model_path: Path to YOLO_1 model (license plate detection).
            yolo2_model_path: Path to YOLO_2 model (plate zone classification).
            cnn_model_path:   Path to custom CNN model (Arabic letter recognition).
            device:           'cuda' for GPU or 'cpu'.
        """
        print("Initialising LPR System...")
        self.device = device

        # YOLO_1 — plate detection
        try:
            from ultralytics import YOLO
            self.yolo1 = YOLO(yolo1_model_path)
            print("  YOLO_1 (Plate Detection) loaded")
        except Exception as exc:
            print(f"  Failed to load YOLO_1: {exc}")
            self.yolo1 = None

        # YOLO_2 — zone classification
        try:
            from ultralytics import YOLO
            self.yolo2 = YOLO(yolo2_model_path)
            print("  YOLO_2 (Zone Classification) loaded")
        except Exception as exc:
            print(f"  Failed to load YOLO_2 (optional): {exc}")
            self.yolo2 = None

        # Custom CNN — Arabic letter recognition
        try:
            self.cnn_model = torch.load(cnn_model_path, map_location=device)
            self.cnn_model.to(device)
            self.cnn_model.eval()
            print("  Custom CNN (Arabic Recognition) loaded")
        except Exception as exc:
            print(f"  Failed to load CNN (optional): {exc}")
            self.cnn_model = None

        # EasyOCR — digits and text
        try:
            import easyocr
            self.ocr_reader = easyocr.Reader(["en", "ar"], gpu=(device == "cuda"))
            print("  EasyOCR initialised")
        except Exception as exc:
            print(f"  Failed to initialise EasyOCR: {exc}")
            self.ocr_reader = None

        # Arabic character class map
        self.arabic_classes = {
            0: "ا", 1: "ب", 2: "ت", 3: "ث", 4: "ج", 5: "ح", 6: "خ", 7: "د",
            8: "ذ", 9: "ر", 10: "ز", 11: "س", 12: "ش", 13: "ص", 14: "ض",
            15: "ط", 16: "ظ", 17: "ع", 18: "غ", 19: "ف", 20: "ق", 21: "ك",
            22: "ل", 23: "م", 24: "ن", 25: "ه", 26: "و", 27: "ي",
        }
        # Cache the last successfully read plate text for display between OCR frames
        self._last_plate_text = ""

    def detect_license_plates(self, frame):
        """Step 1: detect plate bounding boxes using YOLO_1."""
        if self.yolo1 is None:
            return []
        plates = []
        for result in self.yolo1(frame, conf=0.4):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                plates.append((x1, y1, x2, y2, float(box.conf[0])))
        return plates

    def classify_plate_zones(self, plate_roi):
        """Step 2: split plate into digit / Arabic / country zones via YOLO_2."""
        if self.yolo2 is None:
            return {}
        zone_names = {0: "digits", 1: "arabic", 2: "country"}
        zones = {}
        for result in self.yolo2(plate_roi, conf=0.5):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                zones[zone_names.get(int(box.cls[0]), "unknown")] = (x1, y1, x2, y2)
        return zones

    def recognize_arabic_letters(self, letter_roi):
        """Step 3: classify a single Arabic letter via the custom CNN."""
        if self.cnn_model is None:
            return None
        try:
            img = cv2.resize(letter_roi, (32, 32)).astype(np.float32) / 255.0
            img = torch.tensor(np.transpose(img, (2, 0, 1))).unsqueeze(0).to(self.device)
            with torch.no_grad():
                pred = torch.argmax(self.cnn_model(img), dim=1).item()
            return self.arabic_classes.get(pred, "?")
        except Exception as exc:
            print(f"Error recognising letter: {exc}")
            return None

    def extract_digits_ocr(self, digits_roi) -> str:
        """Step 4: extract digit text from the digit zone via EasyOCR."""
        if self.ocr_reader is None:
            return ""
        try:
            return clean_text("".join(self.ocr_reader.readtext(digits_roi, detail=0)))
        except Exception as exc:
            print(f"Error extracting digits: {exc}")
            return ""

    def extract_arabic_zone(self, arabic_roi) -> str:
        """Step 5: extract Arabic text from the Arabic zone."""
        try:
            if self.ocr_reader is not None:
                text = "".join(
                    self.ocr_reader.readtext(arabic_roi, detail=0, languages=["ar"])
                )
                if text:
                    return text
        except Exception as exc:
            print(f"Error extracting Arabic: {exc}")
        return ""

    def extract_country_code(self, country_roi) -> str:
        """Step 6: extract the country code from the country zone."""
        if self.ocr_reader is None:
            return ""
        try:
            return clean_text(
                "".join(
                    self.ocr_reader.readtext(
                        country_roi, detail=0, languages=["en", "ar"]
                    )
                )
            )
        except Exception as exc:
            print(f"Error extracting country code: {exc}")
            return ""

    def process_license_plate(self, plate_roi) -> dict:
        """Full pipeline: zones → OCR → combined result.

        If YOLO_2 is not available, falls back to running EasyOCR directly on
        the full plate ROI so text is still extracted without zone splitting.
        """
        result = {"digits": "", "arabic": "", "country": "", "full_plate": ""}
        zones = self.classify_plate_zones(plate_roi)

        if zones:
            # YOLO_2 available — precise per-zone extraction
            for key, extractor in [
                ("digits",  self.extract_digits_ocr),
                ("arabic",  self.extract_arabic_zone),
                ("country", self.extract_country_code),
            ]:
                if key in zones:
                    x1, y1, x2, y2 = zones[key]
                    result[key] = extractor(plate_roi[y1:y2, x1:x2])

            result["full_plate"] = (
                f"{result['digits']} {result['arabic']} {result['country']}".strip()
            )
        else:
            # YOLO_2 not available — run EasyOCR on the whole plate as a fallback
            if self.ocr_reader is not None:
                try:
                    raw = self.ocr_reader.readtext(plate_roi, detail=0)
                    result["full_plate"] = " ".join(raw).strip()
                except Exception as exc:
                    print(f"Fallback OCR failed: {exc}")

        return result

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self):
        """Real-time license plate recognition loop. Press Q to quit."""
        cap = start_camera()
        frame_count = 0
        print("System running — press Q to quit.")

        while True:
            frame = get_frame(cap)
            if frame is None:
                break

            frame_count += 1
            plates = self.detect_license_plates(frame)

            for x1, y1, x2, y2, conf in plates:
                plate_roi = frame[y1:y2, x1:x2]

                # Run OCR every 10 frames to keep CPU load manageable
                if frame_count % 10 == 0:
                    plate_result = self.process_license_plate(plate_roi)
                    if plate_result["full_plate"]:
                        self._last_plate_text = plate_result["full_plate"]
                        print(f"Plate: {plate_result['full_plate']}  (conf {conf:.2f})")

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Show confidence above the box
                cv2.putText(
                    frame,
                    f"Plate ({conf:.2f})",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                # Show last recognised text below the box (persists between OCR frames)
                if self._last_plate_text:
                    cv2.putText(
                        frame,
                        self._last_plate_text,
                        (x1, y2 + 24),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 200, 255),
                        2,
                    )

            cv2.imshow("License Plate Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("System shutdown complete.")


def main():
    lpr = LicensePlateRecognitionSystem()
    lpr.run()


if __name__ == "__main__":
    main()
