# import cv2
# from cvzone.HandTrackingModule import HandDetector
# from cvzone.ClassificationModule import Classifier
# import numpy as np
# import math
# cap = cv2.VideoCapture(0)
# detector = HandDetector(maxHands=1)
# classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
# offset = 20
# imgSize = 300
# folder = "Data/C"
# counter = 0
# labels = [" Hello", "Help", "Love You","Nice","No","Sad","Sorry","Thanks","Yes"]
# while True:
#     success, img = cap.read()
#     imgOutput = img.copy()
#     hands, img = detector.findHands(img)
#     if hands:
#         hand = hands[0]
#         x, y, w, h = hand['bbox']
#         imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
#         imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
#         imgCropShape = imgCrop.shape
#         aspectRatio = h / w
#         if aspectRatio > 1:
#             k = imgSize / h
#             wCal = math.ceil(k * w)
#             imgResize = cv2.resize(imgCrop, (wCal, imgSize))
#             imgResizeShape = imgResize.shape
#             wGap = math.ceil((imgSize - wCal) / 2)
#             imgWhite[:, wGap:wCal + wGap] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#             print(prediction, index)
#         else:
#             k = imgSize / w
#             hCal = math.ceil(k * h)
#             imgResize = cv2.resize(imgCrop, (imgSize, hCal))
#             imgResizeShape = imgResize.shape
#             hGap = math.ceil((imgSize - hCal) / 2)
#             imgWhite[hGap:hCal + hGap, :] = imgResize
#             prediction, index = classifier.getPrediction(imgWhite, draw=False)
#         cv2.rectangle(imgOutput, (x - offset, y - offset-50),
#                       (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
#         cv2.putText(imgOutput, labels[index], (x, y -26), cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
#         cv2.rectangle(imgOutput, (x-offset, y-offset),
#                       (x + w+offset, y + h+offset), (255, 0, 255), 4)
#         cv2.imshow("ImageCrop", imgCrop)
#         cv2.imshow("ImageWhite", imgWhite)
#     cv2.imshow("Image", imgOutput)
#     cv2.waitKey(1)


import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import pyttsx3
import time
import logging
from collections import deque

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sign_language_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize text-to-speech engine
engine = pyttsx3.init()
engine.setProperty('rate', 150)
engine.setProperty('volume', 1.0)

# Configuration
CONFIDENCE_THRESHOLD = 0.7  # Only accept predictions with confidence above this
PREDICTION_HISTORY_LENGTH = 5  # Number of frames to consider for smoothing
PREDICTION_COOLDOWN = 3  # seconds between voice outputs

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

offset = 20
imgSize = 300
labels = ["Hello", "Help", "Love You", "Nice", "No", "Sad", "Sorry", "Thanks", "Yes"]

# Prediction smoothing variables
prediction_history = deque(maxlen=PREDICTION_HISTORY_LENGTH)
last_prediction_time = 0
last_spoken_label = ""

# Performance tracking
frame_count = 0
start_time = time.time()

def get_smoothed_prediction(current_pred, current_conf):
    """Apply smoothing to predictions using a history buffer"""
    prediction_history.append((current_pred, current_conf))
    
    # Get most frequent prediction with highest average confidence
    pred_counts = {}
    conf_sums = {}
    
    for pred, conf in prediction_history:
        pred_counts[pred] = pred_counts.get(pred, 0) + 1
        conf_sums[pred] = conf_sums.get(pred, 0) + conf
    
    # Find prediction with highest count and confidence
    if pred_counts:
        best_pred = max(pred_counts, key=lambda x: (pred_counts[x], conf_sums[x]))
        avg_conf = conf_sums[best_pred] / pred_counts[best_pred]
        return best_pred, avg_conf
    return current_pred, current_conf

try:
    while True:
        success, img = cap.read()
        if not success:
            logger.error("Failed to capture image from camera")
            break
            
        frame_count += 1
        imgOutput = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']
            
            # Prepare image for classification
            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
            imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            
            try:
                aspectRatio = h / w
                
                if aspectRatio > 1:
                    k = imgSize / h
                    wCal = math.ceil(k * w)
                    imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                    wGap = math.ceil((imgSize - wCal) / 2)
                    imgWhite[:, wGap:wCal + wGap] = imgResize
                else:
                    k = imgSize / w
                    hCal = math.ceil(k * h)
                    imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                    hGap = math.ceil((imgSize - hCal) / 2)
                    imgWhite[hGap:hCal + hGap, :] = imgResize
                
                # Get prediction and confidence
                prediction, index = classifier.getPrediction(imgWhite, draw=False)
                current_label = labels[index]
                confidence = prediction[index]
                
                logger.debug(f"Raw prediction: {current_label} (Confidence: {confidence:.2f})")
                
                # Apply smoothing and confidence threshold
                smoothed_label, smoothed_conf = get_smoothed_prediction(current_label, confidence)
                
                if smoothed_conf >= CONFIDENCE_THRESHOLD:
                    current_time = time.time()
                    
                    # Display prediction
                    cv2.rectangle(imgOutput, (x - offset, y - offset-50),
                                 (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)
                    cv2.putText(imgOutput, f"{smoothed_label} {smoothed_conf:.1f}%", (x, y -26), 
                               cv2.FONT_HERSHEY_COMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.rectangle(imgOutput, (x-offset, y-offset),
                                 (x + w+offset, y + h+offset), (255, 0, 255), 4)
                    
                    # Voice output logic
                    if (smoothed_label != last_spoken_label or 
                        (current_time - last_prediction_time > PREDICTION_COOLDOWN)):
                        logger.info(f"Speaking: {smoothed_label} (Confidence: {smoothed_conf:.2f})")
                        engine.say(smoothed_label)
                        engine.runAndWait()
                        last_prediction_time = current_time
                        last_spoken_label = smoothed_label
                
                else:
                    logger.debug(f"Low confidence prediction: {smoothed_label} ({smoothed_conf:.2f})")
                    cv2.putText(imgOutput, "Low Confidence", (x, y -26), 
                               cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
                
            except Exception as e:
                logger.error(f"Error processing hand image: {str(e)}")
                cv2.putText(imgOutput, "Processing Error", (x, y -26), 
                           cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        
        # Calculate and display FPS
        elapsed_time = time.time() - start_time
        fps = frame_count / elapsed_time
        cv2.putText(imgOutput, f"FPS: {fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Sign Language Detection", imgOutput)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logger.info("User requested exit")
            break

except Exception as e:
    logger.critical(f"Unexpected error: {str(e)}", exc_info=True)

finally:
    # Calculate final performance metrics
    total_time = time.time() - start_time
    logger.info(f"Session ended. Frames processed: {frame_count}, Avg FPS: {frame_count/total_time:.1f}")
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()