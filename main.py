import cv2
import time
from detector import detect_posture, analyze_posture
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from calculations import trimmed_mean, iq_range
import numpy as np
from colorama import Fore
from notifyer import send_notification



def main_loop(rolling_average_len = 5, cva_cutoff = 19, capture_interval = .05, notification_interval = 20):
    
    # Wait until at least half the notification interval to send the first notification
    last_notification = time.time() - (notification_interval/2)
    
    try:
        # Initialize the model
        base_options = python.BaseOptions(model_asset_path='models/pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
                base_options=base_options,
                output_segmentation_masks=False)
        detector = vision.PoseLandmarker.create_from_options(options)

        # Initialize the webcam
        cap = cv2.VideoCapture(0)

        # Set the capture interval (in seconds)
        capture_interval = 1

        # Set the rolling average length and the cutoff value
        rolling_average_len = 10
        rolling_average = [90]*rolling_average_len
        cva_cutoff = 20

        
        while True:
            # Read the current frame from the webcam
            ret, frame = cap.read()

            # Detect the keypoints
            keypoints = detect_posture(frame, detector)
            if len(keypoints) > 0:
                # Analyze the posture and get the CVA of the rotated keypoints
                cva = analyze_posture(keypoints)
                # Add the CVA to the rolling average
                rolling_average.append(cva)
                # Update the rolling average
                rolling_average = rolling_average[1:]
                
                #Calculate metrics: mean, median, trimmed mean and IQR
                mean = np.mean(rolling_average)
                median = np.median(rolling_average)
                t_mean = trimmed_mean(rolling_average, trim = 0.2)
                iqr = iq_range(rolling_average)

                # Print the metrics
                if t_mean < cva_cutoff:
                    # Check if enough time has passed since the last notification
                    if time.time() - last_notification > notification_interval:
                        send_notification(f"Alerta! CVA: {t_mean:.1f}")
                        last_notification = time.time()
                    print(Fore.RED + f"Mean: {mean:.2f} Median: {median:.2f} Trimmed mean: {t_mean:.2f} IQR: {iqr:.2f}   ", end="\r")
                else:
                    print(Fore.WHITE + f"Mean: {mean:.2f} Median: {median:.2f} Trimmed mean: {t_mean:.2f} IQR: {iqr:.2f}   ", end="\r")
                
                time.sleep(capture_interval)

    except KeyboardInterrupt:
        
        print("\nFinalizando ejecución...")
        # Release the webcam
        cap.release()
        

print("Ejecutando. Presiona 'Ctrl+C' para salir.")
main_loop(notification_interval=5)
print("Finalizado.")