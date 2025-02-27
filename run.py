from RobotArmMovement import *
import cv2
import depthai as dai
import os
import datetime
import numpy as np
import threading
from inference_sdk import InferenceHTTPClient

user_pos_number = None
captured_trigger = False 
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
model = RobotArmMovement().to(device)
model.load_state_dict(torch.load("modelV1 (2).pth"))
model.eval()

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="BEizWvAJDl4An1N9l3uX"  # Replace with your actual API key
)

MODEL_ID = "segmentation-9ozwe/3"  # Replace with your model ID

# Folder to save captured images
SAVE_FOLDER = "run"
os.makedirs(SAVE_FOLDER, exist_ok=True)

def preprocess_image(image_path):
    """Resize and normalize the image for AI model input."""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.resize(image, dsize=(160, 90))
    image = image / 255.0
    image = np.transpose(image, (2, 0, 1))
    image_tensor = torch.tensor(image, dtype=torch.float32)
    return image_tensor.unsqueeze(0).to(device)


def get_prediction(image_path):
    """Send image to Roboflow and get predictions."""
    try:
        result = CLIENT.infer(image_path, model_id=MODEL_ID)
        return result  # JSON response
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Color map for each class (RGB)
CLASS_COLORS = {
    'Wall': (238, 130, 238),  # Violet
    'Obstacle': (255, 0, 0),  # Red
    # Add more classes and their corresponding colors here
}

def overlay_segmentation(image, prediction):
    """Overlay segmentation mask onto the original image, using different colors for different classes."""
    if "predictions" not in prediction:
        print("No predictions found.")
        return image

    mask = np.zeros(image.shape[:2], dtype=np.uint8)  # Create an empty mask for storing the segmentation

    # Check the structure of the predictions
    for idx, obj in enumerate(prediction["predictions"]):
        print(f"Prediction {idx}: {obj}")  # Debug each object in predictions

        # Extract the class name and confidence
        class_name = obj.get("class", "Unknown")
        confidence = obj.get("confidence", 0)
        confidence_percentage = int(confidence * 100)

        # Get the corresponding color for this class
        color = CLASS_COLORS.get(class_name, (255, 255, 255))  # Default to white if class is unknown

        # Extract the points from the "points" field
        if "points" in obj:
            points = obj["points"]
            print(f"Points: {points}")  # Debug points

            # Convert the points into a numpy array of integers (for cv2.fillPoly)
            polygon = np.array([(point['x'], point['y']) for point in points], dtype=np.int32)

            # Fill the mask with the polygon for the given class
            cv2.fillPoly(mask, [polygon], color=255)

            # Create a color mask for this class
            color_mask = np.zeros_like(image)
            cv2.fillPoly(color_mask, [polygon], color=color)

            # Smooth the color mask using Gaussian blur
            color_mask = cv2.GaussianBlur(color_mask, (15, 15), 0)

            # Overlay the color mask on the original image using transparency (alpha blending)
            alpha = 0.5  # Transparency level (0 = fully transparent, 1 = opaque)
            image = cv2.addWeighted(image, 1, color_mask, alpha, 0)

            # Add class name and confidence percentage to the image
            text = f"{class_name} ({confidence_percentage}%)"
            text_position = (int(polygon[0][0]), int(polygon[0][1]) - 10)  # Position above the polygon

            # Use Tahoma-like font (OpenCV doesn't support Tahoma directly, using Hershey font as close alternative)
            font = cv2.FONT_HERSHEY_SIMPLEX
            text_color = color  # Text will have the same color as the class

            cv2.putText(image, text, text_position, font, 0.7, text_color, 2)  # Color stays the same

        else:
            print(f"No points data in prediction {idx}. Skipping.")

    return image

def listen_for_pos():
    """Listen for number input in the terminal and trigger image capture."""
    global user_pos_number, captured_trigger
    while True:
        pos = input("\nEnter a number and press Enter to capture: ")
        if pos.isdigit(): 
            user_pos_number = int(pos)
            captured_trigger = True

def to_tensor(distance):
    distance_tensor = torch.tensor(distance, dtype=torch.float32).unsqueeze(0) / 100.0
    distance_tensor = distance_tensor.unsqueeze(0).to(device)
    return distance_tensor

def main():
    global user_pos_number, captured_trigger

    input_thread = threading.Thread(target=listen_for_pos, daemon=True)
    input_thread.start()

    pipeline = dai.Pipeline()

    cam_rgb = pipeline.create(dai.node.ColorCamera)
    cam_rgb.setPreviewSize(1280, 720)  # Set resolution
    cam_rgb.setInterleaved(False)
    cam_rgb.setFps(30)  # Frames per second

    xout = pipeline.create(dai.node.XLinkOut)
    xout.setStreamName("video")
    cam_rgb.preview.link(xout.input)

    image_counter = 1  # To track the number of segmented images

    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name="video", maxSize=4, blocking=False)

        while True:
            frame = queue.get().getCvFrame()
            cv2.imshow("Oak 1 Camera", frame)

            key = cv2.waitKey(1)
            if key == ord(' ') or captured_trigger:  # Space bar to capture
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                img_path = os.path.join(SAVE_FOLDER, f"capture_{timestamp}.jpg")

                # Save the original frame (convert to RGB format)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                cv2.imwrite(img_path, rgb_frame)
                print(f"Image saved: {img_path}")

                # Send image to Roboflow for prediction
                prediction = get_prediction(img_path)

                if prediction:
                    print("Inference Result:", prediction)

                    # Overlay segmentation mask on the image
                    segmented_image = overlay_segmentation(rgb_frame.copy(), prediction)

                    # Save the new segmented image (convert to RGB format)
                    seg_img_path = os.path.join(SAVE_FOLDER, f"image.jpg")
                    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(seg_img_path, segmented_image)
                    print(f"Segmented image saved: {seg_img_path}")

                    # Delete the captured image
                    os.remove(img_path)
                    print(f"Captured image deleted: {img_path}")

                    # Show the image with segmentation overlay
                    cv2.imshow("Segmentation", segmented_image)

                    # Increment image counter for the next image
                    image_counter += 1

                    if user_pos_number is not None:
                        with open(os.path.join(SAVE_FOLDER, f"pos.txt"), "w") as f:
                            f.write(str(user_pos_number))
                        print(f"Pos: {user_pos_number}")
                        user_pos_number = None
                        captured_trigger = False

                    with open("run/pos.txt", "r") as f:
                        with torch.no_grad():
                            image_tensor = preprocess_image("run/image.jpg")
                            distance = float(f.read())
                            distance_tensor = to_tensor(distance)
                            print(distance_tensor)
                            print(image_tensor.shape)
                            predicted_value = model(image_tensor,distance_tensor)
                            predicted_value = torch.clamp(predicted_value, min=0, max=1)
                            predicted_value = predicted_value.cpu().numpy().flatten()
                            predicted_value = np.round(predicted_value.reshape(14, 4) * 180)
                            print(predicted_value)

            if key == ord('q'):  # Quit
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()