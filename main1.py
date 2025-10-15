from ultralytics import YOLO
import cv2

# Load model and image
model = YOLO('runs/detect/train2/weights/best.pt')
image = cv2.imread('images/numbers/test/7.png')

# Run inference
results = model(image)[0]  # Get the first result (for a single image)

# Draw results on the image
annotated_image = results.plot()  # This uses Ultralytics' built-in plotting

# Display the image
cv2.imshow("YOLO Custom Detection", annotated_image)
cv2.waitKey(0)  # Wait for a key press to close
cv2.destroyAllWindows()

# Optionally, save it
cv2.imwrite('my_custom_result.jpg', annotated_image)