
import cv2
from pathlib import Path

# Locate the inspection images relative to this script
BASE_DIR = Path(__file__).resolve().parent
image_folder = BASE_DIR / "inspection_images"

# Find saved JPG images
images = sorted(image_folder.glob("*.jpg"))

if not images:
    print("No images found in inspection_images.")
    raise SystemExit

# Load the latest image
image_path = images[-1]
image = cv2.imread(str(image_path))

if image is None:
    print("Could not load image:", image_path)
    raise SystemExit

print("Inspecting image:", image_path.name)

# Step 1: Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Step 2: Reduce small image noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Step 3: Highlight dark line-like features
blackhat_kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (15, 3)
)

dark_lines = cv2.morphologyEx(
    blurred,
    cv2.MORPH_BLACKHAT,
    blackhat_kernel
)

# Step 4: Create a preliminary feature mask
_, mask = cv2.threshold(
    dark_lines,
    30,
    255,
    cv2.THRESH_BINARY
)

# Step 5: Remove some small isolated regions
kernel = cv2.getStructuringElement(
    cv2.MORPH_ELLIPSE, (3, 3)
)

mask = cv2.morphologyEx(
    mask,
    cv2.MORPH_OPEN,
    kernel
)

# Step 6: Calculate the highlighted pixel ratio
white_pixels = cv2.countNonZero(mask)
total_pixels = mask.shape[0] * mask.shape[1]

defect_ratio = white_pixels / total_pixels

print("Image processing completed!")
print("Highlighted pixel ratio:", round(defect_ratio * 100, 2), "%")

# Step 7: Produce a preliminary screening result
if defect_ratio > 0.02:
    result = "POSSIBLE DEFECT"
else:
    result = "NO OBVIOUS DEFECT"

print("Preliminary inspection result:", result)
print("Note: highlighted lines are not confirmed cracks.")

# Step 8: Display the images
cv2.imshow("Original Image", image)
cv2.imshow("Possible Crack Features", mask)

cv2.waitKey(0)
cv2.destroyAllWindows()
