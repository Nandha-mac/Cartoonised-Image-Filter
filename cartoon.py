import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the image
image = cv2.imread("input.jpg")
image = cv2.resize(image, (600, 600))  # Resize for better performance

# Step 1: Apply Bilateral Filter to reduce noise but keep edges sharp
filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)

# Step 2: Convert to grayscale and apply median blur
gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)
blurred_gray = cv2.medianBlur(gray, 7)

# Step 3: Edge detection using adaptive thresholding
edges = cv2.adaptiveThreshold(blurred_gray, 255,
                              cv2.ADAPTIVE_THRESH_MEAN_C,
                              cv2.THRESH_BINARY, blockSize=9, C=2)

# Step 4: Color quantization using K-Means clustering
data = np.float32(image).reshape((-1, 3))
k = 8  # Number of colors
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(data)
quantized = kmeans.cluster_centers_[kmeans.labels_].reshape(image.shape)
quantized = np.uint8(quantized)

# Step 5: Apply Gaussian blur to smooth the quantized image
blurred_quantized = cv2.GaussianBlur(quantized, (7, 7), 0)

# Step 6: Combine edges with the color image
edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
cartoon = cv2.bitwise_and(blurred_quantized, edges_colored)

# Save and display
cv2.imwrite("cartoon_output.jpg", cartoon)
cv2.imshow("Cartoonised Image", cartoon)
cv2.waitKey(0)
cv2.destroyAllWindows()
