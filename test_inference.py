from inference_sdk import InferenceHTTPClient
# Initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://serverless.roboflow.com",
    api_key="oo64LiCRWeP5aY7CKU27"  # Your Roboflow API key
)
# Correct image path
image_path = "testvideo/ir6.mp4"  # or "test_images\\rgb1.png"
# Run inference
result = CLIENT.infer(image_path, model_id="drone-fsixa/2")  # Your model ID
# Print the results
print(result)