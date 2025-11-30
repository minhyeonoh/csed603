import litellm
import base64


def image_to_base64(image_path):
    """
    Encodes an image file to a Base64 string.

    Args:
        image_path (str): The path to the image file.

    Returns:
        str: The Base64 encoded string of the image.
    """
    try:
        with open(image_path, "rb") as image_file:
            # Read the image content as bytes
            image_bytes = image_file.read()
            # Encode the bytes to Base64
            encoded_image_bytes = base64.b64encode(image_bytes)
            # Decode the Base64 bytes to a UTF-8 string
            encoded_image_string = encoded_image_bytes.decode('utf-8')
            return encoded_image_string
    except FileNotFoundError:
        return "Error: Image file not found at " + image_path
    except Exception as e:
        return "An error occurred during encoding: " + str(e)


image_base64 = image_to_base64("2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg")
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What’s in this image?"
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image_base64}"
                },
            },
        ],
    }
]

response = litellm.completion(
    # model="hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    model="ollama/qwen2.5vl:3b",
    messages=messages,
    base_url="http://127.0.0.1:11109",
    # api_key=None,
    # mock_response=self.mock_response,
)
print(response.choices[0].message)
messages.append(response.choices[0].message)
messages.append(
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "Translate your answer in Korean."
            }
        ],
    }
)
response = litellm.completion(
    # model="hosted_vllm/Qwen/Qwen2.5-VL-7B-Instruct-AWQ",
    model="ollama/qwen2.5vl:3b",
    messages=messages,
    base_url="http://127.0.0.1:11109",
    # api_key=None,
    # mock_response=self.mock_response,
)
print(response.choices[0].message)