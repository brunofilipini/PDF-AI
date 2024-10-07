import requests

response = requests.get(
    'https://api-inference.huggingface.co/models/google/flan-t5-xxl',
    headers={'Authorization': f'Bearer YOUR_KEY'}
)

print(response.status_code)
print(response.json())
