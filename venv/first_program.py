from openai import OpenAI
with open('/Users/davidortega/DavidProject/venv/openaiapikey.txt', 'r') as file:
    api_key = file.read().strip()
client = OpenAI(api_key=api_key)

client.api_key = api_key
prompt = input("Enter the prompt for the image: ")
response = client.images.generate(
    model="dall-e-3",
    prompt=prompt,
    size="1024x1024",
    quality="standard",
    n=1,
)

print(response.data[0].url)