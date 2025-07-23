from flask import Flask, request, render_template_string
from openai import OpenAI

app = Flask(__name__)
with open('/Users/davidortega/DavidProject/venv/openaiapikey.txt', 'r') as file:
    openai_api_key = file.read().strip()

client = OpenAI(api_key=openai_api_key)
html_template = """
<!doctype html>
<html>
    <head>
        <title>Chat with GPT</title>
        <style>
            body {
                font-family: Arial, sans-serif;
            }
        </style>
    </head>
    <body>
        <h1>Chat with GPT</h1>
        <form method="post">
            <label for="user_input">You:</label>
            <input type="text" id="user_input" name="user_input">
            <input type="submit" value="Send">
        </form>
        <p>Assistant: {{ response }}</p>
    </body>
</html>
"""

@app.route('/', methods=['GET', 'POST'])
def chat():
    response = ""
    if request.method == 'POST':
        user_input = request.form['user_input']
        if user_input.lower() != 'exit':
            ai_response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_input}
                ]
            )
            response = ai_response.choices[0].message.content
    return render_template_string(html_template, response=response)

if __name__ == '__main__':
    app.run(debug=True)
