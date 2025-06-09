import os
from datetime import datetime
import pytz
from telegram import Bot

token = os.getenv('TELEGRAM_BOT_TOKEN')
chat_id = os.getenv('TELEGRAM_CHAT_ID')
timezone = pytz.timezone('Africa/Lagos')
now = datetime.now(timezone)
hour = now.hour

if 0 <= hour < 9:
    message = 'Good morning! ☀️ Have a great day!'
elif 18 <= hour < 24:
    message = 'Good night! 🌙 Sleep well!'
else:
    print('No message to send at this hour.')
    exit(0)

print(f'Sending message: {message}')
bot = Bot(token=token)
bot.send_message(chat_id=chat_id, text=message)
print("done")