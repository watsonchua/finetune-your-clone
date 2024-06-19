# Deploying the Telegram Bot

This README file is specifically for hosting the Telgram bot.

## Setting up the enviroment

1. Create a new virtual environment for preprocessing using the command `python3 -m venv path_to_your_env`. E.g. `python3 -m venv ~/environments/telegram_bot`. 
2. Activate the environment. `source path_to_your_env/bin/activate`
3. Change directory to the finetuning folder using `cd telegram_bot`. You should be in the folder `finetune-your-clone/telegram_bot` now.
4. Install the necessary packages: `pip install -r requirements.txt`

## Hosting the Telegram Bot
1. Get your Telegram Bot API Key from BotFather in Telegram by following [this guide](https://core.telegram.org/bots/features#creating-a-new-bot).
2. Create a `.env` in the root folder of this project (i.e. `finetune-your-clone/.env`). 
3. Create two entries in the `.env` file:
```
TELEGRAM_TOKEN = <your_telegram_bot_api_token>
GENERATE_URL = <your_fast_api_url_from_your_model>
```
4. Run `python app.py`
5. Make sure that the service is running all the time. The bot will stop responding if this script is killed.
