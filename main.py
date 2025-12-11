import os
from src.running import process_dataset

try:
    # optional: auto-load .env for local development
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


def _build_config_from_env():
    return {
        "ACCESS_TOKEN": os.getenv("VNPT_ACCESS_TOKEN"),
        "TOKEN_ID": os.getenv("VNPT_TOKEN_ID"),
        "TOKEN_KEY": os.getenv("VNPT_TOKEN_KEY"),
        "MODEL_NAME": os.getenv("MODEL_NAME", "vnptai-hackathon-small"),
        "CHAT_PROVIDER": os.getenv("CHAT_PROVIDER", "vnpt"),
        "SLEEP_TIME": int(os.getenv("SLEEP_TIME", "65")),
        # Add other relevant keys from async_main if they are needed for sync mode
    }


if __name__ == "__main__":
    config = _build_config_from_env()
    process_dataset(
        input_file='E:\\VNPT_AI_Water_Margin\\data\\test.json',
        output_file=f'pred/test_{config.get("CHAT_PROVIDER", "")}.csv',
        config=config,
        mode='test'
    )

    # To run on validation set instead, call with a val file:
    # process_dataset(
    #     input_file='E:\\VNPT_AI_Water_Margin\\data\\val.json',
    #     output_file=f'pred/val_{config.get("CHAT_PROVIDER", "")}.csv',
    #     config=config,
    #     mode='valid'
    # )