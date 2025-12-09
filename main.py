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
        "ACCESS_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6ImRkMzhkNGEzLWI2YmUtNDBkNC05NjY5LWQ5NDY4YzM0YmEwMCIsInN1YiI6IjIwNDc3MjU5LWQxMmEtMTFmMC1hMDI3LWJiODI2MDRmMjU4NSIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ2JhY2gyMDA4QGdtYWlsLmNvbSIsInNjb3BlIjpbInJlYWQiXSwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3QiLCJuYW1lIjoibmdiYWNoMjAwOEBnbWFpbC5jb20iLCJ1dWlkX2FjY291bnQiOiIyMDQ3NzI1OS1kMTJhLTExZjAtYTAyNy1iYjgyNjA0ZjI1ODUiLCJhdXRob3JpdGllcyI6WyJVU0VSIiwiVFJBQ0tfMiJdLCJqdGkiOiI2ZWNjOGYxYy03ZmU2LTRmZjMtYjA3OC1kNTY2YjJkZmI0M2EiLCJjbGllbnRfaWQiOiJhZG1pbmFwcCJ9.3iO1aLyW5JJ-_Y9ZQ1i2U71jHUrly_IDa-H2gExtO-vyRuPCl1b08MvSyxeQ0PaFtNRy29tBKpgudV9TQ2R8lh3a1dlBJCpJWHpE9RqeiyLRMurrqr_ofUbs8kq8ocQnHkzWTb-yt2fJQqzYaDOPAHQ2uCwJqNVSEDiOpQdUkDEKAKvu_-bweIckIm9QFwLMhfu6Q4PNhyLnoIpfGuGwf_1Y2P5T-rNAQIkIwZafMTpT4Mvin8wbmmlHVLUK719JLsZMyC33mO1PnK9F2tjC0hjsF9HARcu_LFfoqvyvZAxgBhrrTffIFwOsW-O1HLDslFAL0kyWSYV-AkTUPxhCfQ",
        "TOKEN_ID": "4525a88b-e7db-4f0c-e063-62199f0a3a11",
        "TOKEN_KEY": "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAJivUf+ovda9JbCzUkcrs7mHHaNMDmDJK+Hz0yexuxuGjUztbqmfdCIPJGBaGMkRscI4GYtx5p09WCpigc/QkdkCAwEAAQ==",
        "MODEL_NAME": "vnptai-hackathon-small",

        # "ACCESS_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6ImRkMzhkNGEzLWI2YmUtNDBkNC05NjY5LWQ5NDY4YzM0YmEwMCIsInN1YiI6IjIwNDc3MjU5LWQxMmEtMTFmMC1hMDI3LWJiODI2MDRmMjU4NSIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ2JhY2gyMDA4QGdtYWlsLmNvbSIsInNjb3BlIjpbInJlYWQiXSwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3QiLCJuYW1lIjoibmdiYWNoMjAwOEBnbWFpbC5jb20iLCJ1dWlkX2FjY291bnQiOiIyMDQ3NzI1OS1kMTJhLTExZjAtYTAyNy1iYjgyNjA0ZjI1ODUiLCJhdXRob3JpdGllcyI6WyJVU0VSIiwiVFJBQ0tfMiJdLCJqdGkiOiI2ZWNjOGYxYy03ZmU2LTRmZjMtYjA3OC1kNTY2YjJkZmI0M2EiLCJjbGllbnRfaWQiOiJhZG1pbmFwcCJ9.3iO1aLyW5JJ-_Y9ZQ1i2U71jHUrly_IDa-H2gExtO-vyRuPCl1b08MvSyxeQ0PaFtNRy29tBKpgudV9TQ2R8lh3a1dlBJCpJWHpE9RqeiyLRMurrqr_ofUbs8kq8ocQnHkzWTb-yt2fJQqzYaDOPAHQ2uCwJqNVSEDiOpQdUkDEKAKvu_-bweIckIm9QFwLMhfu6Q4PNhyLnoIpfGuGwf_1Y2P5T-rNAQIkIwZafMTpT4Mvin8wbmmlHVLUK719JLsZMyC33mO1PnK9F2tjC0hjsF9HARcu_LFfoqvyvZAxgBhrrTffIFwOsW-O1HLDslFAL0kyWSYV-AkTUPxhCfQ",
        # "TOKEN_KEY": "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAIKOkkLUohQIfhfL42rRUyJqc9GVrj42P6/Z9EHl/NnRM19yI7TnVrhXK9pzBhBNS4L6Ks6ohcrlIwqf2CE6rr0CAwEAAQ==",
        # "TOKEN_ID": "4525a84b-002f-2031-e063-62199f0af9db",
        # "MODEL_NAME": "vnptai-hackathon-large",

        # "ACCESS_TOKEN": "eyJhbGciOiJSUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0cmFuc2FjdGlvbl9pZCI6ImRkMzhkNGEzLWI2YmUtNDBkNC05NjY5LWQ5NDY4YzM0YmEwMCIsInN1YiI6IjIwNDc3MjU5LWQxMmEtMTFmMC1hMDI3LWJiODI2MDRmMjU4NSIsImF1ZCI6WyJyZXN0c2VydmljZSJdLCJ1c2VyX25hbWUiOiJuZ2JhY2gyMDA4QGdtYWlsLmNvbSIsInNjb3BlIjpbInJlYWQiXSwiaXNzIjoiaHR0cHM6Ly9sb2NhbGhvc3QiLCJuYW1lIjoibmdiYWNoMjAwOEBnbWFpbC5jb20iLCJ1dWlkX2FjY291bnQiOiIyMDQ3NzI1OS1kMTJhLTExZjAtYTAyNy1iYjgyNjA0ZjI1ODUiLCJhdXRob3JpdGllcyI6WyJVU0VSIiwiVFJBQ0tfMiJdLCJqdGkiOiI2ZWNjOGYxYy03ZmU2LTRmZjMtYjA3OC1kNTY2YjJkZmI0M2EiLCJjbGllbnRfaWQiOiJhZG1pbmFwcCJ9.3iO1aLyW5JJ-_Y9ZQ1i2U71jHUrly_IDa-H2gExtO-vyRuPCl1b08MvSyxeQ0PaFtNRy29tBKpgudV9TQ2R8lh3a1dlBJCpJWHpE9RqeiyLRMurrqr_ofUbs8kq8ocQnHkzWTb-yt2fJQqzYaDOPAHQ2uCwJqNVSEDiOpQdUkDEKAKvu_-bweIckIm9QFwLMhfu6Q4PNhyLnoIpfGuGwf_1Y2P5T-rNAQIkIwZafMTpT4Mvin8wbmmlHVLUK719JLsZMyC33mO1PnK9F2tjC0hjsF9HARcu_LFfoqvyvZAxgBhrrTffIFwOsW-O1HLDslFAL0kyWSYV-AkTUPxhCfQ",
        # "TOKEN_KEY": "MFwwDQYJKoZIhvcNAQEBBQADSwAwSAJBAJDSVhVUIP2GhUW5uewM9DpMRIulH4phd9Jj3obq8JslRMp7SLtSnVcqoabZJKX70shW/nRtSz+3lcv3jNIGd/sCAwEAAQ==",
        # "TOKEN_ID": "4525a842-6ca4-553c-e063-62199f0a1086",
        # "MODEL_NAME": "vnptai-hackathon-embedding",

        "PROVIDER": "vnpt",

        "SLEEP_TIME": int(os.getenv("SLEEP_TIME", "65")),
        # "MODEL_NAME": "gemma3:270m",
        # "PROVIDER": "ollama",
    }


if __name__ == "__main__":
    config = _build_config_from_env()
    process_dataset(
        input_file='E:\\VNPT_AI_Water_Margin\\data\\val.json',
        output_file=f'pred/val_{config.get("PROVIDER", "")}.csv',
        config=config,
        mode='valid'
    )

    # To run on validation set instead, call with a val file:
    # process_dataset(input_file='data/val.json', output_file='pred/val_predictions.csv', config=config)