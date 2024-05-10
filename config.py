import os
import dotenv


def create_config() -> dict[str, str]:
    config = {
        **os.environ,
        **dotenv.dotenv_values("../.env"),
        **dotenv.dotenv_values("../.env.local"),
        **dotenv.dotenv_values("../.env.development.local"),
        **dotenv.dotenv_values(".env"),
        **dotenv.dotenv_values(".env.local"),
        **dotenv.dotenv_values(".env.development.local"),
    }
    return config

config = create_config()
