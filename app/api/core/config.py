# generative-ai-service/app/api/core/config.py

from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Define the application name with a default value
    app_name: str = "Generative AI Services"
    azure_endpoint_url: str
    azure_deployment_name: str
    azure_openai_api_key: str
    azure_openai_version:str

    class Config:
        # Specify the name of the environment file and enconding to load variables from
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
