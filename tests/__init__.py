from dotenv import load_dotenv

from .utils.register import (
    register_all_embeddings_providers,
    register_all_model_providers,
)


load_dotenv()
register_all_model_providers()
register_all_embeddings_providers()
