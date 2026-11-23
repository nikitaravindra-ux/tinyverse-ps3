"""One-time download of the GLiNER weights (the only step that needs the internet).

    .venv/bin/python -m app.fetch_model

The app itself runs with Hugging Face in offline mode behind the local-only network
guard, so it loads the cached weights and never contacts the Hub.
"""
from app.config import GLINER_MODEL


def main() -> None:
    from gliner import GLiNER

    GLiNER.from_pretrained(GLINER_MODEL)
    print(f"cached {GLINER_MODEL}")


if __name__ == "__main__":
    main()
