"""Day 2 standalone script: send one prompt to the LLM provider.

Run with:  python first_call.py

This script only prints the generated answer. It never prints the API key.
"""

from __future__ import annotations

import model_client

PROMPT = "Explain what an API is in two simple sentences."


def main() -> None:
    try:
        answer = model_client.generate_response(PROMPT)
    except model_client.ConfigurationError as exc:
        print(f"[configuration error] {exc}")
        return
    except model_client.AuthenticationError:
        print("[authentication error] The provider rejected the API key.")
        return
    except model_client.EmptyModelResponseError:
        print("[empty response] The provider returned no text.")
        return
    except model_client.ProviderRequestError:
        print("[provider error] The request to the provider failed.")
        return

    print("Prompt:", PROMPT)
    print("Answer:", answer)


if __name__ == "__main__":
    main()
