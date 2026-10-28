"""Command line: ask the agent to do something in a CRM.

    python -m agent "What is the status of the SSO login loop case?"      # Hugging Face API (HF_TOKEN or `hf auth login`)
    python -m agent --crm salesforce "List open cases for Acme"
    python -m agent --crm salesforce --check        # read-only connectivity test, no model call
"""
from __future__ import annotations

import argparse
import os
import sys

from .core import Agent
from .crm import CRMError, InMemoryCRM
from .llm import DEFAULT_HF_MODEL, AnthropicLLM, HuggingFaceLLM, LLMError
from .salesforce import SalesforceCRM


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="agent", description="CRM support agent powered by an LLM (Hugging Face Inference API by default)")
    parser.add_argument("prompt", nargs="?", help="what to ask the agent")
    parser.add_argument("--crm", choices=["memory", "salesforce"], default="memory")
    parser.add_argument("--llm", choices=["hf", "anthropic"], default="hf", help="model backend (default: Hugging Face API)")
    parser.add_argument("--model", help=f"model id (default for hf: {DEFAULT_HF_MODEL})")
    parser.add_argument("--provider", default="auto", help="Hugging Face inference provider, e.g. auto, groq, cerebras")
    parser.add_argument("--key-env", help="environment variable holding a provider's own API key (calls that provider directly)")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--max-writes", type=int, default=3, help="most changes one request may make")
    parser.add_argument("--check", action="store_true", help="read-only connectivity test of the CRM (no model call)")
    args = parser.parse_args(argv)

    try:
        crm = SalesforceCRM.from_env() if args.crm == "salesforce" else InMemoryCRM()
        if args.check:
            accounts = crm.find_accounts("a")
            print(f"Connected to {args.crm}: found {len(accounts)} account(s) containing 'a'.")
            for account in accounts[:5]:
                print(f"  {account['Id']}  {account['Name']}")
            return 0
        if not args.prompt:
            parser.error("give a prompt, or use --check")
        if args.llm == "hf":
            key = os.environ.get(args.key_env) if args.key_env else None
            if args.key_env and not key:
                parser.error(f"environment variable {args.key_env} is not set")
            llm = HuggingFaceLLM(model=args.model or DEFAULT_HF_MODEL, provider=args.provider, token=key)
        else:
            llm = AnthropicLLM(model=args.model or "claude-opus-5-5")
        result = Agent(llm, crm, args.max_steps, args.max_writes).run(args.prompt)
    except CRMError as error:
        print(f"CRM error: {error}", file=sys.stderr)
        return 2
    except LLMError as error:
        print(f"Model error: {error}", file=sys.stderr)
        return 3

    for call in result.tool_calls:
        note = f"  BLOCKED by {call.blocked_by}" if call.blocked_by else ("  (error)" if call.is_error else "")
        print(f"  tool: {call.name}({call.input}){note}")
    print(f"\n{result.final_text}\n")
    print(f"[{result.steps} step(s), {result.writes} change(s), {result.input_tokens} in / {result.output_tokens} out tokens]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
