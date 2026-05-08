from collections import Counter

from langsmith import Client, traceable

from config import load_config
from qa_pairs import SAMPLE_QUESTIONS
from rag_utils import (
    build_prompt_hub_names,
    build_vectorstore,
    create_remote_chat_client,
    get_prompt_templates,
    get_prompt_version,
    run_prompt_with_retriever,
)


def push_prompts_to_hub(client: Client, prompt_names: dict[str, str], prompts: dict) -> None:
    for version, prompt in prompts.items():
        name = prompt_names[version]
        try:
            url = client.push_prompt(
                name,
                object=prompt,
                description=f"Day 22 {version.upper()} prompt version",
            )
            print(f"Pushed {version.upper()} -> {url}")
        except Exception as exc:
            print(f"Prompt push skipped for {name}: {exc}")


def pull_prompts_from_hub(client: Client, prompt_names: dict[str, str], local_prompts: dict) -> dict:
    pulled = {}
    for version, name in prompt_names.items():
        try:
            pulled[version] = client.pull_prompt(name)
            print(f"Pulled {name} from Prompt Hub")
        except Exception as exc:
            pulled[version] = local_prompts[version]
            print(f"Using local fallback for {name}: {exc}")
    return pulled


@traceable(name="ab-rag-query", tags=["ab-test", "step2"])
def ask_ab(retriever, llm, prompt, question: str, version: str) -> dict:
    result = run_prompt_with_retriever(retriever, llm, prompt, question)
    result["version"] = version
    return result


def main() -> None:
    config = load_config()
    client = Client(api_key=config.langsmith_api_key)
    local_prompts = get_prompt_templates()
    prompt_names = build_prompt_hub_names(config.langsmith_project)

    print("=" * 60)
    print("  Step 2: Prompt Hub A/B Routing")
    print("=" * 60)

    push_prompts_to_hub(client, prompt_names, local_prompts)
    prompts = pull_prompts_from_hub(client, prompt_names, local_prompts)

    vectorstore = build_vectorstore(config)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = create_remote_chat_client(config)

    counts = Counter()
    for index, question in enumerate(SAMPLE_QUESTIONS, start=1):
        request_id = f"req-{index:04d}"
        version = get_prompt_version(request_id)
        result = ask_ab(retriever, llm, prompts[version], question, version)
        counts[version] += 1
        print(f"[{index:02d}] [prompt-{version}] {result['question']}")

    print(f"\nRouting summary: v1={counts['v1']} v2={counts['v2']}")


if __name__ == "__main__":
    main()
