from src import llm_core


def test_reva_local_ollama_disables_mmap():
    assert (
        llm_core._ollama_use_mmap_override(
            "http://host.docker.internal:11434"
        )
        is False
    )


def test_loopback_ollama_disables_mmap():
    assert (
        llm_core._ollama_use_mmap_override(
            "http://127.0.0.1:11434/api/chat"
        )
        is False
    )


def test_remote_ollama_keeps_server_default_mmap_policy():
    assert (
        llm_core._ollama_use_mmap_override(
            "https://ollama.com"
        )
        is None
    )

    assert (
        llm_core._ollama_use_mmap_override(
            "http://remote-gpu.example:11434"
        )
        is None
    )


def test_builder_emits_explicit_false_only_when_requested():
    base = dict(
        model="deepseek-r1:14b-32k",
        messages=[
            {
                "role": "user",
                "content": "test",
            }
        ],
        temperature=0.0,
        max_tokens=16,
        num_ctx=4096,
    )

    local_payload = llm_core._build_ollama_payload(
        **base,
        use_mmap=False,
    )

    default_payload = llm_core._build_ollama_payload(
        **base,
    )

    assert (
        local_payload["options"]["use_mmap"]
        is False
    )

    assert (
        "use_mmap"
        not in default_payload["options"]
    )
