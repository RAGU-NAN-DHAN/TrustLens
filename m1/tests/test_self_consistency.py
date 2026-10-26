from __future__ import annotations

import pytest

from tests.conftest import FakeGroqClient
from trustlens_m1.config import SelfConsistencyConfig
from trustlens_m1.errors import GenerationError, InvalidInputError
from trustlens_m1.self_consistency import run_self_consistency, score_claim_consistency


@pytest.mark.asyncio
async def test_identical_samples_yield_high_consistency(generator_config, fake_embedding_model):
    client = FakeGroqClient(answers="The Eiffel Tower is in Paris.")
    sc_cfg = SelfConsistencyConfig(sample_count=3, temperature=0.9, max_output_tokens=256)
    result = await run_self_consistency(
        "Where is the Eiffel Tower?",
        generator_config=generator_config,
        sc_config=sc_cfg,
        client=client,
        _embedding_model=fake_embedding_model,
    )
    assert result.sample_count == 3
    assert result.overall_consistency_score > 0.9
    assert result.errors == []


@pytest.mark.asyncio
async def test_dissimilar_samples_yield_lower_consistency(generator_config, fake_embedding_model):
    answers = [
        "The Eiffel Tower is in Paris.",
        "Quantum mechanics describes subatomic particles.",
        "The stock market fell sharply yesterday afternoon.",
        "Bananas are a good source of potassium.",
    ]
    client = FakeGroqClient(answers=answers)
    sc_cfg = SelfConsistencyConfig(sample_count=3, temperature=0.9, max_output_tokens=256)
    result = await run_self_consistency(
        "Where is the Eiffel Tower?",
        generator_config=generator_config,
        sc_config=sc_cfg,
        client=client,
        _embedding_model=fake_embedding_model,
    )
    assert result.overall_consistency_score < 0.9


@pytest.mark.asyncio
async def test_similar_but_not_identical_samples_are_in_between(generator_config, fake_embedding_model):
    answers = [
        "The Eiffel Tower is located in Paris, France.",
        "The Eiffel Tower stands in Paris.",
        "Paris is home to the Eiffel Tower.",
        "You can find the Eiffel Tower in the city of Paris.",
    ]
    client = FakeGroqClient(answers=answers)
    sc_cfg = SelfConsistencyConfig(sample_count=3, temperature=0.9, max_output_tokens=256)
    result = await run_self_consistency(
        "Where is the Eiffel Tower?",
        generator_config=generator_config,
        sc_config=sc_cfg,
        client=client,
        _embedding_model=fake_embedding_model,
    )
    assert 0.3 < result.overall_consistency_score <= 1.0


@pytest.mark.parametrize("count", [3, 4, 5])
def test_sample_count_boundaries_accepted(monkeypatch, count):
    monkeypatch.setenv("TRUSTLENS_SC_SAMPLE_COUNT", str(count))
    cfg = SelfConsistencyConfig.from_env()
    assert cfg.sample_count == count


@pytest.mark.parametrize("count", [0, 1, 2, 6, 10])
def test_sample_count_outside_boundaries_rejected(monkeypatch, count):
    from trustlens_m1.config import ConfigError

    monkeypatch.setenv("TRUSTLENS_SC_SAMPLE_COUNT", str(count))
    with pytest.raises(ConfigError):
        SelfConsistencyConfig.from_env()


@pytest.mark.asyncio
async def test_all_samples_failing_raises(generator_config, fake_embedding_model):
    client = FakeGroqClient(fail_with=GenerationError, fail_times=99)
    # canonical call also fails -> should raise immediately from generate_answer
    with pytest.raises(GenerationError):
        await run_self_consistency(
            "Where is the Eiffel Tower?",
            generator_config=generator_config,
            client=client,
            _embedding_model=fake_embedding_model,
        )


def test_score_claim_consistency_malformed_input(fake_embedding_model):
    with pytest.raises(InvalidInputError):
        score_claim_consistency("", ["some answer"], _embedding_model=fake_embedding_model)
    with pytest.raises(InvalidInputError):
        score_claim_consistency("a claim", [], _embedding_model=fake_embedding_model)
    with pytest.raises(InvalidInputError):
        score_claim_consistency("a claim", "not a list", _embedding_model=fake_embedding_model)  # type: ignore[arg-type]


def test_score_claim_consistency_basic(fake_embedding_model):
    claim = "The Eiffel Tower is in Paris."
    samples = [
        "The Eiffel Tower is in Paris, a famous landmark. It was built in 1889.",
        "Paris is a city in France. The Eiffel Tower is located there.",
        "Completely unrelated sentence about volcanoes erupting in Iceland.",
    ]
    result = score_claim_consistency(claim, samples, _embedding_model=fake_embedding_model)
    assert result.sample_count == 3
    assert len(result.best_sentence_similarities) == 3
    assert 0.0 <= result.consistency_score <= 1.0
    assert result.support_count <= 3
