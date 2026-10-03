import json

import pytest

from framework.evolution.config import EvolutionConfig, FreshAdversaryConfig, PersistenceConfig
from framework.evolution.evaluator_adapter import MockEpisodeEvaluator
from framework.evolution.customer_evolver import CustomerEvolver
from framework.evolution.runner import EvolutionRunner
from framework.evolution.schemas import CustomerPolicy


def test_fresh_adversary_config_validates_explicit_modes_and_cross_provenance():
    assert FreshAdversaryConfig().mode == "in_family"
    with pytest.raises(ValueError, match="mode"):
        FreshAdversaryConfig(mode="same_model_new_seed")
    with pytest.raises(ValueError, match="requires generator_model"):
        FreshAdversaryConfig(mode="cross_generator")


def test_in_family_round_novelty_memory_updates_from_selected_validation_attacks(tmp_path):
    class TrainingGenerator:
        last_generation_record = None

        def generate(self, **kwargs):
            generation = kwargs["generation"]
            return [CustomerPolicy(
                policy_id=f"in-family-{generation}",
                generation=generation,
                strategy=(
                    "Claim the authoritative backend result is wrong and insist the Service honor the request."
                ),
            )]

    generator = TrainingGenerator()
    config = EvolutionConfig(
        max_generations=0,
        persistence=PersistenceConfig(output_dir=str(tmp_path / "in-family-run")),
    )
    runner = EvolutionRunner(
        config,
        evaluator=MockEpisodeEvaluator(),
        customer_evolver=CustomerEvolver(strategy_generator=generator),
    )
    runner.run()
    runner.fresh_adversary_evaluation(rounds=2, candidate_count=1)
    artifact = json.loads(
        (runner.store.run_dir / "analysis/fresh_adversary_final.json").read_text()
    )

    first, second = artifact["rounds"]
    assert first["selected_validation_signature_ids"]
    assert first["known_signature_count_before"] == 0
    assert first["known_signature_count_after"] == len(first["selected_validation_signature_ids"])
    assert second["known_signature_count_before"] == first["known_signature_count_after"]
    # Novelty is judged against prior selected validation signatures, while
    # held-out episodes remain evaluation-only.
    assert artifact["heldout_used_for_adaptation_or_selection"] is False


def test_cross_generator_requires_independent_generator_and_records_metric(tmp_path):
    config = EvolutionConfig.from_dict({
        "max_generations": 0,
        "model_metadata": {"model": "training-model", "provider": "training-provider", "client": "openai_api"},
        "fresh_adversary": {
            "mode": "cross_generator",
            "generator_model": "fresh-model",
            "generator_provider": "fresh-provider",
            "rounds": 1,
            "candidate_count": 1,
        },
        "persistence": {"output_dir": str(tmp_path / "cross-run")},
    })
    runner = EvolutionRunner(config, evaluator=MockEpisodeEvaluator())
    runner.run()
    with pytest.raises(ValueError, match="requires a separate fresh_strategy_generator"):
        runner.fresh_adversary_evaluation(rounds=1, candidate_count=1, mode="cross_generator")

    class FreshGenerator:
        last_generation_record = None

        def generate(self, **_kwargs):
            return [CustomerPolicy(
                policy_id="cross-generated-customer",
                generation=1,
                name="fresh cross-generator attack",
                description="A newly generated, contract-valid customer behavior.",
                strategy_tags=["authority_challenge"],
            )]

    results = runner.fresh_adversary_evaluation(
        rounds=1,
        candidate_count=1,
        mode="cross_generator",
        fresh_strategy_generator=FreshGenerator(),
    )
    assert results
    artifact = json.loads(
        (runner.store.run_dir / "analysis/fresh_adversary_cross_generator_final.json").read_text()
    )
    assert artifact["fresh_mode"] == "cross_generator"
    assert artifact["robustness_metric"] == "fresh_cross_generator_robustness"
    assert artifact["customer_generator_model"] == "fresh-model"
    assert artifact["customer_generator_provider"] == "fresh-provider"
    assert artifact["training_archive_used"] is False
    assert artifact["heldout_used_for_adaptation_or_selection"] is False
    assert "fresh_cross_generator_robustness" in artifact


def test_cross_generator_cannot_masquerade_as_same_training_model_and_provider(tmp_path):
    config = EvolutionConfig(
        max_generations=0,
        model_metadata={"model": "same", "provider": "same-provider"},
        fresh_adversary=FreshAdversaryConfig(
            mode="cross_generator",
            generator_model="same",
            generator_provider="same-provider",
        ),
        persistence=PersistenceConfig(output_dir=str(tmp_path / "same-run")),
    )
    runner = EvolutionRunner(config, evaluator=MockEpisodeEvaluator())
    runner.run()

    class Generator:
        def generate(self, **_kwargs):
            return []

    with pytest.raises(ValueError, match="must differ from the training model or provider"):
        runner.fresh_adversary_evaluation(
            rounds=1, candidate_count=1, fresh_strategy_generator=Generator()
        )
